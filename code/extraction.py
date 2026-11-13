"""Extraction du dataset B5G Network Slicing (Farreras et al., Data in Brief 2024,
DOI 10.5281/zenodo.10610616) vers des tables Parquet.

Pourquoi un parseur maison ?
  Le dataset fournit datanetAPI.py, qui décode les fichiers JSON avec jsonpickle : cette
  bibliothèque peut instancier des objets Python arbitraires, donc exécuter du code contenu
  dans un fichier téléchargé. Nous avons lu datanetAPI.py et features.pdf pour comprendre
  les formats, puis écrit ce parseur qui n'utilise que json, tarfile et networkx.

Organisation des fichiers bruts (data/raw/2.2.1-10kprocessed/) :
  graphs/graph_n.txt      topologie GML de la simulation n (nœuds typés, files, poids WFQ)
  routings/routing_n.txt  matrice R[nœud courant][destination] = port de sortie
  slices/slices_n.json    slices (type, delta) et leurs flux (origine, destination, débit)
  results_*.tar.gz        100 simulations par archive ; chaque fichier interne contient une
                          ligne par simulation, dans le même ordre :
      simulationResults.txt  "globaux|cellule_0;cellule_1;..." matrice N×N à plat
                             (cellule src*N+dst = débit, paquets, pertes, délais du flux)
      traffic.txt            "maxAvgLambda|..." paramètres de trafic (même indexation)
      linkUsage.txt          utilisation et pertes de chaque port (cellule u*N+v), puis
                             statistiques de chaque file séparées par « : »
      stability.txt          durée simulée, critère de stabilité, statut OK / non OK
      input_files.txt        numéro de ligne ; graph_n ; routing_n ; slices_n

Sorties (data/processed/) :
  samples.parquet : une ligne par simulation (topologie, taille, mesures globales)
  flows.parquet   : une ligne par flux (slice, demande, réservation WFQ, chemin, QoS mesurée)
  links.parquet   : une ligne par lien orienté (capacité, utilisation et pertes mesurées)
  queues.parquet  : une ligne par file WFQ active d'un port d'antenne

Usage : python extraction.py [--workers 14] [--limit N]
"""
import argparse
import glob
import json
import os
import tarfile
from multiprocessing import Pool

import networkx as nx
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data", "raw", "2.2.1-10kprocessed"))
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "data", "processed"))

# Ordre des 11 valeurs d'une cellule de simulationResults.txt (voir features.pdf) :
# débit mesuré (kbit/s), paquets générés/s, paquets perdus/s, délai moyen (s),
# moyenne de ln(délai), percentiles 10-20-50-80-90 du délai, gigue (variance du délai).
PERF_KEYS = ["meas_bw_kbps", "pkts_gen", "pkts_drop", "delay", "ln_delay",
             "p10", "p20", "p50", "p80", "p90", "jitter"]


def parse_queues(node_attr, port):
    """Files WFQ d'un port de sortie : liste de (poids en %, ensemble des ToS servis).

    Deux attributs du nœud décrivent l'ordonnancement, avec des conventions différentes :
      - tosToQoSqueue est défini PAR NŒUD : « ; » sépare les files, « , » les ToS d'une file.
        Exemple « 36;35,40,42;37 » : la file 0 sert le ToS 36, la file 1 les ToS 35, 40, 42…
      - schedulingWeights est défini PAR PORT : « ; » sépare les ports, « , » les files.
        Il faut donc prendre le bloc numéro `port` (attribut « port » du lien sortant).
    Un poids est un pourcentage de la capacité du port réservé à la file (somme ≈ 100).
    """
    n_q = int(node_attr["levelsQoS"])                       # nombre de files du nœud
    tos_groups = [set(int(x) for x in grp.split(",") if x != "")
                  for grp in str(node_attr["tosToQoSqueue"]).split(";")][:n_q]
    per_port = str(node_attr.get("schedulingWeights", "-")).split(";")
    raw_w = per_port[port] if port < len(per_port) else per_port[0]
    # quelques poids valent -0,01 à -0,03 (résidus d'arrondi du générateur) : ramenés à 0
    weights = [max(float(x), 0.0) for x in raw_w.split(",") if x not in ("", "-")]
    weights = (weights + [0.0] * n_q)[:n_q]                 # complète si la liste est courte
    return list(zip(weights, tos_groups))


def pkt_size_mean(t):
    """Taille moyenne réelle des paquets (bits) à partir de la chaîne de trafic d'un flux.

    Format d'une chaîne (champs séparés par des virgules) :
      code de loi d'inter-arrivée, ses paramètres, code de loi de taille, ses paramètres, ToS.
    `off` est la position du code de loi de taille ; elle dépend du nombre de paramètres de
    la loi d'inter-arrivée (0 = exponentielle : 3 paramètres, donc taille en position 4…).

    Pour la loi binomiale (code 2), le simulateur tire PktSize1 ou PktSize2 avec
    probabilité 1/2 : la moyenne est (PktSize1 + PktSize2) / 2. Le champ « AvgPktSize »
    ne vaut pas toujours cette moyenne (eMBB : 6000 au lieu de 15000 bits) ; le débit
    mesuré (22,5 Mb/s pour 1500 paquets/s) confirme la valeur 15000.
    """
    off = {"0": 4, "1": 3, "2": 4, "3": 4, "4": 6, "5": 8}[t[0]]
    if t[off] == "2":                                       # binomiale : deux tailles
        return (float(t[off + 2]) + float(t[off + 3])) / 2
    return float(t[off + 1])                                # autres lois : taille moyenne


def routing_paths(G, routing_file):
    """Fonction chemin(src, dst) à partir du fichier de routage réellement simulé.

    Le fichier contient une matrice R[nœud courant][destination] = port de sortie. Le chemin
    s'obtient en suivant les ports depuis la source jusqu'à la destination.
    ATTENTION : le champ « path » des fichiers de slices ne correspond pas à ce routage pour
    42 % des flux (76 % des simulations en contiennent au moins un). Seul le routage décrit
    le trafic simulé, comme dans datanetAPI.py ; le contrôle « charge offerte calculée /
    utilisation mesurée » (corrélation 0,98) le confirme.
    """
    with open(routing_file) as fh:
        R = [[int(x) for x in line.strip().rstrip(",").split(",")] for line in fh if line.strip()]
    # port2nei[u][p] = voisin atteint depuis u par le port p
    port2nei = {u: {int(G.edges[u, v, 0]["port"]): v for v in G.successors(u)} for u in G}

    def path(src, dst):
        p = [src]
        while p[-1] != dst:
            p.append(port2nei[p[-1]][R[p[-1]][dst]])
            if len(p) > len(R):                             # sécurité : pas de boucle infinie
                raise ValueError(f"boucle de routage {src}->{dst}")
        return p
    return path


def parse_sample(root, lines):
    """Transforme les 5 lignes d'une simulation en dictionnaires (simulation, flux, liens, files)."""
    res_line, traf_line, stab_line, inp_line, lu_line = lines
    sim_id, graph_f, routing_f, slices_f = inp_line.split(";")[:4]
    G = nx.read_gml(os.path.join(root, "graphs", graph_f), destringizer=int)
    N = G.number_of_nodes()
    ntype = {n: G.nodes[n]["type"] for n in G}   # origin_*, antenna, inner, destination_*

    # Chaque ligne = en-tête global, « | », puis N×N cellules séparées par « ; »
    head, res = res_line.split("|", 1)
    g_pkts, g_loss, g_delay = map(float, head.split(","))
    res = res.rstrip(";").split(";")
    mal, traf = traf_line.split("|", 1)
    traf = traf.rstrip(";").split(";")
    lu = lu_line.rstrip(";").split(";")
    assert len(res) == N * N == len(traf) == len(lu), (sim_id, N, len(res), len(traf), len(lu))
    stab = stab_line.split(";")

    with open(os.path.join(root, "slices", slices_f)) as fh:
        slices = json.load(fh)                    # JSON simple : json.load suffit (pas jsonpickle)
    route = routing_paths(G, os.path.join(root, "routings", routing_f))

    # --- liens : capacité (entrée) et mesures de sortie du port u -> v ----------------
    link_rows, link_id = [], {}
    for (u, v, _k) in G.edges(keys=True):
        e = G.edges[u, v, 0]
        cell = lu[u * N + v]                      # « -1 » : port sans statistique
        util = loss = pkt = float("nan")
        if cell != "-1":
            f = cell.split(":")[0].split(",")     # 1er bloc = port entier ; suivants = files
            util, loss = float(f[0]), float(f[1])
            pkt = float(f[2]) if f[2] not in ("-nan", "nan") else float("nan")
        link_id[(u, v)] = len(link_rows)
        link_rows.append(dict(link=len(link_rows), src=u, dst=v, src_type=ntype[u], dst_type=ntype[v],
                              capacity=float(e["bandwidth"]), meas_util=util, meas_loss=loss,
                              meas_pkt_size=pkt, n_queues=int(G.nodes[u]["levelsQoS"]),
                              sched=G.nodes[u]["schedulingPolicy"]))

    # --- flux : entrées (slice, trafic, chemin) et QoS mesurée -------------------------
    flow_rows = []
    for sl in slices:
        for fl in sl["flows"]:
            s, d = int(fl["origin_node"]), int(fl["destination"])
            slice_path = json.loads(fl["path"]) if isinstance(fl["path"], str) else fl["path"]
            path = route(s, d)                    # chemin réellement simulé (routage)
            t = traf[s * N + d].split(",")        # paramètres de trafic du flux
            perf = list(map(float, res[s * N + d].split(",")))
            row = dict(src=s, dst=d, antenna=path[1],          # le 2e nœud est l'antenne
                       slice_type=sl["type"], slice_id=int(sl["number"]), delta=float(sl["delta"]),
                       demand_bw=float(fl["bandwidth"]), tos=int(t[-1]),   # ToS = dernier champ
                       pkt_size=pkt_size_mean(t),
                       time_dist=int(t[0]), avg_pkts_lambda=float(t[2]),
                       path=[int(x) for x in path],
                       slice_path_ok=[int(x) for x in slice_path] == path,  # pour l'EDA
                       links=[link_id[(a, b)] for a, b in zip(path[:-1], path[1:])])
            row.update(dict(zip(PERF_KEYS, perf)))
            flow_rows.append(row)

    # --- files WFQ des antennes : à quelle file (donc quel poids) appartient chaque flux ?
    # Un flux est servi, au port de son antenne vers le saut suivant, par la file dont
    # l'ensemble de ToS contient son ToS.
    tos_to_flow = {}
    for i, r in enumerate(flow_rows):
        tos_to_flow.setdefault((r["antenna"], r["tos"]), []).append(i)
    queue_rows = []
    for a in (n for n in G if ntype[n] == "antenna"):
        for v in G.successors(a):
            cell = lu[a * N + v]
            if cell == "-1" or ntype[v].startswith("origin"):   # ports vers les terminaux : inutiles
                continue
            queues = parse_queues(G.nodes[a], int(G.edges[a, v, 0]["port"]))
            qstats = cell.split(":")[1:]                         # statistiques par file
            for q, (w, tos_set) in enumerate(queues):
                # flux de cette antenne, de ToS servi par la file q, sortant par le port a -> v
                fids = [i for tos in tos_set for i in tos_to_flow.get((a, tos), [])
                        if flow_rows[i]["path"][2 if len(flow_rows[i]["path"]) > 2 else 1] == v]
                if not fids:                                     # file sans trafic : ignorée
                    continue
                qs = list(map(float, qstats[q].split(","))) if q < len(qstats) else [float("nan")] * 5
                types = {flow_rows[i]["slice_type"] for i in fids}
                sids = {flow_rows[i]["slice_id"] for i in fids}
                queue_rows.append(dict(antenna=a, next_hop=v, queue=q, weight=w,
                                       capacity=link_rows[link_id[(a, v)]]["capacity"],
                                       n_flows=len(fids),
                                       slice_type=types.pop() if len(types) == 1 else ("mixed" if types else "none"),
                                       slice_id=sids.pop() if len(sids) == 1 else -1,
                                       demand=sum(flow_rows[i]["demand_bw"] for i in fids),
                                       # débit offert réel = paquets/s × taille moyenne réelle
                                       offered=sum(flow_rows[i]["avg_pkts_lambda"] * flow_rows[i]["pkt_size"]
                                                   for i in fids),
                                       # mesures de sortie (analyse seulement, jamais features)
                                       meas_util=qs[0], meas_loss=qs[1], avg_occ=qs[2], max_occ=qs[3]))
                for i in fids:
                    flow_rows[i]["wfq_queue"] = q
                    flow_rows[i]["wfq_weight"] = w

    # Le numéro de ligne de input_files.txt recommence à 0 dans chaque archive :
    # l'identifiant global de la simulation est le numéro du fichier graph_n.txt.
    sim_num = int(graph_f.split("_")[1].split(".")[0])
    sample = dict(sim=sim_num, line=int(sim_id), topology=str(G.graph.get("label", G.graph.get("Network", "?"))),
                  n_nodes=N, n_links=G.number_of_edges(), n_flows=len(flow_rows),
                  n_slices=len(slices), n_antennas=sum(t == "antenna" for t in ntype.values()),
                  global_packets=g_pkts, global_losses=g_loss, global_delay=g_delay,
                  max_avg_lambda=float(mal), sim_time=float(stab[0]), stability=float(stab[1]))
    return sample, flow_rows, link_rows, queue_rows


def process_tar(path):
    """Lit une archive (100 simulations) en avançant ligne par ligne dans ses 5 fichiers."""
    root = os.path.dirname(path)
    out = dict(samples=[], flows=[], links=[], queues=[], errors=[])
    with tarfile.open(path, "r:gz") as tar:
        d = tar.getnames()[0]                                    # dossier interne de l'archive
        files = ["simulationResults.txt", "traffic.txt", "stability.txt", "input_files.txt", "linkUsage.txt"]
        handles = [tar.extractfile(f"{d}/{f}") for f in files]
        while True:
            # la k-ième ligne de chaque fichier décrit la même simulation
            lines = [h.readline().decode().rstrip("\n") for h in handles]
            if not lines[0] or not lines[1]:                     # fin de l'archive
                break
            if ";OK;" not in lines[2]:                           # régime stationnaire non atteint
                out["errors"].append((os.path.basename(path), lines[3], "not OK"))
                continue
            try:
                sample, flows, links, queues = parse_sample(root, lines)
            except Exception as exc:  # on garde la trace, sans arrêter l'extraction
                out["errors"].append((os.path.basename(path), lines[3], repr(exc)))
                continue
            sim = sample["sim"]
            out["samples"].append(sample)
            for tbl, rows in (("flows", flows), ("links", links), ("queues", queues)):
                for r in rows:
                    r["sim"] = sim                               # clé de jointure entre tables
                out[tbl].extend(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0, help="nombre d'archives (0 = toutes)")
    args = ap.parse_args()
    tars = sorted(glob.glob(os.path.join(DATA_DIR, "*.tar.gz")))
    if args.limit:
        tars = tars[:args.limit]
    os.makedirs(OUT_DIR, exist_ok=True)
    acc = dict(samples=[], flows=[], links=[], queues=[], errors=[])
    # une archive par processus : les archives sont indépendantes
    with Pool(args.workers) as pool:
        for k, out in enumerate(pool.imap_unordered(process_tar, tars), 1):
            for key in acc:
                acc[key].extend(out[key])
            print(f"{k}/{len(tars)} archives, {len(acc['samples'])} simulations", flush=True)
    for key in ("samples", "flows", "links", "queues"):
        df = pd.DataFrame(acc[key])
        df.to_parquet(os.path.join(OUT_DIR, f"{key}.parquet"), index=False)
        print(key, df.shape)
    pd.DataFrame(acc["errors"], columns=["archive", "input", "error"]).to_csv(
        os.path.join(OUT_DIR, "extraction_errors.csv"), index=False)
    print("erreurs :", len(acc["errors"]))


if __name__ == "__main__":
    main()
