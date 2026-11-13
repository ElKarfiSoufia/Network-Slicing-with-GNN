"""Construction des features par flux à partir des tables extraites (extraction.py).

Règle de base (absence de fuite d'information) : toutes les features sont calculées à
partir des ENTRÉES de la simulation (topologie, routage, demandes de trafic, poids WFQ).
Aucune mesure de sortie (utilisation, pertes, délais mesurés) n'est utilisée comme
feature : ces mesures servent uniquement de cibles ou d'analyse exploratoire. Un modèle
qui les utiliserait serait inutilisable pour évaluer une configuration jamais simulée.

Notations utilisées dans les commentaires :
  ρ (rho)   charge offerte / capacité d'une ressource (lien, port, file WFQ)
  b         charge du goulot du flux = max(ρ de sa file WFQ, ρ des liens de son chemin)
  B = 32    taille des files en paquets (constante dans tout le dataset)
"""
import os

import numpy as np
import pandas as pd

PROCESSED = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "data", "processed"))
SLICES = ["eMBB", "URLLC", "mMTC"]

# Features tabulaires utilisées par les modèles sans graphe, regroupées par famille.
# Les familles servent à l'ablation (run_tabular.DROP retire une famille entière).
FEATURE_GROUPS = {
    # ce que la slice injecte : type, débit réel, taille moyenne des paquets
    "slice": ["is_eMBB", "is_URLLC", "is_mMTC", "log_offered", "log_pkt_size"],
    # le levier de l'opérateur : δ, poids WFQ, débit réservé et charges de la file
    "reservation": ["delta", "wfq_weight", "log_reserved", "rho_q", "rho_q_eff"],
    # le port de sortie de l'antenne partagé par les slices
    "antenna_port": ["log_port_cap", "rho_port", "n_queues_port", "n_flows_q"],
    # résumés à la main des liens de transport du chemin (= « graphe résumé »)
    "path": ["hops", "rho_path_max", "rho_path_mean", "n_over_path", "log_cap_path_min"],
    # taille du réseau simulé
    "network": ["n_nodes", "n_flows_sim", "n_antennas"],
    # modèle fluide : temps de vidage d'une file pleine (connaissance du domaine)
    "physique": ["log_qdelay_full", "log_path_delay_full"],
}
BUFFER = 32  # taille des files (paquets), identique pour tous les nœuds du dataset
FEATURES = [c for cols in FEATURE_GROUPS.values() for c in cols]   # 24 features
TARGETS = ["y_delay", "y_loss"]   # log10(délai moyen en s), taux de pertes dans [0, 1]


def load_raw(path=PROCESSED):
    """Charge les 4 tables produites par extraction.py."""
    read = lambda name: pd.read_parquet(os.path.join(path, f"{name}.parquet"))
    return read("samples"), read("flows"), read("links"), read("queues")


def link_loads(flows, links):
    """Charge offerte par lien (somme des débits offerts des flux qui le traversent).

    Colonnes ajoutées à la table des liens :
      off           débit offert total (bit/s) des flux dont le chemin contient le lien
      n_flows_link  nombre de flux qui traversent le lien
      rho           off / capacité : > 1 signifie que le lien est surchargé
      full_delay    temps pour vider une file FIFO pleine (B paquets de la taille moyenne
                    des paquets du lien) à la capacité du lien : délai d'attente d'un lien
                    saturé dans un modèle fluide (voir chapitre 1, file M/M/1/B).
    """
    # « explode » : une ligne par couple (flux, lien traversé)
    ex = flows[["sim", "links", "offered", "avg_pkts_lambda"]].explode("links")
    ex["links"] = ex["links"].astype(np.int64)
    lo = (ex.groupby(["sim", "links"]).agg(off=("offered", "sum"), n_flows_link=("offered", "size"),
                                           pkts=("avg_pkts_lambda", "sum"))
            .reset_index().rename(columns={"links": "link"}))
    links = links.merge(lo, on=["sim", "link"], how="left")
    # liens traversés par aucun flux : charge nulle
    links[["off", "n_flows_link", "pkts"]] = links[["off", "n_flows_link", "pkts"]].fillna(0)
    links["rho"] = links["off"] / links["capacity"]
    mean_pkt = (links.off / links.pkts.where(links.pkts > 0)).fillna(0)   # bits / paquet
    links["full_delay"] = BUFFER * mean_pkt / links.capacity
    return links


def queue_features(queues):
    """Charge des files WFQ d'antenne : offerte / réservée (poids nominal) et effective.

    WFQ approche le Generalized Processor Sharing (Parekh et Gallager, 1993), qui est
    conservatif : la capacité des files sans trafic est redistribuée aux files actives au
    prorata de leurs poids. Le débit garanti d'une file active vaut donc
        r_eff = w / somme(w des files actives) × C_port,
    et non w/100 × C_port. rho_q_eff utilise cette part effective.
    """
    q = queues.copy()
    port = q.groupby(["sim", "antenna", "next_hop"])          # un groupe = un port d'antenne
    q["w_active"] = port.weight.transform("sum")              # somme des poids actifs du port
    q["port_offered"] = port.offered.transform("sum")         # charge totale du port
    q["n_queues_port"] = port.queue.transform("size")         # nombre de slices sur le port
    q["reserved"] = q.weight / 100 * q.capacity               # débit réservé nominal (bit/s)
    q["rho_q"] = q.offered / q.reserved.clip(lower=1.0)       # charge / réservation nominale
    q["rho_q_eff"] = q.offered / (q.weight / q.w_active.clip(lower=1e-9) * q.capacity).clip(lower=1.0)
    q["rho_port"] = q.port_offered / q.capacity               # charge du port entier
    q["rate_eff"] = (q.weight / q.w_active.clip(lower=1e-9) * q.capacity).clip(lower=1.0)
    return q


def build_flow_table(samples, flows, links, queues):
    """Table d'apprentissage : une ligne par flux, avec features, cibles et métadonnées.

    Retourne (table des flux, table des liens enrichie, table des files enrichie).
    """
    f = flows.copy()
    f["offered"] = f.avg_pkts_lambda * f.pkt_size             # débit offert réel (bit/s)
    links = link_loads(f, links)
    q = queue_features(queues)

    # 1) rattacher à chaque flux les caractéristiques de sa file WFQ d'antenne
    qcols = ["sim", "antenna", "next_hop", "queue", "reserved", "rho_q", "rho_q_eff", "rho_port",
             "n_queues_port", "capacity", "n_flows", "rate_eff"]
    f["next_hop"] = f.path.str[2]  # nœud après l'antenne : identifie le port de sortie WFQ
    f = f.merge(q[qcols].rename(columns={"queue": "wfq_queue", "capacity": "port_cap",
                                         "n_flows": "n_flows_q"}),
                on=["sim", "antenna", "next_hop", "wfq_queue"], how="left", validate="many_to_one")

    # 2) agrégats sur les liens de transport du chemin (après le port de l'antenne)
    ex = f[["sim", "links"]].reset_index().explode("links")
    ex["pos"] = ex.groupby("index").cumcount()                # position du lien dans le chemin
    ex = ex[ex.pos >= 2]  # 0 : origine→antenne (accès dédié), 1 : port WFQ de l'antenne
    ex["links"] = ex["links"].astype(np.int64)
    lk = links.set_index(["sim", "link"])[["rho", "capacity", "full_delay"]]
    ex = ex.join(lk, on=["sim", "links"])
    ex["sat_delay"] = np.where(ex.rho > 1, ex.full_delay, 0.0)  # seuls les liens saturés comptent
    agg = ex.groupby("index").agg(rho_path_max=("rho", "max"), rho_path_mean=("rho", "mean"),
                                  n_over_path=("rho", lambda x: int((x > 1).sum())),
                                  cap_path_min=("capacity", "min"), path_delay_full=("sat_delay", "sum"))
    f = f.join(agg)
    # chemin sans lien de transport (destination directement après l'antenne) : valeurs neutres
    f[["rho_path_max", "rho_path_mean", "n_over_path"]] = f[["rho_path_max", "rho_path_mean",
                                                             "n_over_path"]].fillna(0)
    f["cap_path_min"] = f.cap_path_min.fillna(f.port_cap)
    f["path_delay_full"] = f.path_delay_full.fillna(0)

    # 3) features « physiques » (modèle fluide d'une file pleine de B paquets)
    #    délai de file pleine à l'antenne = B × taille de paquet / débit WFQ effectif
    f["log_qdelay_full"] = np.log10(BUFFER * f.pkt_size / f.rate_eff)
    f["log_path_delay_full"] = np.log10(1e-6 + f.path_delay_full)   # 1e-6 s : évite log(0)

    # 4) encodages simples : one-hot du type de slice et logarithmes des grandeurs à forte
    #    dynamique (débits, capacités), pour que les modèles linéaires les traitent bien
    for s in SLICES:
        f[f"is_{s}"] = (f.slice_type == s).astype(float)
    f["log_offered"] = np.log10(f.offered)
    f["log_pkt_size"] = np.log10(f.pkt_size)
    f["log_reserved"] = np.log10(f.reserved.clip(lower=1.0))
    f["log_port_cap"] = np.log10(f.port_cap)
    f["log_cap_path_min"] = np.log10(f.cap_path_min)
    f["hops"] = f.links.str.len()
    f = f.merge(samples[["sim", "topology", "n_nodes", "n_antennas"]], on="sim")
    f["n_flows_sim"] = f.groupby("sim").src.transform("size")

    # 5) cibles : log10 du délai (8 ordres de grandeur) et taux de pertes
    f["loss"] = f.pkts_drop / f.pkts_gen
    # délai = -1 quand tous les paquets sont perdus : cible délai manquante (NaN)
    f["y_delay"] = np.where(f.delay > 0, np.log10(f.delay.clip(lower=1e-12)), np.nan)
    f["y_loss"] = f.loss.clip(0, 1)
    return f, links, q


# Deux flux ayant la même valeur pour toutes ces colonnes ont exactement les mêmes entrées.
CLASS_KEY = ["sim", "path_key", "antenna", "next_hop", "wfq_queue", "slice_type", "delta",
             "offered", "pkt_size"]


def deduplicate(f):
    """Regroupe les flux aux entrées strictement identiques dans une même simulation.

    Deux flux d'une même slice, partant de la même antenne, servis par la même file WFQ
    et suivant le même chemin après l'antenne ne diffèrent que par leur nœud d'origine
    (lien d'accès dédié, très peu chargé). Leurs entrées sont identiques et leurs QoS
    mesurées quasi égales (écart-type de 0,01 décade). On garde une ligne par classe avec
    son poids w (nombre de flux) et la moyenne des cibles. Les métriques pondérées par w
    sont alors exactement les métriques par flux, pour un coût de calcul divisé par ~6.
    """
    f = f.copy()
    f["path_key"] = f.links.apply(lambda x: tuple(x[1:]))   # chemin sans le lien d'accès
    g = f.groupby(CLASS_KEY, sort=False)
    out = g.first()                                          # entrées communes de la classe
    out["w"] = g.size()                                      # multiplicité = nombre de flux
    out[["y_delay", "y_loss", "delay", "loss"]] = g[["y_delay", "y_loss", "delay", "loss"]].mean()
    out["y_delay_std"] = g.y_delay.std().fillna(0)           # contrôle : dispersion intra-classe
    return out.reset_index().drop(columns="path_key")
