"""GNN hétérogène flux / files WFQ / liens pour prédire la QoS par flux (« SliceGNN »).

Représentation d'une simulation (un graphe) :
  - nœuds « flux »  : une classe de flux (slice, débit offert, taille de paquet, δ, nb de sauts)
  - nœuds « file »  : une file WFQ active d'un port d'antenne (poids, réservation, charge)
  - nœuds « lien »  : un lien traversé par au moins un flux (capacité, charge offerte, type)
  - arêtes : flux → liens ORDONNÉES (le chemin), flux → sa file WFQ, file → port de l'antenne.

Passage de messages (inspiré de RouteNet, Rusek et al. 2020, et RouteNet-Fermi 2023),
répété T fois :
  1. chaque flux parcourt son chemin avec un GRU : à chaque saut il lit l'état du lien
     (et, au premier saut = port de l'antenne, l'état de sa file WFQ) ;
  2. chaque lien agrège (moyenne pondérée par w) les sorties du GRU des flux qui le traversent ;
  3. chaque file agrège de même les sorties des flux qu'elle sert ;
  4. liens et files mettent à jour leur état avec un GRUCell (mémoire entre itérations).
La lecture finale (MLP) transforme l'état de chaque flux en [log10 délai, taux de pertes].

Pourquoi un GRU le long du chemin plutôt qu'une agrégation de type GCN / GraphSAGE / GAT ?
  L'ordre des liens compte : un flux qui perd des paquets en amont charge moins les liens
  en aval. Un GRU lit la séquence dans l'ordre ; une somme ou une moyenne l'ignorerait.

Notations des tenseurs d'un lot (plusieurs graphes concaténés) :
  F = nombre de classes de flux, Q = nombre de files, L_ = nombre de liens,
  Lmax = longueur maximale d'un chemin dans le lot, d = dimension cachée.
"""
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Attributs d'entrée de chaque type de nœud (colonnes des tables de features)
FLOW_X = ["is_eMBB", "is_URLLC", "is_mMTC", "log_offered", "log_pkt_size", "delta", "hops"]
QUEUE_X = ["w_frac", "log_reserved", "rho_q", "rho_q_eff", "log_n_flows_q", "log_qdelay_full"]
LINK_X = ["log_cap", "rho", "log_n_flows_link", "is_port", "log_n_queues", "is_wfq", "log_full_delay"]


# ---------------------------------------------------------------------------
# Construction des graphes (une fois pour toutes, dans preparation.py)
# ---------------------------------------------------------------------------
def build_graphs(ft, links, queues):
    """Construit un dictionnaire {simulation: graphe}, chaque graphe étant un dict numpy :

      xf   (F, 7)     attributs des flux         xq (Q, 6)  attributs des files
      xl   (L_, 7)    attributs des liens        fq (F,)    indice de la file de chaque flux
      path (F, Lmax)  indices des liens du chemin, complétés par -1 (padding)
      lens (F,)       longueur réelle de chaque chemin
      y    (F, 2)     cibles [log10 délai, pertes]   w (F,) multiplicité   row (F,) identifiant

    ft     : classes de flux (features.deduplicate + preparation.event_mask)
    links  : table des liens avec la charge offerte (features.link_loads)
    queues : table des files (features.queue_features)
    """
    # --- attributs des files ---
    q = queues[queues.n_flows > 0].copy()
    q["w_frac"] = q.weight / 100                              # poids WFQ en fraction
    q["log_reserved"] = np.log10(q.reserved.clip(lower=1.0))
    q["log_n_flows_q"] = np.log10(q.n_flows)                  # cardinalité (la moyenne la perd)
    # délai de vidage d'une file pleine (32 paquets de la slice) au débit WFQ effectif
    pkt = ft.groupby(["sim", "antenna", "next_hop", "wfq_queue"]).pkt_size.first()
    q = q.join(pkt, on=["sim", "antenna", "next_hop", "queue"])
    q["log_qdelay_full"] = np.log10(32 * q.pkt_size.fillna(1000) / q.rate_eff)
    # --- attributs des liens ---
    links = links.copy()
    links["log_cap"] = np.log10(links.capacity)
    links["log_n_flows_link"] = np.log10(1 + links.n_flows_link)
    links["is_port"] = (links.src_type == "antenna").astype(float)   # port WFQ d'antenne ?
    links["log_n_queues"] = np.log10(links.n_queues)
    links["is_wfq"] = (links.sched == "WFQ").astype(float)
    links["log_full_delay"] = np.log10(1e-6 + links.full_delay)

    ft = ft.sort_values(["sim"]).reset_index(drop=True)
    graphs = {}
    q_by_sim = dict(tuple(q.groupby("sim")))
    l_by_sim = dict(tuple(links.groupby("sim")))
    for sim, fs in ft.groupby("sim", sort=False):
        qs = q_by_sim[sim]
        ls = l_by_sim[sim]
        # chemin sans le lien d'accès origine→antenne (dédié à un terminal, sans intérêt) :
        # le premier lien gardé est donc le port WFQ de l'antenne
        paths = [list(p[1:]) for p in fs.links]
        used = np.unique(np.concatenate(paths))               # liens réellement traversés
        lmap = pd.Series(np.arange(len(used)), index=used)    # identifiant global -> local
        ls = ls.set_index("link").loc[used]
        qkey = list(zip(qs.antenna, qs.next_hop, qs.queue))   # une file = (antenne, port, numéro)
        qmap = {k: i for i, k in enumerate(qkey)}
        fq = np.array([qmap[k] for k in zip(fs.antenna, fs.next_hop, fs.wfq_queue)])
        lens = np.array([len(p) for p in paths])
        path_idx = np.full((len(paths), lens.max()), -1, dtype=np.int64)  # -1 = case vide
        for i, p in enumerate(paths):
            path_idx[i, :len(p)] = lmap.loc[p].to_numpy()
        graphs[sim] = dict(
            xf=fs[FLOW_X].to_numpy(np.float32), xq=qs[QUEUE_X].to_numpy(np.float32),
            xl=ls[LINK_X].to_numpy(np.float32), path=path_idx, lens=lens, fq=fq,
            y=fs[["y_delay", "y_loss"]].to_numpy(np.float32), row=fs["row"].to_numpy(),
            w=fs["w"].to_numpy(np.float32))
    return graphs


class Normalizer:
    """Standardisation (moyenne 0, écart-type 1) des attributs et des cibles.

    Les statistiques sont calculées sur les simulations d'ENTRAÎNEMENT uniquement, puis
    appliquées telles quelles à la validation et au test (pas de fuite d'information).
    """

    def __init__(self, graphs, sims):
        self.stats = {}
        for k in ("xf", "xq", "xl"):
            x = np.concatenate([graphs[s][k] for s in sims])
            self.stats[k] = (x.mean(0), x.std(0) + 1e-6)      # +1e-6 : colonnes constantes
        y = np.concatenate([graphs[s]["y"] for s in sims])
        self.y_mean, self.y_std = np.nanmean(y, 0), np.nanstd(y, 0)

    def x(self, k, a):
        m, s = self.stats[k]
        return (a - m) / s


def collate(graphs, sims, norm, drop=None):
    """Concatène plusieurs graphes en un seul lot (« batch ») de tenseurs PyTorch.

    Les graphes sont mis bout à bout : les indices de liens et de files du graphe k sont
    décalés du nombre de liens / files des graphes précédents, pour pointer au bon endroit
    dans les tableaux concaténés. Les chemins sont complétés par -1 jusqu'à Lmax.

    drop : {"xf"|"xq"|"xl": [colonnes]} pour l'ablation. Les colonnes retirées sont mises
           à 0, c'est-à-dire à leur moyenne après standardisation : le modèle ne reçoit plus
           aucune information de ces attributs.
    """
    xf, xq, xl, path, lens, fq, y, rows, w = [], [], [], [], [], [], [], [], []
    nq = nl = 0                                               # décalages cumulés
    lmax = max(graphs[s]["path"].shape[1] for s in sims)
    for s in sims:
        g = graphs[s]
        xf.append(norm.x("xf", g["xf"]))
        xq.append(norm.x("xq", g["xq"]))
        xl.append(norm.x("xl", g["xl"]))
        p = np.full((len(g["path"]), lmax), -1, dtype=np.int64)
        p[:, :g["path"].shape[1]] = np.where(g["path"] >= 0, g["path"] + nl, -1)
        path.append(p)
        lens.append(g["lens"])
        fq.append(g["fq"] + nq)
        y.append(g["y"])
        rows.append(g["row"])
        w.append(g["w"])
        nq += len(g["xq"])
        nl += len(g["xl"])
    b = dict(xf=np.concatenate(xf), xq=np.concatenate(xq), xl=np.concatenate(xl),
             path=np.concatenate(path), lens=np.concatenate(lens), fq=np.concatenate(fq),
             y=(np.concatenate(y) - norm.y_mean) / norm.y_std, row=np.concatenate(rows),
             w=np.concatenate(w))
    if drop:  # ablation : colonnes mises à zéro (= valeur moyenne après standardisation)
        for k, cols in drop.items():
            b[k][:, cols] = 0.0
    return {k: torch.from_numpy(v) for k, v in b.items()}


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------
def mlp(i, h, o):
    """Petit perceptron à une couche cachée : encodeur d'attributs."""
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class SliceGNN(nn.Module):
    """GNN hétérogène de type RouteNet avec nœuds « file WFQ ».

    d          : dimension des états cachés (64)
    T          : nombre d'itérations de passage de messages (4). T = 0 (ablation) : une seule
                 lecture des chemins, sans mise à jour des liens ni des files, donc sans
                 interaction entre flux.
    use_queues : False (ablation) : le GNN ignore les files WFQ, donc les réservations par slice.
    """

    def __init__(self, d=64, T=4, use_queues=True):
        super().__init__()
        self.T, self.use_queues = T, use_queues
        # un encodeur par type de nœud : graphe hétérogène, attributs de natures différentes
        self.enc_f = mlp(len(FLOW_X), d, d)
        self.enc_q = mlp(len(QUEUE_X), d, d)
        self.enc_l = mlp(len(LINK_X), d, d)
        # GRU qui parcourt le chemin ; entrée = [état du lien, état de la file (1er saut)]
        self.path_rnn = nn.GRU(2 * d, d, batch_first=True)
        # mises à jour avec mémoire des liens et des files (comme les Gated GNN)
        self.upd_l = nn.GRUCell(d, d)
        self.upd_q = nn.GRUCell(d, d)
        # lecture : état final du flux -> 2 cibles standardisées
        self.read = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d // 2), nn.ReLU(),
                                  nn.Linear(d // 2, 2))

    def forward(self, b):
        # états initiaux : hf (F, d), hq (Q, d), hl (L_, d)
        hf, hq, hl = self.enc_f(b["xf"]), self.enc_q(b["xq"]), self.enc_l(b["xl"])
        path, lens = b["path"], b["lens"]                     # (F, Lmax), (F,)
        valid = path >= 0                                     # masque des vraies cases du chemin
        safe = path.clamp(min=0)                              # -1 remplacé par 0 pour l'indexation
        nF, L = path.shape
        d = hf.shape[1]
        # agrégations pondérées par la multiplicité w des classes de flux :
        # une classe de 10 flux compte 10 fois dans la moyenne d'un lien
        w = b["w"].unsqueeze(1)                               # (F, 1)
        w_path = w.expand(nF, L)[valid]                       # poids de chaque couple (flux, lien)
        dev = hf.device
        cnt_l = torch.zeros(len(hl), device=dev).index_add_(0, path[valid], w_path)  # Σw par lien
        cnt_q = torch.zeros(len(hq), device=dev).index_add_(0, b["fq"], b["w"])      # Σw par file
        for _ in range(max(self.T, 1)):
            # --- 1. chaque flux lit son chemin dans l'ordre --------------------------
            q_in = torch.zeros(nF, L, d, device=dev)
            if self.use_queues:
                q_in[:, 0] = hq[b["fq"]]  # position 0 du chemin = port WFQ de l'antenne
            seq = torch.cat([hl[safe] * valid.unsqueeze(-1), q_in], dim=-1)   # (F, Lmax, 2d)
            # pack_padded_sequence : le GRU s'arrête à la vraie longueur de chaque chemin
            packed = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False)
            out, h_last = self.path_rnn(packed, hf.unsqueeze(0))   # état initial = hf
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=L)  # (F, Lmax, d)
            hf = h_last[0]                                    # nouvel état du flux
            if self.T == 0:  # ablation : pas de rétroaction flux→liens
                break
            # --- 2. chaque lien agrège les flux qui le traversent (moyenne pondérée) ----
            # out[f, k] décrit le flux f tel qu'il arrive sur son k-ième lien
            m_l = (torch.zeros_like(hl).index_add_(0, path[valid], out[valid] * w_path.unsqueeze(1))
                   / cnt_l.clamp(min=1).unsqueeze(1))
            hl = self.upd_l(m_l, hl)
            # --- 3. chaque file agrège les flux qu'elle sert (sortie au premier saut) ----
            if self.use_queues:
                m_q = (torch.zeros_like(hq).index_add_(0, b["fq"], out[:, 0] * w)
                       / cnt_q.clamp(min=1).unsqueeze(1))
                hq = self.upd_q(m_q, hq)
        return self.read(hf)                                  # (F, 2) cibles standardisées


# ---------------------------------------------------------------------------
# Entraînement / évaluation
# ---------------------------------------------------------------------------
def masked_mse(pred, y, w):
    """MSE pondérée par w (= MSE par flux), en ignorant les cibles manquantes (NaN)."""
    m = ~torch.isnan(y)
    ww = w.unsqueeze(1).expand_as(y)[m]
    return (ww * (pred[m] - y[m]) ** 2).sum() / ww.sum()


def weighted_mae(p, y, w):
    """MAE pondérée par w, en ignorant les cibles manquantes."""
    m = ~np.isnan(y)
    return float(np.sum(w[m] * np.abs(p[m] - y[m])) / np.sum(w[m]))


def to_device(b, device):
    """Copie un lot sur le GPU si demandé (les longueurs de chemin restent sur CPU pour pack_padded)."""
    return {k: (v if k in ("lens", "row") else v.to(device)) for k, v in b.items()}


def predict(model, graphs, sims, norm, batch=32, drop=None, device="cpu"):
    """Prédictions remises à l'échelle d'origine, et identifiants de ligne correspondants."""
    model.eval()                                              # désactive le mode entraînement
    preds, rows = [], []
    with torch.no_grad():                                     # pas de gradient en prédiction
        for i in range(0, len(sims), batch):
            b = to_device(collate(graphs, sims[i:i + batch], norm, drop), device)
            preds.append(model(b).cpu().numpy() * norm.y_std + norm.y_mean)  # dé-standardisation
            rows.append(b["row"].numpy())
    return np.concatenate(preds), np.concatenate(rows)


def train(model, graphs, train_sims, val_sims, norm, epochs=30, batch=16, lr=1e-3, patience=6,
          seed=0, drop=None, log=print, device="cpu"):
    """Entraînement avec Adam et arrêt précoce sur la validation.

    Critère de validation : somme des MAE des deux cibles, chacune divisée par l'écart-type
    de sa cible (pour que délai et pertes pèsent autant). Contrôle du surapprentissage :
      - arrêt après `patience` époques sans amélioration, et restauration des meilleurs poids ;
      - division du taux d'apprentissage par 2 après 2 époques sans gain (ReduceLROnPlateau) ;
      - décroissance de poids 1e-5 et écrêtage de la norme du gradient à 1.
    Retourne l'historique (une ligne par époque).
    """
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    best, best_state, wait, hist = math.inf, None, 0, []
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        order = rng.permutation(train_sims)                   # nouvel ordre des graphes à chaque époque
        for i in range(0, len(order), batch):
            b = to_device(collate(graphs, order[i:i + batch], norm, drop), device)
            loss = masked_mse(model(b), b["y"], b["w"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # évite les pas de gradient extrêmes
            opt.step()
            tot += loss.item()
            n += 1
        # évaluation sur la validation (topologies jamais vues à l'entraînement)
        p, rows = predict(model, graphs, val_sims, norm, drop=drop, device=device)
        y = np.concatenate([graphs[s]["y"] for s in val_sims])
        wv = np.concatenate([graphs[s]["w"] for s in val_sims])
        mae = [weighted_mae(p[:, j], y[:, j], wv) for j in range(2)]
        err = np.array(mae) / norm.y_std
        val = float(err.sum())
        sched.step(val)
        hist.append(dict(epoch=ep + 1, train_mse=tot / n, val_mae_delay=mae[0], val_mae_loss=mae[1],
                         time=time.time() - t0))
        log(f"époque {ep + 1:02d}  train MSE {tot / n:.4f}  val MAE délai {hist[-1]['val_mae_delay']:.4f}"
            f"  perte {hist[-1]['val_mae_loss']:.4f}  ({hist[-1]['time']:.0f} s)")
        if val < best - 1e-4:                                 # amélioration significative
            best, wait = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)                         # meilleurs poids, pas les derniers
    return pd.DataFrame(hist)
