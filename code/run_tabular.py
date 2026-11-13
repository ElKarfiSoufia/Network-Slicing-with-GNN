"""Modèles sans graphe : baseline naïve, Ridge, Gradient Boosting, MLP.

Chaque modèle prédit les deux cibles (log10 délai, taux de perte) d'une classe de flux à
partir des features tabulaires de features.FEATURES. Les hyperparamètres sont choisis sur
la validation ; l'apprentissage et les métriques sont pondérés par w (= par flux).

Rôle de chaque modèle dans la démarche :
  naive : référence minimale (ce que l'on obtient sans rien apprendre de la configuration)
  ridge : modèle linéaire régularisé (Hoerl et Kennard, 1970) : teste si une relation
          linéaire suffit (non : la QoS a un coude en b ≈ 1)
  hgb   : boosting de gradient sur histogrammes (Friedman, 2001 ; principe de LightGBM) :
          le modèle tabulaire non linéaire le plus fort, référence à battre pour le GNN
  mlp   : réseau dense sur les mêmes features : « GNN sans graphe »

Chaque fonction fit_* reçoit (train, val, test, liste de features) et renvoie
({cible: [prédictions val, prédictions test]}, hyperparamètres retenus).

Usage : python run_tabular.py [--split topo|shift] [--drop reservation|charge|physique|allocation]
                              [--models naive,ridge,hgb,mlp] [--force]
"""
import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES, PROCESSED, TARGETS
from splits import make_splits

RESULTS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))

# familles retirées pour l'ablation (alignées sur run_gnn.DROP)
DROP = {
    "reservation": ["delta", "wfq_weight", "log_reserved", "rho_q", "rho_q_eff"],
    "charge": ["rho_q", "rho_q_eff", "rho_port", "n_flows_q", "rho_path_max", "rho_path_mean", "n_over_path"],
    "physique": ["log_qdelay_full", "log_path_delay_full"],
}
DROP["allocation"] = DROP["reservation"] + DROP["physique"]  # aucune information de réservation


def wmae(p, y, w):
    """MAE pondérée par w (une classe de w flux compte w fois), cibles manquantes ignorées."""
    m = ~np.isnan(y)
    return float(np.sum(w[m] * np.abs(p[m] - y[m])) / np.sum(w[m]))


def fit_naive(tr, va, te, feats):
    """Médiane (pondérée par w) de chaque cible par type de slice, apprise sur le train."""
    out = {}
    for t in TARGETS:
        # répétition w fois : médiane calculée par flux et non par classe
        rep = tr.loc[tr.index.repeat(tr.w), ["slice_type", t]].dropna()
        med = rep.groupby("slice_type")[t].median()
        out[t] = [d.slice_type.map(med).to_numpy() for d in (va, te)]
    return out, {}


def fit_ridge(tr, va, te, feats):
    """Ridge sur features standardisées ; α choisi sur la validation (une valeur par cible)."""
    out, best = {}, {}
    for t in TARGETS:
        m_tr = tr[t].notna()                       # lignes dont la cible est connue
        scores = {}
        for a in (0.01, 0.1, 1, 10, 100):          # force de la régularisation L2
            model = make_pipeline(StandardScaler(), Ridge(alpha=a))
            model.fit(tr.loc[m_tr, feats], tr.loc[m_tr, t], ridge__sample_weight=tr.loc[m_tr, "w"])
            scores[a] = (wmae(model.predict(va[feats]), va[t].to_numpy(), va.w.to_numpy()), model)
        a = min(scores, key=lambda k: scores[k][0])   # meilleur α = plus petite MAE de validation
        best[t] = {"alpha": a}
        out[t] = [scores[a][1].predict(d[feats]) for d in (va, te)]
    return out, best


def fit_hgb(tr, va, te, feats):
    """Gradient Boosting : grille de 8 configurations, choix sur la validation (par cible).

    Grille : taux d'apprentissage {0,05 ; 0,1} × feuilles max {31 ; 127} × taille minimale
    d'une feuille {20 ; 200}. 400 arbres, régularisation L2 = 1.
    """
    out, best = {}, {}
    grid = list(itertools.product((0.05, 0.1), (31, 127), (20, 200)))
    for t in TARGETS:
        m_tr = tr[t].notna()
        scores = {}
        for lr, leaves, msl in grid:
            model = HistGradientBoostingRegressor(learning_rate=lr, max_iter=400, max_leaf_nodes=leaves,
                                                  min_samples_leaf=msl, l2_regularization=1.0,
                                                  random_state=0)
            model.fit(tr.loc[m_tr, feats], tr.loc[m_tr, t], sample_weight=tr.loc[m_tr, "w"])
            scores[(lr, leaves, msl)] = (wmae(model.predict(va[feats]), va[t].to_numpy(), va.w.to_numpy()), model)
        k = min(scores, key=lambda c: scores[c][0])
        best[t] = dict(learning_rate=k[0], max_leaf_nodes=k[1], min_samples_leaf=k[2])
        out[t] = [scores[k][1].predict(d[feats]) for d in (va, te)]
    return out, best


def fit_mlp(tr, va, te, feats, epochs=60, patience=6, seed=0):
    """MLP 2 couches cachées (128) multi-sortie, MSE pondérée, arrêt précoce sur la val."""
    torch.manual_seed(seed)
    sc = StandardScaler().fit(tr[feats])           # standardisation apprise sur le train
    # cibles standardisées : délai et pertes pèsent autant dans la perte
    ym, ys = np.nanmean(tr[TARGETS].to_numpy(), 0), np.nanstd(tr[TARGETS].to_numpy(), 0)
    X = lambda d: torch.tensor(sc.transform(d[feats]), dtype=torch.float32)
    Y = lambda d: torch.tensor((d[TARGETS].to_numpy() - ym) / ys, dtype=torch.float32)
    xtr, ytr, wtr = X(tr), Y(tr), torch.tensor(tr.w.to_numpy(), dtype=torch.float32)
    xva = X(va)
    net = nn.Sequential(nn.Linear(len(feats), 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                        nn.Linear(128, 2))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    best, best_state, wait = np.inf, None, 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xtr))            # mélange à chaque époque, lots de 1024
        for i in range(0, len(perm), 1024):
            idx = perm[i:i + 1024]
            y, w = ytr[idx], wtr[idx].unsqueeze(1).expand(-1, 2)
            m = ~torch.isnan(y)
            loss = (w[m] * (net(xtr[idx])[m] - y[m]) ** 2).sum() / w[m].sum()   # MSE pondérée, NaN ignorés
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            p = net(xva).numpy() * ys + ym
        val = sum(wmae(p[:, j], va[t].to_numpy(), va.w.to_numpy()) / ys[j] for j, t in enumerate(TARGETS))
        if val < best - 1e-4:                      # arrêt précoce (Prechelt, 1998)
            best, wait = val, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)                # restaure les meilleurs poids
    net.eval()
    with torch.no_grad():
        pv, pt = (net(X(d)).numpy() * ys + ym for d in (va, te))
    return {t: [pv[:, j], pt[:, j]] for j, t in enumerate(TARGETS)}, {"hidden": [128, 128]}


FITTERS = {"naive": fit_naive, "ridge": fit_ridge, "hgb": fit_hgb, "mlp": fit_mlp}


def run(split="topo", drop=None, models=("naive", "ridge", "hgb", "mlp"), force=False, log=print):
    """Entraîne les modèles demandés et sauvegarde leurs prédictions (val et test).

    Si les prédictions existent déjà et force=False, le modèle n'est pas ré-entraîné.
    Retourne la liste des noms de résultats (ex. « hgb_shift_no_charge »).
    """
    os.makedirs(RESULTS, exist_ok=True)
    suffix = ("" if split == "topo" else f"_{split}") + (f"_no_{drop}" if drop else "")
    todo = [m for m in models
            if force or not os.path.exists(os.path.join(RESULTS, f"{m}{suffix}_preds.parquet"))]
    if todo:
        ft = pd.read_parquet(os.path.join(PROCESSED, "flow_classes.parquet"),
                             columns=["row", "sim", "slice_type", "w"] + FEATURES + TARGETS)
        samples = pd.read_parquet(os.path.join(PROCESSED, "samples.parquet"))
        sp = make_splits(samples)[split]
        tr, va, te = (ft[ft.sim.isin(sp[k])].reset_index(drop=True) for k in ("train", "val", "test"))
        feats = [c for c in FEATURES if c not in DROP.get(drop, [])]
    for name in todo:
        t0 = time.time()
        out, params = FITTERS[name](tr, va, te, feats)
        rows = []
        for part, d, k in (("val", va, 0), ("test", te, 1)):
            rows.append(pd.DataFrame(dict(row=d.row, part=part, p_delay=out["y_delay"][k],
                                          p_loss=out["y_loss"][k])))
        pd.concat(rows).to_parquet(os.path.join(RESULTS, f"{name}{suffix}_preds.parquet"), index=False)
        with open(os.path.join(RESULTS, f"{name}{suffix}_params.json"), "w") as fh:
            json.dump({"features": feats, "best": params, "time_s": time.time() - t0}, fh, indent=1)
        log(f"{name}{suffix} : {time.time() - t0:.0f} s, {params}")
    return [f"{m}{suffix}" for m in models]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="topo")
    ap.add_argument("--drop", default=None, choices=list(DROP))
    ap.add_argument("--models", default="naive,ridge,hgb,mlp")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run(args.split, args.drop, args.models.split(","), force=args.force)


if __name__ == "__main__":
    main()
