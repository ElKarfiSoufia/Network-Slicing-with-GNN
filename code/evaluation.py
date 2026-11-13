"""Métriques pondérées par flux (poids w des classes) et tableaux de comparaison.

Toutes les prédictions sont stockées dans results/<modèle>_preds.parquet avec la colonne
« row » (identifiant de classe de flux). On les joint aux métadonnées de flow_classes
(cible réelle, poids w, slice, étiquette d'événement) pour calculer les métriques sur
n'importe quel sous-ensemble : global, hors événement, événement, surcharge, délestage.
"""
import os

import numpy as np
import pandas as pd

from features import PROCESSED

RESULTS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
META = ["row", "sim", "w", "slice_type", "y_delay", "y_loss", "event", "event_type", "z", "b"]


def load_meta():
    """Métadonnées de chaque classe de flux : cibles réelles, poids w, slice, événement."""
    return pd.read_parquet(os.path.join(PROCESSED, "flow_classes.parquet"), columns=META)


def load_preds(name, meta, part="test"):
    """Prédictions d'un modèle (« part » = "val" ou "test") jointes aux métadonnées."""
    p = pd.read_parquet(os.path.join(RESULTS, f"{name}_preds.parquet"))
    return p[p.part == part].merge(meta, on="row")


def metrics(d):
    """MAE / MSE pondérées par w pour les deux cibles (log10 délai, taux de perte).

    MAE = Σ w |prédit - réel| / Σ w ;  MSE = Σ w (prédit - réel)² / Σ w.
    La MSE pénalise davantage les grosses erreurs : comparer MAE et MSE indique si un
    modèle fait quelques très grosses erreurs ou beaucoup de petites.
    """
    out = {}
    for t, p in (("delay", "p_delay"), ("loss", "p_loss")):
        y = d[f"y_{t}"].to_numpy()
        m = ~np.isnan(y)
        w, e = d.w.to_numpy()[m], d[p].to_numpy()[m] - y[m]
        out[f"MAE_{t}"] = float(np.sum(w * np.abs(e)) / np.sum(w))
        out[f"MSE_{t}"] = float(np.sum(w * e ** 2) / np.sum(w))
    out["flux"] = int(d.w.sum())
    return out


def evaluate(names, meta, part="test", labels=None):
    """Tableau : une ligne par (modèle, sous-ensemble) avec MAE/MSE des deux cibles."""
    rows = []
    for n in names:
        d = load_preds(n, meta, part)
        subsets = {"global": d, "hors événement": d[~d.event], "événement": d[d.event],
                   "surcharge": d[d.event_type == "surcharge"], "délestage": d[d.event_type == "délestage"]}
        for s, ds in subsets.items():
            rows.append(dict(modèle=(labels or {}).get(n, n), sous_ensemble=s, **metrics(ds)))
    return pd.DataFrame(rows)
