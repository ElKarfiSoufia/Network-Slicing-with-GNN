"""Préparation complète : tables de features, classes de flux, masque d'événements, graphes.

Étapes (fonction main) :
  1. features.build_flow_table : 24 features par flux + cibles (≈ 1,5 min)
  2. features.deduplicate      : regroupement en classes de flux de poids w
  3. event_mask                : écart robuste z et étiquette normal / surcharge / délestage
  4. gnn.build_graphs          : un graphe flux / files WFQ / liens par simulation
Fichiers écrits : flow_classes.parquet, links_feat.parquet, queues_feat.parquet, graphs.pkl.
La colonne « row » de flow_classes est l'identifiant de ligne utilisé par toutes les
prédictions sauvegardées dans results/ (jointure prédiction <-> métadonnées).

Usage : python preparation.py [--events-only]   (≈ 5 min ; écrit dans data/processed/)
"""
import os
import pickle
import time

import numpy as np
import pandas as pd

from features import PROCESSED, build_flow_table, deduplicate, load_raw
from gnn import build_graphs

# Choix justifiés au chapitre 5 du rapport :
#  - Hampel (1974) ; Leys et al. (2013) : médiane et MAD, seuil 3 « très conservateur »
Z_EVENT = 3.0      # seuil de Hampel : |z| >= 3 écarts robustes
MAD_FLOOR = 0.02   # plancher de dispersion de log10(b) (évite les divisions par ~0)


def event_mask(ft):
    """Événement = flux dont la charge du goulot s'écarte fortement du régime habituel.

    b  = max(rho_q_eff, rho_path_max) : charge offerte / capacité de la ressource la plus
         chargée du flux (file WFQ de l'antenne ou lien de transport).
    b couvre plus de 10 ordres de grandeur (files WFQ quasi affamées) : l'écart est mesuré
    en log10, c'est-à-dire de façon multiplicative.
    z  = (log10 b - médiane) / (1,4826 · MAD), médiane et MAD calculées sur tous les flux de
         la même topologie et du même type de slice (pondérés par w) = « régime habituel ».
    Événement si |z| >= 3 (identificateur de Hampel) : surcharge (z >= 3) ou délestage (z <= -3).
    """
    ft = ft.copy()
    # charge du goulot : la ressource la plus chargée du flux (file d'antenne ou lien)
    ft["b"] = ft[["rho_q_eff", "rho_path_max"]].max(axis=1)
    ft["log_b"] = np.log10(ft.b.clip(lower=1e-6))
    # on répète chaque classe w fois : médiane et MAD sont ainsi calculées par flux
    rep = ft.loc[ft.index.repeat(ft.w), ["topology", "slice_type", "log_b"]]
    g = rep.groupby(["topology", "slice_type"]).log_b
    med = g.median().rename("b_med")
    # MAD = médiane des écarts absolus à la médiane du groupe (topologie, slice)
    mad = (rep.log_b - g.transform("median")).abs().groupby([rep.topology, rep.slice_type]).median().rename("b_mad")
    ft = ft.drop(columns=["b_med", "b_mad"], errors="ignore")
    ft = ft.join(med, on=["topology", "slice_type"]).join(mad, on=["topology", "slice_type"])
    # 1,4826 × MAD estime l'écart-type d'une loi normale de façon robuste
    ft["z"] = (ft.log_b - ft.b_med) / (1.4826 * ft.b_mad).clip(lower=MAD_FLOOR)
    ft["event"] = ft.z.abs() >= Z_EVENT
    ft["event_type"] = np.select([ft.z >= Z_EVENT, ft.z <= -Z_EVENT], ["surcharge", "délestage"], "normal")
    return ft


def events_only():
    """Recalcule seulement le masque d'événements (les indices de ligne ne changent pas)."""
    path = os.path.join(PROCESSED, "flow_classes.parquet")
    ft = event_mask(pd.read_parquet(path))
    ft.to_parquet(path, index=False)
    share = ft.groupby("event_type").w.sum() / ft.w.sum()
    print("masque recalculé :", share.round(4).to_dict())


def main():
    t0 = time.time()
    samples, flows, links, queues = load_raw()
    f, links_f, queues_f = build_flow_table(samples, flows, links, queues)
    print(f"table des flux : {f.shape} ({time.time() - t0:.0f} s)")
    ft = deduplicate(f)
    ft = event_mask(ft)
    ft["row"] = np.arange(len(ft))          # identifiant stable d'une classe de flux
    print(f"classes de flux : {len(ft)} (multiplicité moyenne {ft.w.mean():.2f})")
    ft.to_parquet(os.path.join(PROCESSED, "flow_classes.parquet"), index=False)
    links_f.to_parquet(os.path.join(PROCESSED, "links_feat.parquet"), index=False)
    queues_f.to_parquet(os.path.join(PROCESSED, "queues_feat.parquet"), index=False)
    graphs = build_graphs(ft, links_f, queues_f)
    with open(os.path.join(PROCESSED, "graphs.pkl"), "wb") as fh:
        pickle.dump(graphs, fh)
    print(f"{len(graphs)} graphes ; terminé en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    import sys
    events_only() if "--events-only" in sys.argv else main()
