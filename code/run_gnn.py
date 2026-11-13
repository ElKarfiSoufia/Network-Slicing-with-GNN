"""Entraîne une configuration du GNN et sauvegarde prédictions, historique et poids.

Exemples :
  python run_gnn.py --name gnn                       # modèle principal (T=4, files WFQ)
  python run_gnn.py --name gnn_T0 --T 0              # sans rétroaction flux -> liens
  python run_gnn.py --name gnn_noqueue --no-queues   # sans nœuds « file WFQ »
  python run_gnn.py --name gnn_no_reservation --drop reservation
  python run_gnn.py --name gnn_shift --split shift
  python run_gnn.py --name gnn --device cuda          # même calcul sur GPU (≈ 5× plus rapide)

Fichiers écrits dans results/ :
  <name>_log.txt        journal d'entraînement (une ligne par époque)
  <name>_history.csv    historique (perte d'entraînement, MAE de validation)
  <name>.pt             poids du meilleur modèle
  <name>_preds.parquet  prédictions sur la validation et le test (row, part, p_delay, p_loss)
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch

from features import PROCESSED
from gnn import FLOW_X, LINK_X, QUEUE_X, Normalizer, SliceGNN, predict, train
from splits import make_splits

RESULTS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))

# Familles de features retirées pour l'ablation (mêmes familles que pour les modèles
# tabulaires, voir run_tabular.DROP) : {type de nœud: [indices des colonnes à neutraliser]}
DROP = {
    "reservation": {"xf": [FLOW_X.index("delta")],
                    "xq": [QUEUE_X.index(c) for c in ("w_frac", "log_reserved", "rho_q", "rho_q_eff")]},
    "charge": {"xl": [LINK_X.index(c) for c in ("rho", "log_n_flows_link")],
               "xq": [QUEUE_X.index(c) for c in ("rho_q", "rho_q_eff", "log_n_flows_q")]},
    "physique": {"xl": [LINK_X.index("log_full_delay")], "xq": [QUEUE_X.index("log_qdelay_full")]},
}
# toute l'information d'allocation : réservation + délai de file pleine (qui dépend du poids WFQ)
DROP["allocation"] = {"xf": DROP["reservation"]["xf"], "xq": DROP["reservation"]["xq"] + DROP["physique"]["xq"],
                      "xl": DROP["physique"]["xl"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="topo")
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--no-queues", action="store_true")
    ap.add_argument("--drop", default=None, choices=list(DROP))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cpu", help="cpu ou cuda (mêmes résultats attendus, plus rapide)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)                    # initialisation reproductible des poids
    os.makedirs(RESULTS, exist_ok=True)
    samples = pd.read_parquet(os.path.join(PROCESSED, "samples.parquet"))
    with open(os.path.join(PROCESSED, "graphs.pkl"), "rb") as fh:
        graphs = pickle.load(fh)
    sp = make_splits(samples)[args.split]
    tr, va, te = (np.array([s for s in sp[k] if s in graphs]) for k in ("train", "val", "test"))
    norm = Normalizer(graphs, tr)                   # statistiques du train uniquement
    drop = DROP.get(args.drop)

    model = SliceGNN(d=args.d, T=args.T, use_queues=not args.no_queues).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    log_path = os.path.join(RESULTS, f"{args.name}_log.txt")
    with open(log_path, "w", encoding="utf-8") as log:
        def say(msg):
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
        say(f"{args} | paramètres : {n_params} | train {len(tr)} val {len(va)} test {len(te)}")
        t0 = time.time()
        hist = train(model, graphs, tr, va, norm, epochs=args.epochs, batch=args.batch, lr=args.lr,
                     patience=args.patience, seed=args.seed, drop=drop, log=say, device=args.device)
        say(f"entraînement : {time.time() - t0:.0f} s")

    hist.to_csv(os.path.join(RESULTS, f"{args.name}_history.csv"), index=False)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, os.path.join(RESULTS, f"{args.name}.pt"))
    preds = []
    for part, sims in (("val", va), ("test", te)):
        p, rows = predict(model, graphs, sims, norm, drop=drop, device=args.device)
        preds.append(pd.DataFrame(dict(row=rows, part=part, p_delay=p[:, 0], p_loss=p[:, 1])))
    pd.concat(preds).to_parquet(os.path.join(RESULTS, f"{args.name}_preds.parquet"), index=False)
    print("prédictions sauvegardées")


if __name__ == "__main__":
    main()
