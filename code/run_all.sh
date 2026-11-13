#!/bin/bash
# Pipeline complet : extraction -> préparation -> modèles tabulaires (CPU) + GNN (GPU si disponible).
#
# Usage : bash code/run_all.sh [chemin_python_gpu]
#   chemin_python_gpu : interpréteur Python disposant de PyTorch CUDA (optionnel).
#                       À défaut, les GNN tournent sur CPU (≈ 5× plus lent).
# Durées indicatives (CPU 16 cœurs + GPU RTX 4070 portable) :
#   extraction ≈ 6 min, préparation ≈ 3 min, tabulaires ≈ 25 min, GNN ≈ 16 min chacun (GPU).
# Tous les journaux sont écrits dans results/logs/.
set -e
cd "$(dirname "$0")"
PY=python
GPY=${1:-python}
DEV=$($GPY -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")
L=../results/logs
mkdir -p $L

# 1. archives brutes -> data/processed/{samples,flows,links,queues}.parquet
$PY extraction.py --workers 14 > ../data/processed/extraction_log.txt 2>&1
# 2. features, classes de flux, masque d'événements, graphes
$PY preparation.py > $L/preparation.out 2>&1

# 3. GNN : deux chaînes en parallèle sur le même GPU (configuration principale, ablations,
#    graines supplémentaires, décalage de distribution)
gnn() { $GPY run_gnn.py --device $DEV --threads 3 --name "$@" > $L/$1.out 2>&1; }
( gnn gnn; gnn gnn_T0 --T 0; gnn gnn_noqueue --no-queues; gnn gnn_no_physique --drop physique;
  gnn gnn_no_allocation --drop allocation; gnn gnn_seed1 --seed 1 ) &
( gnn gnn_shift --split shift; gnn gnn_no_reservation --drop reservation; gnn gnn_no_charge --drop charge;
  gnn gnn_seed2 --seed 2 ) &

# 4. modèles sans graphe sur CPU, pendant que le GPU travaille
(
  export OMP_NUM_THREADS=8
  $PY run_tabular.py --split topo                                    # 4 modèles, découpage principal
  for d in reservation charge physique allocation; do                # ablations des features
    $PY run_tabular.py --split topo --drop $d --models ridge,hgb,mlp
  done
  $PY run_tabular.py --split shift                                   # décalage vers les grands réseaux
) > $L/tabular.out 2>&1 &
wait
echo "pipeline terminé"
