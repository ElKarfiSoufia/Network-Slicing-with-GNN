"""Génère projet_B5G_slicing_GNN.ipynb à partir des cellules définies dans notebook_cells.py.

Le notebook est écrit en Python (notebook_cells.py) pour pouvoir être relu, comparé et
modifié comme du code ; ce script le convertit au format .ipynb. Pour l'exécuter ensuite :
  jupyter nbconvert --to notebook --execute --inplace projet_B5G_slicing_GNN.ipynb
"""
import os

import nbformat as nbf

from notebook_cells import CELLS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "projet_B5G_slicing_GNN.ipynb"))

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
for kind, src in CELLS:                      # ("md" | "code", texte de la cellule)
    src = src.strip("\n")
    nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
nbf.write(nb, OUT)
print(f"{len(nb.cells)} cellules -> {OUT}")
