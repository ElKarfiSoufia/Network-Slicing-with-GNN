"""Découpages train / validation / test, toujours par topologie entière.

Une même topologie (ex. « Renater1999 ») est simulée ~44 fois avec des slices et des
réservations différentes. Mettre des simulations d'une même topologie à la fois dans
l'entraînement et le test laisserait le modèle mémoriser la topologie : on découpe donc
par groupes de topologies (aucune topologie partagée entre ensembles). C'est la
validation « par blocs » recommandée pour les données groupées (Roberts et al., 2017) :
un découpage aléatoire des simulations surestimerait la performance.

La graine SEED est fixée : les découpages sont identiques à chaque exécution, pour tous
les modèles (comparaison équitable).
"""
import numpy as np
import pandas as pd

SEED = 42


def _assign(topos, sizes, fractions, rng):
    """Répartit des topologies mélangées pour atteindre les fractions de simulations.

    topos     : noms des topologies ; sizes : nombre de simulations de chacune
    fractions : parts visées, par ex. [0.70, 0.15, 0.15]
    Les topologies sont prises dans un ordre aléatoire ; on remplit la partie 0 jusqu'à
    70 % des simulations, puis la partie 1 jusqu'à 85 %, etc. Une topologie n'est jamais
    coupée entre deux parties. Retourne {topologie: numéro de partie}.
    """
    order = rng.permutation(len(topos))
    total, acc, out = sizes.sum(), 0, []
    bounds = np.cumsum(fractions)[:-1] * total     # seuils cumulés en nombre de simulations
    part = 0
    for i in order:
        while part < len(bounds) and acc >= bounds[part]:
            part += 1
        out.append((topos[i], part))
        acc += sizes[i]
    return dict(out)


def make_splits(samples, seed=SEED):
    """Retourne {nom: {"train": sims, "val": sims, "test": sims}}.

    - "topo"  : découpage aléatoire des topologies, ~70/15/15 % des simulations.
    - "shift" : décalage de distribution sur la taille du réseau. Les topologies dont la
                taille médiane est dans le quart supérieur (en nombre de simulations)
                forment le test ; les autres sont réparties 82/18 entre train et val.
    """
    rng = np.random.default_rng(seed)
    # une ligne par topologie : nombre de simulations et taille médiane (nœuds)
    per_topo = samples.groupby("topology").agg(n=("sim", "size"), size=("n_nodes", "median"))
    topos, sizes = per_topo.index.to_numpy(), per_topo.n.to_numpy()

    # --- découpage principal : topologies tirées au hasard -------------------------
    part = _assign(topos, sizes, [0.70, 0.15, 0.15], rng)
    lab = samples.topology.map(part).map({0: "train", 1: "val", 2: "test"})
    splits = {"topo": {k: samples.sim[lab == k].to_numpy() for k in ("train", "val", "test")}}

    # --- découpage « shift » : les plus grands réseaux en test -----------------------
    # on trie les topologies par taille et on garde en test le dernier quart des simulations
    by_size = per_topo.sort_values("size")
    cum = by_size.n.cumsum() / by_size.n.sum()
    large = set(by_size.index[cum > 0.75])
    small = per_topo.drop(index=list(large))
    part2 = _assign(small.index.to_numpy(), small.n.to_numpy(), [0.82, 0.18], rng)
    lab2 = samples.topology.map(lambda t: "test" if t in large else ("train" if part2[t] == 0 else "val"))
    splits["shift"] = {k: samples.sim[lab2 == k].to_numpy() for k in ("train", "val", "test")}
    splits["shift_threshold"] = float(per_topo.loc[list(large), "size"].min())
    return splits


def describe(splits, samples):
    """Tableau récapitulatif : simulations, topologies et taille médiane de chaque partie."""
    rows = []
    s = samples.set_index("sim")
    for name in ("topo", "shift"):
        for k in ("train", "val", "test"):
            sims = splits[name][k]
            rows.append(dict(split=name, part=k, simulations=len(sims),
                             topologies=s.loc[sims, "topology"].nunique(),
                             noeuds_median=float(s.loc[sims, "n_nodes"].median())))
    return pd.DataFrame(rows)
