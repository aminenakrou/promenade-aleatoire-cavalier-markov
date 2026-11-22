"""Promenade aléatoire d'un cavalier sur un échiquier : simulations.

Auteurs : Soufiane Derouich, Amine Nakrou (janvier 2026)

Trois parties, qui produisent les figures du rapport :
  1. le graphe de transition du cavalier (couleur des arêtes = probabilité) ;
  2. le temps de retour à la case centrale 27 : histogramme et QQ-plot exponentiel ;
  3. la sous-chaîne Y_n = X_{2n} : heatmaps des fréquences pour 4 cases de départ.

Utilisation :
    pip install -r requirements.txt
    python code_simulation.py

Les figures sont enregistrées dans le dossier `figures/`.
"""

import os
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from scipy import stats

SEED = 2026
DOSSIER_FIGURES = "figures"
MOUVEMENTS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]

# Matrice D(x) du nombre de déplacements possibles (case (0, 0) en haut à gauche)
D = np.array([
    [2, 3, 4, 4, 4, 4, 3, 2],
    [3, 4, 6, 6, 6, 6, 4, 3],
    [4, 6, 8, 8, 8, 8, 6, 4],
    [4, 6, 8, 8, 8, 8, 6, 4],
    [4, 6, 8, 8, 8, 8, 6, 4],
    [4, 6, 8, 8, 8, 8, 6, 4],
    [3, 4, 6, 6, 6, 6, 4, 3],
    [2, 3, 4, 4, 4, 4, 3, 2],
])


def to_index(r, c):
    return r * 8 + c


def from_index(index):
    return index // 8, index % 8


def mouvements_possibles(case):
    """Retourne les cases accessibles depuis une case donnée."""
    ligne, colonne = divmod(case, 8)
    deplacements = []
    for dl, dc in MOUVEMENTS:
        nl, nc = ligne + dl, colonne + dc
        if 0 <= nl < 8 and 0 <= nc < 8:
            deplacements.append(nl * 8 + nc)
    return deplacements


def chemin_figure(nom):
    os.makedirs(DOSSIER_FIGURES, exist_ok=True)
    return os.path.join(DOSSIER_FIGURES, nom)


# --------------------------------------------------------------------------
# Partie 1 : graphe de transition
# --------------------------------------------------------------------------
def partie1_graphe_transition():
    G = nx.DiGraph()
    for i in range(64):
        r, c = from_index(i)
        G.add_node(i, moves=D[r, c], r=r, c=c)

    for u in range(64):
        r_u, c_u = from_index(u)
        num_moves = G.nodes[u]["moves"]
        prob = 1.0 / num_moves if num_moves > 0 else 0.0
        for dr, dc in MOUVEMENTS:
            r_v, c_v = r_u + dr, c_u + dc
            if 0 <= r_v < 8 and 0 <= c_v < 8:
                G.add_edge(u, to_index(r_v, c_v), weight=prob)

    pos = {}
    for i in range(64):
        r, c = from_index(i)
        pos[i] = (c, 7 - r)

    plt.figure(figsize=(12, 12))

    # Une couleur par probabilité de transition
    prob_map = {
        2: {"P": 0.5, "color": "red"},
        3: {"P": 1 / 3, "color": "orange"},
        4: {"P": 0.25, "color": "yellow"},
        6: {"P": 1 / 6, "color": "deepskyblue"},
        8: {"P": 0.125, "color": "darkblue"},
    }

    legend_handles = []
    for d_moves, data in prob_map.items():
        source_nodes = [n for n, attr in G.nodes(data=True) if attr["moves"] == d_moves]
        edges_subset = [(u, v) for u, v in G.edges() if u in source_nodes]
        nx.draw_networkx_edges(G, pos, edgelist=edges_subset, edge_color=data["color"],
                               width=1.5, alpha=0.6, arrows=False)
        legend_handles.append(plt.Line2D([0], [0], color=data["color"], lw=3,
                                         label=f"$P = 1/{d_moves} \\approx {data['P']:.3f}$"))

    node_labels = {i: str(G.nodes[i]["moves"]) for i in G.nodes()}
    nx.draw_networkx_nodes(G, pos, node_color="lightgray", node_size=600, edgecolors="black", alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_color="black")

    plt.title("Graphe de transition du cavalier : probabilités par couleur d'arête", fontsize=16)
    plt.xticks(np.arange(8), ["A", "B", "C", "D", "E", "F", "G", "H"], fontsize=12)
    plt.yticks(np.arange(8), ["8", "7", "6", "5", "4", "3", "2", "1"], fontsize=12)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(handles=legend_handles, title="Probabilité de transition ($P_{x,y}$)", loc="lower left", fontsize=12)
    plt.savefig(chemin_figure("graphe_transition.png"), dpi=150, bbox_inches="tight")
    print(f"Somme des degrés : {D.sum()} (attendu : 336)")


# --------------------------------------------------------------------------
# Partie 2 : temps de retour à la case 27
# --------------------------------------------------------------------------
def simuler_temps_retour(case_depart, n_simulations=5000):
    """Simule le temps de retour à une case donnée."""
    temps_retours = []
    for _ in range(n_simulations):
        position = random.choice(mouvements_possibles(case_depart))  # premier mouvement
        temps = 1
        while position != case_depart:
            position = random.choice(mouvements_possibles(position))
            temps += 1
        temps_retours.append(temps)
    return np.array(temps_retours)


def partie2_temps_retour(case_depart=27, n_simulations=5000):
    degre = len(mouvements_possibles(case_depart))
    pi_theorique = degre / 336
    temps_theorique = 1 / pi_theorique

    print(f"\n=== Analyse case {case_depart} ===")
    print(f"Degré : {degre}")
    print(f"π({case_depart}) = {degre}/336 = {pi_theorique:.6f}")
    print(f"Temps moyen théorique : {temps_theorique:.2f} coups")

    temps_simules = simuler_temps_retour(case_depart, n_simulations)
    temps_moyen = temps_simules.mean()
    temps_median = np.median(temps_simules)
    ecart_type = temps_simules.std()
    erreur = abs(temps_moyen - temps_theorique) / temps_theorique * 100

    print(f"\n=== Résultats ({n_simulations} simulations) ===")
    print(f"Temps moyen observé : {temps_moyen:.2f} coups")
    print(f"Médiane : {temps_median:.2f} coups")
    print(f"Écart-type : {ecart_type:.2f}")
    print(f"Minimum : {temps_simules.min()} coups | Maximum : {temps_simules.max()} coups")
    print(f"Erreur relative : {erreur:.2f} %")

    t_stat, p_value = stats.ttest_1samp(temps_simules, temps_theorique)
    print(f"Test t : t = {t_stat:.3f}, p-value = {p_value:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Histogramme
    ax1.hist(temps_simules, bins=50, density=True, alpha=0.7, edgecolor="black",
             color="skyblue", label="Distribution simulée")
    ax1.axvline(temps_theorique, color="red", linestyle="--", linewidth=3,
                label=f"Théorique : {temps_theorique:.1f}")
    ax1.axvline(temps_moyen, color="green", linestyle="--", linewidth=3,
                label=f"Empirique : {temps_moyen:.1f}")
    ax1.set_xlabel("Temps de retour (coups)", fontsize=12)
    ax1.set_ylabel("Densité de probabilité", fontsize=12)
    ax1.set_title(f"Temps de retour au centre (case {case_depart})\n"
                  f"D(x)={degre}, π={pi_theorique:.4f}, E[T]={temps_theorique:.1f}",
                  fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11, loc="upper right")
    ax1.grid(True, alpha=0.3)
    stats_text = (f"Simulations : {n_simulations:,}\n"
                  f"Moyenne : {temps_moyen:.2f}\n"
                  f"Médiane : {temps_median:.2f}\n"
                  f"Écart-type : {ecart_type:.2f}\n"
                  f"Erreur : {erreur:.2f} %\n"
                  f"p-value : {p_value:.4f}")
    ax1.text(0.98, 0.70, stats_text, transform=ax1.transAxes, verticalalignment="top",
             horizontalalignment="right", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    # QQ-plot face à une loi exponentielle
    stats.probplot(temps_simules, dist="expon", plot=ax2)
    ax2.set_title("QQ-plot vs distribution exponentielle", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Quantiles théoriques (exponentielle)", fontsize=12)
    ax2.set_ylabel("Quantiles observés", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Analyse du temps de retour - case {case_depart} (centre de l'échiquier)",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(chemin_figure(f"temps_retour_case_{case_depart}.png"), dpi=150, bbox_inches="tight")
    return temps_moyen, temps_theorique


# --------------------------------------------------------------------------
# Partie 3 : sous-chaîne Y_n = X_{2n}
# --------------------------------------------------------------------------
def simuler_chaine_pairs(case_depart, n_iterations=30000):
    """Simule la chaîne Y_n = X_{2n} : on enregistre la position toutes les deux étapes."""
    position = case_depart
    compteur = np.zeros(64, dtype=int)
    compteur[position] += 1
    for _ in range(n_iterations):
        for _ in range(2):  # deux mouvements du cavalier
            position = random.choice(mouvements_possibles(position))
        compteur[position] += 1
    frequences = compteur / compteur.sum()
    return frequences, compteur


def partie3_chaine_pairs(n_iterations=30000):
    departs = [0, 1, 18, 27]
    noms = ["Case 0 (coin pair)", "Case 1 (bord impair)", "Case 18 (centre pair)", "Case 27 (centre impair)"]
    parites = ["PAIRE", "IMPAIRE", "PAIRE", "IMPAIRE"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    resultats = {}

    for idx, (depart, nom, parite) in enumerate(zip(departs, noms, parites)):
        print(f"\n=== Simulation case {depart} ({nom}) ===")
        frequences, compteur = simuler_chaine_pairs(depart, n_iterations)

        # Masque des cases inaccessibles (parité différente de la case de départ)
        mask = np.ones((8, 8), dtype=bool)
        for i in range(64):
            ligne, colonne = divmod(i, 8)
            if (ligne + colonne) % 2 == (depart // 8 + depart % 8) % 2:
                mask[ligne, colonne] = False

        heatmap_data = frequences.reshape(8, 8)
        ax = axes[idx]
        sns.heatmap(heatmap_data, mask=mask, annot=True, fmt=".4f", cmap="YlOrRd",
                    cbar_kws={"label": "Fréquence"}, ax=ax, linewidths=0.5,
                    linecolor="black", annot_kws={"size": 8})

        cases_visitees = int((compteur > 0).sum())
        resultats[depart] = cases_visitees
        ax.set_title(f"{nom}\nParité {parite} - {cases_visitees}/64 cases visitées", fontsize=12, fontweight="bold")
        ax.set_xlabel("Colonne", fontsize=10)
        ax.set_ylabel("Ligne", fontsize=10)
        ax.set_xticks(np.arange(8) + 0.5)
        ax.set_yticks(np.arange(8) + 0.5)
        ax.set_xticklabels(["A", "B", "C", "D", "E", "F", "G", "H"])
        ax.set_yticklabels(["1", "2", "3", "4", "5", "6", "7", "8"])

        print(f"  Cases visitées : {cases_visitees}/64")
        print(f"  Fréquence maximale : {frequences.max():.4f}")
        print(f"  Fréquence minimale (parmi visitées) : {frequences[frequences > 0].min():.4f}")

    plt.suptitle("Distribution des fréquences de la chaîne Y_n = X_{2n}\n"
                 f"Dépendance à la parité de la case initiale ({n_iterations:,} itérations)",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(chemin_figure("chaine_pairs_comparaison.png"), dpi=150, bbox_inches="tight", facecolor="white")
    print("\nLa chaîne Y_n ne visite que les cases de même parité que la case initiale :")
    print("cela confirme la structure bipartite du graphe et la période 2 de la chaîne.")
    return resultats


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    partie1_graphe_transition()
    partie2_temps_retour()
    partie3_chaine_pairs()
    print(f"\nFigures enregistrées dans le dossier '{DOSSIER_FIGURES}/'.")
    plt.show()
