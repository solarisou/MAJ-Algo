"""
Génération de graphiques MULTI-ALGORITHMES - Style gen_graph_extended_v2
Adapté pour TOUS les algorithmes testés sur instances Pisinger
"""

import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from collections import defaultdict


def lire_csv(filename):
    """Lit un fichier CSV et retourne les données"""
    data = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"❌ Fichier {filename} non trouvé")
        return None
    return data


def grouper_par_algo(data):
    """Groupe les données par algorithme"""
    algos = defaultdict(list)
    for row in data:
        algos[row['algorithme']].append(row)
    return dict(algos)


def grouper_par_taille(data):
    """Groupe les données par taille (n)"""
    tailles = defaultdict(list)
    for row in data:
        n = int(row['n'])
        tailles[n].append(row)
    return dict(sorted(tailles.items()))


# PALETTE DE COULEURS (style gen_graph_extended_v2)
COULEURS_ALGOS = {
    'DP Bottom-Up': '#3498db',
    'DP Top-Down': '#e74c3c',
    'Greedy Ratio': '#27ae60',
    'Greedy Value': '#16a085',
    'Greedy Weight': '#1abc9c',
    'Branch & Bound BFS': '#9b59b6',
    'Branch & Bound DFS': '#8e44ad',
    'Branch & Bound Least-Cost': '#34495e',
    'Branch & Bound IDDFS': '#95a5a6',
    'FPTAS': '#e67e22',
    'Fractional Approximation': '#d35400',
    'Genetic Algorithm': '#c0392b',
    'Ant Colony': '#e74c3c',
    'Randomized': '#f39c12',
}

MARQUEURS = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', '+', 'x']


def graphique_temps_par_taille():
    """Graphique 1: Temps d'exécution vs Taille (TOUS algorithmes)"""
    data = lire_csv('benchmark_all_algorithms.csv')
    if data is None:
        return
    
    # Filtrer les erreurs
    data = [row for row in data if row['erreur'] in [None, '', 'None']]
    
    algos_data = grouper_par_algo(data)
    tailles_data = grouper_par_taille(data)
    
    plt.figure(figsize=(14, 8))
    
    tailles_uniques = sorted(set(int(row['n']) for row in data))
    
    for idx, (algo_nom, algo_rows) in enumerate(algos_data.items()):
        # Calculer temps moyen par taille
        temps_par_taille = defaultdict(list)
        for row in algo_rows:
            if row['temps_ms'] and row['temps_ms'] != 'None':
                temps_par_taille[int(row['n'])].append(float(row['temps_ms']))
        
        tailles = sorted(temps_par_taille.keys())
        temps_moyens = [np.mean(temps_par_taille[t]) for t in tailles]
        
        if not tailles:
            continue
        
        couleur = COULEURS_ALGOS.get(algo_nom, f'C{idx}')
        marqueur = MARQUEURS[idx % len(MARQUEURS)]
        
        plt.plot(tailles, temps_moyens, 
                marker=marqueur, linewidth=2.5, markersize=8,
                label=algo_nom, color=couleur, alpha=0.9)
    
    plt.xlabel('Taille du problème (nombre d\'objets)', fontsize=14, fontweight='bold')
    plt.ylabel('Temps d\'exécution moyen (ms)', fontsize=14, fontweight='bold')
    plt.title('Impact de la Taille sur le Temps d\'Exécution - Tous Algorithmes', 
             fontsize=16, fontweight='bold')
    plt.legend(fontsize=11, loc='upper left', ncol=2)
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('graphique_temps_tous_algorithmes.png', dpi=300, bbox_inches='tight')
    print("✓ Graphique: graphique_temps_tous_algorithmes.png")
    plt.close()


def graphique_ratio_optimal_par_algorithme():
    """Graphique 2: Ratio optimal moyen par algorithme (barres)"""
    data = lire_csv('benchmark_all_algorithms.csv')
    if data is None:
        return
    
    # Filtrer les données valides
    data = [row for row in data 
            if row['ratio_optimal'] and row['ratio_optimal'] != 'None' 
            and row['erreur'] in [None, '', 'None']]
    
    algos_data = grouper_par_algo(data)
    
    # Calculer ratio moyen par algo
    algos_noms = []
    ratios_moyens = []
    
    for algo_nom, algo_rows in algos_data.items():
        ratios = [float(row['ratio_optimal']) for row in algo_rows 
                 if row['ratio_optimal'] and row['ratio_optimal'] != 'None']
        if ratios:
            algos_noms.append(algo_nom)
            ratios_moyens.append(np.mean(ratios))
    
    # Trier par ratio décroissant
    indices = np.argsort(ratios_moyens)[::-1]
    algos_noms = [algos_noms[i] for i in indices]
    ratios_moyens = [ratios_moyens[i] for i in indices]
    
    couleurs = [COULEURS_ALGOS.get(nom, '#95a5a6') for nom in algos_noms]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(algos_noms))
    bars = ax.bar(x, ratios_moyens, color=couleurs, alpha=0.8, edgecolor='black')
    
    # Ligne à 100%
    ax.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Optimal (100%)')
    
    # Zone verte 99-101%
    ax.fill_between(range(-1, len(algos_noms)+1), 99, 101, alpha=0.2, color='green')
    
    ax.set_xlabel('Algorithme', fontsize=14, fontweight='bold')
    ax.set_ylabel('Ratio optimal moyen (%)', fontsize=14, fontweight='bold')
    ax.set_title('Qualité des Solutions - Comparaison Tous Algorithmes', 
                fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algos_noms, rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.4, axis='y')
    ax.set_ylim([min(ratios_moyens) - 5, 102])
    
    # Annotations
    for i, (bar, ratio) in enumerate(zip(bars, ratios_moyens)):
        height = bar.get_height()
        ax.annotate(f'{ratio:.2f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('graphique_ratio_optimal_comparaison.png', dpi=300, bbox_inches='tight')
    print("✓ Graphique: graphique_ratio_optimal_comparaison.png")
    plt.close()


def graphique_temps_vs_qualite_scatter():
    """Graphique 3: Scatter plot Temps vs Qualité (trade-off)"""
    data = lire_csv('benchmark_all_algorithms.csv')
    if data is None:
        return
    
    # Filtrer données valides
    data = [row for row in data 
            if row['temps_ms'] and row['ratio_optimal'] 
            and row['temps_ms'] != 'None' and row['ratio_optimal'] != 'None'
            and row['erreur'] in [None, '', 'None']]
    
    algos_data = grouper_par_algo(data)
    
    plt.figure(figsize=(14, 8))
    
    for idx, (algo_nom, algo_rows) in enumerate(algos_data.items()):
        temps = [float(row['temps_ms']) for row in algo_rows]
        ratios = [float(row['ratio_optimal']) for row in algo_rows]
        
        # Calculer moyennes
        temps_moyen = np.mean(temps)
        ratio_moyen = np.mean(ratios)
        
        couleur = COULEURS_ALGOS.get(algo_nom, f'C{idx}')
        
        # Scatter des points individuels
        plt.scatter(temps, ratios, alpha=0.3, s=30, color=couleur)
        
        # Point moyen avec label
        plt.scatter(temps_moyen, ratio_moyen, 
                   s=300, color=couleur, alpha=0.9,
                   edgecolor='black', linewidth=2,
                   marker='o', label=algo_nom)
        
        # Annotation du nom
        plt.annotate(algo_nom, 
                    (temps_moyen, ratio_moyen),
                    xytext=(10, 5), textcoords='offset points',
                    fontsize=9, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.xlabel('Temps d\'exécution moyen (ms)', fontsize=14, fontweight='bold')
    plt.ylabel('Ratio optimal moyen (%)', fontsize=14, fontweight='bold')
    plt.title('Trade-off Temps vs Qualité - Tous Algorithmes', 
             fontsize=16, fontweight='bold')
    plt.xscale('log')
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Optimal')
    
    plt.tight_layout()
    plt.savefig('graphique_temps_vs_qualite_scatter.png', dpi=300, bbox_inches='tight')
    print("✓ Graphique: graphique_temps_vs_qualite_scatter.png")
    plt.close()


def graphique_4_metriques_combine():
    """Graphique 4: Vue d'ensemble 4 métriques (style original)"""
    data = lire_csv('benchmark_all_algorithms.csv')
    if data is None:
        return
    
    # Filtrer données valides
    data = [row for row in data if row['erreur'] in [None, '', 'None']]
    
    algos_data = grouper_par_algo(data)
    tailles_data = grouper_par_taille(data)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    tailles_uniques = sorted(set(int(row['n']) for row in data))
    
    # (A) Taille vs Temps
    for idx, (algo_nom, algo_rows) in enumerate(algos_data.items()):
        temps_par_taille = defaultdict(list)
        for row in algo_rows:
            if row['temps_ms'] and row['temps_ms'] != 'None':
                temps_par_taille[int(row['n'])].append(float(row['temps_ms']))
        
        tailles = sorted(temps_par_taille.keys())
        temps_moyens = [np.mean(temps_par_taille[t]) for t in tailles]
        
        if not tailles:
            continue
        
        couleur = COULEURS_ALGOS.get(algo_nom, f'C{idx}')
        marqueur = MARQUEURS[idx % len(MARQUEURS)]
        
        ax1.plot(tailles, temps_moyens, 
                marker=marqueur, linewidth=2, markersize=6,
                label=algo_nom, color=couleur, alpha=0.8)
    
    ax1.set_xlabel('Nombre d\'objets', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Temps (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Taille vs Temps d\'Exécution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left', ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # (B) Ratio optimal par algorithme
    algos_noms = []
    ratios_moyens = []
    couleurs = []
    
    for algo_nom, algo_rows in algos_data.items():
        ratios = [float(row['ratio_optimal']) for row in algo_rows 
                 if row['ratio_optimal'] and row['ratio_optimal'] != 'None']
        if ratios:
            algos_noms.append(algo_nom[:15])  # Tronquer pour lisibilité
            ratios_moyens.append(np.mean(ratios))
            couleurs.append(COULEURS_ALGOS.get(algo_nom, '#95a5a6'))
    
    x = np.arange(len(algos_noms))
    ax2.bar(x, ratios_moyens, color=couleurs, alpha=0.8, edgecolor='black')
    ax2.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(algos_noms, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Ratio optimal (%)', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Qualité Moyenne par Algorithme', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([min(ratios_moyens) - 5 if ratios_moyens else 0, 102])
    
    # (C) Gap moyen par type d'instance
    types_data = defaultdict(lambda: defaultdict(list))
    for row in data:
        if row['gap_percent'] and row['gap_percent'] != 'None':
            types_data[row['instance_type']][row['algorithme']].append(float(row['gap_percent']))
    
    types_noms = list(types_data.keys())[:5]  # Top 5 types
    
    if types_noms:
        x_types = np.arange(len(types_noms))
        width = 0.8 / len(algos_data)
        
        for idx, algo_nom in enumerate(list(algos_data.keys())[:5]):  # Top 5 algos
            gaps_moyens = []
            for type_nom in types_noms:
                if algo_nom in types_data[type_nom]:
                    gaps_moyens.append(np.mean(types_data[type_nom][algo_nom]))
                else:
                    gaps_moyens.append(0)
            
            couleur = COULEURS_ALGOS.get(algo_nom, f'C{idx}')
            ax3.bar(x_types + idx * width, gaps_moyens, width,
                   label=algo_nom[:12], color=couleur, alpha=0.8)
        
        ax3.set_xticks(x_types + width * (len(algos_data[:5]) - 1) / 2)
        ax3.set_xticklabels(types_noms, rotation=45, ha='right', fontsize=9)
        ax3.set_ylabel('Gap moyen (%)', fontsize=12, fontweight='bold')
        ax3.set_title('(C) Gap par Type d\'Instance', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=8, loc='upper left')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # (D) Nombre d'algorithmes optimaux par taille
    optimaux_par_taille = defaultdict(int)
    total_par_taille = defaultdict(int)
    
    for row in data:
        if row['ratio_optimal'] and row['ratio_optimal'] != 'None':
            n = int(row['n'])
            total_par_taille[n] += 1
            if float(row['ratio_optimal']) >= 99.99:
                optimaux_par_taille[n] += 1
    
    tailles = sorted(total_par_taille.keys())
    pourcentages = [(optimaux_par_taille[t] / total_par_taille[t] * 100) 
                    for t in tailles]
    
    ax4.plot(tailles, pourcentages, 'o-', linewidth=2.5, markersize=10,
            color='#27ae60', label='% Solutions optimales')
    ax4.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax4.fill_between(tailles, 0, pourcentages, alpha=0.3, color='green')
    
    ax4.set_xlabel('Nombre d\'objets', fontsize=12, fontweight='bold')
    ax4.set_ylabel('% Solutions optimales', fontsize=12, fontweight='bold')
    ax4.set_title('(D) Taux de Succès par Taille', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 105])
    
    plt.suptitle('Vue d\'Ensemble - Tous Algorithmes sur Instances Pisinger', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('graphique_vue_ensemble_4_metriques.png', dpi=300, bbox_inches='tight')
    print("✓ Graphique: graphique_vue_ensemble_4_metriques.png")
    plt.close()


def graphique_par_type_instance():
    """Graphique 5: Performance par type d'instance"""
    data = lire_csv('benchmark_all_algorithms.csv')
    if data is None:
        return
    
    data = [row for row in data if row['erreur'] in [None, '', 'None']]
    
    # Grouper par type
    types_data = defaultdict(lambda: defaultdict(list))
    for row in data:
        if row['temps_ms'] and row['temps_ms'] != 'None':
            types_data[row['instance_type']][row['algorithme']].append(float(row['temps_ms']))
    
    types_noms = list(types_data.keys())
    
    if not types_noms:
        print("⚠ Pas assez de données pour graphique par type")
        return
    
    n_types = len(types_noms)
    n_cols = 2
    n_rows = (n_types + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
    if n_types == 1:
        axes = [axes]
    axes = axes.flatten()
    
    for idx, (type_nom, algos_dict) in enumerate(types_data.items()):
        ax = axes[idx]
        
        algos_noms = list(algos_dict.keys())
        temps_moyens = [np.mean(algos_dict[algo]) for algo in algos_noms]
        couleurs = [COULEURS_ALGOS.get(algo, f'C{i}') for i, algo in enumerate(algos_noms)]
        
        y_pos = np.arange(len(algos_noms))
        ax.barh(y_pos, temps_moyens, color=couleurs, alpha=0.8, edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(algos_noms, fontsize=10)
        ax.set_xlabel('Temps moyen (ms)', fontsize=11, fontweight='bold')
        ax.set_title(f'{type_nom}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xscale('log')
        
        # Annotations
        for i, (y, t) in enumerate(zip(y_pos, temps_moyens)):
            ax.text(t, y, f'  {t:.1f}ms', va='center', fontsize=9)
    
    # Cacher axes vides
    for idx in range(len(types_noms), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Performance par Type d\'Instance Pisinger', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('graphique_par_type_instance.png', dpi=300, bbox_inches='tight')
    print("✓ Graphique: graphique_par_type_instance.png")
    plt.close()


def generer_tous_graphiques():
    """Génère tous les graphiques"""
    print("\n" + "="*80)
    print(" "*15 + "GÉNÉRATION DES GRAPHIQUES - TOUS ALGORITHMES")
    print("="*80 + "\n")
    
    print("Graphiques à générer:")
    print("  1. Temps vs Taille (tous algorithmes)")
    print("  2. Ratio optimal - Comparaison")
    print("  3. Trade-off Temps vs Qualité (scatter)")
    print("  4. Vue d'ensemble 4 métriques")
    print("  5. Performance par type d'instance")
    print()
    
    graphique_temps_par_taille()
    graphique_ratio_optimal_par_algorithme()
    graphique_temps_vs_qualite_scatter()
    graphique_4_metriques_combine()
    graphique_par_type_instance()
    
    print("\n" + "="*80)
    print("✓ Tous les graphiques ont été générés avec succès!")
    print("="*80)
    print("\nFichiers créés:")
    print("  • graphique_temps_tous_algorithmes.png")
    print("  • graphique_ratio_optimal_comparaison.png")
    print("  • graphique_temps_vs_qualite_scatter.png")
    print("  • graphique_vue_ensemble_4_metriques.png      [VUE D'ENSEMBLE]")
    print("  • graphique_par_type_instance.png")
    print("\nCes graphiques couvrent:")
    print("   ✓ Taille")
    print("   ✓ Vitesse")
    print("   ✓ Qualité (ratio optimal)")
    print("   ✓ Trade-offs")
    print("   ✓ Types d'instances")


if __name__ == "__main__":
    generer_tous_graphiques()
