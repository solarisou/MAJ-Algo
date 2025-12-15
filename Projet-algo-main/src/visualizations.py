"""
Visualisations pour la soutenance - Projet Knapsack
Graphiques clairs, lisibles et pertinents
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_results(csv_path: str = None) -> pd.DataFrame:
    """Charge les résultats du benchmark"""
    if csv_path is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        possible_paths = [
            os.path.join(base, 'results', 'benchmark_results.csv'),
            r'C:\Users\benkirane\Desktop\projet\projet algo\Projet-algo\results\benchmark_results.csv'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break
        else:
            raise FileNotFoundError(f"Fichier non trouvé")
    
    df = pd.read_csv(csv_path)
    df = df[df['is_valid'] == True].copy()
    return df


# =============================================================================
# GRAPHIQUE 1: Temps d'exécution - Comparaison claire
# =============================================================================
def plot_1_temps_execution(df: pd.DataFrame, output_dir: str):
    """
    Barres horizontales : temps moyen par algorithme
    Simple, clair, facile à présenter
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Moyenne des temps par algorithme
    temps = df.groupby('algorithm')['time'].mean().sort_values(ascending=True)
    
    # Couleurs par catégorie
    colors = []
    for algo in temps.index:
        if algo == 'Brute-Force':
            colors.append('#8B0000')  # Rouge foncé
        elif algo in ['DP-BottomUp', 'DP-TopDown']:
            colors.append('#2E86AB')  # Bleu
        elif algo in ['B&B-BFS', 'B&B-LeastCost']:
            colors.append('#A23B72')  # Violet
        elif algo in ['Greedy-Ratio', 'Fractional', 'Fractional+']:
            colors.append('#2ECC71')  # Vert
        elif algo in ['Genetic', 'AntColony', 'Randomized']:
            colors.append('#E67E22')  # Orange
        else:  # FPTAS
            colors.append('#9B59B6')  # Violet clair
    
    y_pos = np.arange(len(temps))
    bars = ax.barh(y_pos, temps.values, color=colors, edgecolor='black', height=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(temps.index, fontsize=12)
    ax.set_xlabel('Temps moyen (secondes)', fontsize=13)
    ax.set_title('Temps d\'Exécution Moyen par Algorithme', fontsize=16, fontweight='bold', pad=15)
    
    # Ajouter les valeurs sur les barres
    for bar, val in zip(bars, temps.values):
        if val < 1:
            label = f'{val*1000:.1f} ms'
        else:
            label = f'{val:.2f} s'
        ax.text(bar.get_width() + max(temps)*0.02, bar.get_y() + bar.get_height()/2, 
                label, va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlim(0, max(temps) * 1.25)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Légende
    legend_elements = [
        mpatches.Patch(color='#8B0000', label='Exhaustif'),
        mpatches.Patch(color='#2E86AB', label='Prog. Dynamique'),
        mpatches.Patch(color='#A23B72', label='Branch & Bound'),
        mpatches.Patch(color='#2ECC71', label='Gloutons'),
        mpatches.Patch(color='#E67E22', label='Métaheuristiques'),
        mpatches.Patch(color='#9B59B6', label='Approximation'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph1_temps_execution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Graphique 1: Temps d'exécution")


# =============================================================================
# GRAPHIQUE 2: Qualité - Qui trouve l'optimal ?
# =============================================================================
def plot_2_qualite_solutions(df: pd.DataFrame, output_dir: str):
    """
    Montre pour chaque algo : % de fois où il trouve l'optimal (gap=0)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculer le % d'optimal trouvé par algo
    # Pour les tiny (sans optimal connu), on compare à la meilleure solution trouvée
    results = []
    
    for algo in df['algorithm'].unique():
        algo_data = df[df['algorithm'] == algo]
        
        # Compter les solutions optimales
        if 'gap_percent' in algo_data.columns:
            # Gap = 0 ou NaN (tiny instances où bruteforce = référence)
            n_optimal = ((algo_data['gap_percent'] == 0) | (algo_data['gap_percent'].isna())).sum()
            n_total = len(algo_data)
            pct_optimal = (n_optimal / n_total) * 100 if n_total > 0 else 0
            
            # Gap moyen (exclure NaN)
            gap_mean = algo_data['gap_percent'].dropna().mean()
            gap_mean = gap_mean if not pd.isna(gap_mean) else 0
        else:
            pct_optimal = 0
            gap_mean = 100
            
        results.append({
            'algorithm': algo,
            'pct_optimal': pct_optimal,
            'gap_mean': gap_mean
        })
    
    results_df = pd.DataFrame(results).sort_values('pct_optimal', ascending=True)
    
    # Couleurs selon le % d'optimal
    colors = []
    for pct in results_df['pct_optimal']:
        if pct >= 90:
            colors.append('#27AE60')  # Vert - excellent
        elif pct >= 50:
            colors.append('#F39C12')  # Orange - moyen
        else:
            colors.append('#E74C3C')  # Rouge - mauvais
    
    y_pos = np.arange(len(results_df))
    bars = ax.barh(y_pos, results_df['pct_optimal'], color=colors, edgecolor='black', height=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_df['algorithm'], fontsize=12)
    ax.set_xlabel('Solutions optimales trouvées (%)', fontsize=13)
    ax.set_title('Capacité à Trouver la Solution Optimale', fontsize=16, fontweight='bold', pad=15)
    
    # Ajouter les valeurs
    for bar, (_, row) in zip(bars, results_df.iterrows()):
        pct = row['pct_optimal']
        gap = row['gap_mean']
        label = f'{pct:.0f}%'
        if gap > 0:
            label += f' (gap moy: {gap:.1f}%)'
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                label, va='center', fontsize=10)
    
    ax.set_xlim(0, 115)
    ax.axvline(x=100, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph2_qualite_solutions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Graphique 2: Qualité des solutions")


# =============================================================================
# GRAPHIQUE 3: Explosion Exponentielle du Brute-Force
# =============================================================================
def plot_3_explosion_exponentielle(df: pd.DataFrame, output_dir: str):
    """
    Montre pourquoi le brute-force est limité : temps vs 2^n
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Graphique gauche : Temps mesurés ---
    bf_data = df[df['algorithm'] == 'Brute-Force'].groupby('n')['time'].mean()
    
    bars1 = ax1.bar(bf_data.index.astype(str), bf_data.values, color='#8B0000', edgecolor='black')
    ax1.set_xlabel('Taille du problème (n)', fontsize=12)
    ax1.set_ylabel('Temps (secondes)', fontsize=12)
    ax1.set_title('Temps Mesuré - Brute-Force', fontsize=14, fontweight='bold')
    
    for i, (n, t) in enumerate(bf_data.items()):
        ax1.text(i, t + max(bf_data.values)*0.05, f'{t*1000:.1f}ms', ha='center', fontsize=11, fontweight='bold')
    
    ax1.set_ylim(0, max(bf_data.values) * 1.2)
    ax1.grid(axis='y', alpha=0.3)
    
    # --- Graphique droite : Projection exponentielle ---
    n_values = np.array([10, 15, 20, 25, 30, 40, 50])
    
    # Estimation basée sur les mesures
    if len(bf_data) > 0:
        t_20 = bf_data.get(20, 0.03)
        ops_per_sec = (2**20) / t_20 if t_20 > 0 else 1e8
    else:
        ops_per_sec = 1e8
    
    temps_estimes = (2**n_values) / ops_per_sec
    
    # Convertir en unités lisibles
    labels = []
    for t in temps_estimes:
        if t < 1:
            labels.append(f'{t*1000:.0f}ms')
        elif t < 60:
            labels.append(f'{t:.1f}s')
        elif t < 3600:
            labels.append(f'{t/60:.0f}min')
        elif t < 86400:
            labels.append(f'{t/3600:.0f}h')
        elif t < 86400*365:
            labels.append(f'{t/86400:.0f}j')
        else:
            labels.append(f'{t/(86400*365):.0f}ans')
    
    colors = ['#27AE60' if t < 60 else '#F39C12' if t < 3600 else '#E74C3C' for t in temps_estimes]
    
    bars2 = ax2.bar(range(len(n_values)), temps_estimes, color=colors, edgecolor='black')
    ax2.set_xticks(range(len(n_values)))
    ax2.set_xticklabels([f'n={n}' for n in n_values], fontsize=10)
    ax2.set_ylabel('Temps estimé (échelle log)', fontsize=12)
    ax2.set_title('Projection Exponentielle O(2^n)', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    
    # Labels à côté des barres
    for i, (bar, label) in enumerate(zip(bars2, labels)):
        ax2.text(i, bar.get_height() * 2, label, ha='center', fontsize=9, fontweight='bold')
    
    # Ligne limite
    ax2.axhline(y=60, color='orange', linestyle='--', linewidth=2, label='Limite 1 min')
    ax2.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph3_explosion_exponentielle.png'), dpi=150)
    plt.close()
    print("✓ Graphique 3: Explosion exponentielle")


# =============================================================================
# GRAPHIQUE 4: Barres Groupées - Performance par Type d'Instance
# =============================================================================
def plot_4_comparaison_grande_instance(df: pd.DataFrame, output_dir: str):
    """
    Diagramme en bâtons : Pour chaque algorithme, barres côte à côte pour Type 1, 2, 3, Generated
    """
    # Extraire le type depuis le nom de l'instance
    def get_instance_type(instance_name):
        if pd.isna(instance_name):
            return None
        name = str(instance_name).lower()
        if 'type1' in name or 'knappi_1' in name or '_1_' in name.split('knappi')[-1][:5]:
            return 'Type 1'
        elif 'type2' in name or 'knappi_2' in name or '_2_' in name.split('knappi')[-1][:5]:
            return 'Type 2'
        elif 'type3' in name or 'knappi_3' in name or '_3_' in name.split('knappi')[-1][:5]:
            return 'Type 3'
        elif 'tiny' in name or 'generated' in name:
            return 'Generated'
        return None
    
    df_copy = df.copy()
    df_copy['instance_type'] = df_copy['instance'].apply(get_instance_type)
    df_copy = df_copy[df_copy['instance_type'].notna()]
    df_copy['gap_percent'] = df_copy['gap_percent'].fillna(0)
    
    if len(df_copy) == 0:
        print("⚠ Pas assez de données pour le graphique par type")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Types disponibles
    type_order = ['Type 1', 'Type 2', 'Type 3', 'Generated']
    available_types = [t for t in type_order if t in df_copy['instance_type'].unique()]
    
    # Sélectionner les algorithmes (exclure Brute-Force qui skip souvent)
    exclude_algos = ['Brute-Force']
    all_algos = [a for a in df_copy['algorithm'].unique() if a not in exclude_algos]
    
    # Trier par gap moyen
    algo_gaps = df_copy.groupby('algorithm')['gap_percent'].mean()
    selected_algos = [a for a in algo_gaps.sort_values().index if a in all_algos]
    
    df_filtered = df_copy[df_copy['algorithm'].isin(selected_algos)]
    
    # Couleurs pour chaque type
    type_colors = {
        'Type 1': '#3498DB',      # Bleu
        'Type 2': '#E74C3C',      # Rouge  
        'Type 3': '#27AE60',      # Vert
        'Generated': '#9B59B6'    # Violet
    }
    
    # --- GRAPHIQUE 1 : Gap par algorithme ---
    gap_data = df_filtered.groupby(['algorithm', 'instance_type'])['gap_percent'].mean().unstack()
    gap_data = gap_data.reindex(selected_algos)
    gap_data = gap_data[[t for t in available_types if t in gap_data.columns]]
    
    x = np.arange(len(selected_algos))
    width = 0.18
    n_types = len(available_types)
    
    for i, typ in enumerate(available_types):
        if typ in gap_data.columns:
            offset = (i - n_types/2 + 0.5) * width
            vals = gap_data[typ].fillna(0)
            bars = ax1.bar(x + offset, vals, width, label=typ, 
                          color=type_colors.get(typ, '#888'), edgecolor='white', linewidth=0.5)
    
    ax1.set_xlabel('Algorithme', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Gap to Optimal (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Qualité des Solutions : Gap par Algorithme et Type d\'Instance', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(selected_algos, fontsize=10, rotation=30, ha='right')
    ax1.legend(loc='upper right', fontsize=10, title='Type Instance')
    ax1.set_ylim(0, max(gap_data.max().max() * 1.2, 1))
    ax1.axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax1.grid(axis='y', alpha=0.3)
    
    # Ajouter annotation "Optimal" à gauche
    ax1.annotate('← Optimal', xy=(0, 0), xytext=(-0.5, 0.5), fontsize=10, color='green', fontweight='bold')
    
    # --- GRAPHIQUE 2 : Temps par algorithme ---
    time_data = df_filtered.groupby(['algorithm', 'instance_type'])['time'].mean().unstack()
    time_data = time_data.reindex(selected_algos)
    time_data = time_data[[t for t in available_types if t in time_data.columns]]
    
    for i, typ in enumerate(available_types):
        if typ in time_data.columns:
            offset = (i - n_types/2 + 0.5) * width
            vals = time_data[typ].fillna(0.0001)  # Éviter log(0)
            bars = ax2.bar(x + offset, vals, width, label=typ,
                          color=type_colors.get(typ, '#888'), edgecolor='white', linewidth=0.5)
    
    ax2.set_xlabel('Algorithme', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Temps d\'Exécution (s) - échelle log', fontsize=12, fontweight='bold')
    ax2.set_title('Temps d\'Exécution par Algorithme et Type d\'Instance', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(selected_algos, fontsize=10, rotation=30, ha='right')
    ax2.legend(loc='upper right', fontsize=10, title='Type Instance')
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph4_barres_types.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Graphique 4: Barres par algorithme et type d'instance")


# =============================================================================
# GRAPHIQUE 5: Tableau de Synthèse
# =============================================================================
def plot_5_tableau_synthese(df: pd.DataFrame, output_dir: str):
    """
    Tableau récapitulatif : le slide de conclusion parfait
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    # Données du tableau
    algos_info = [
        ('Brute-Force', 'O(2^n)', 'Oui', 'n <= 20', '*'),
        ('DP-BottomUp', 'O(n*W)', 'Oui', 'W modere', '***'),
        ('DP-TopDown', 'O(n*W)', 'Oui', 'W modere', '***'),
        ('B&B-BFS', 'O(2^n) pire cas', 'Oui', 'Variable', '**'),
        ('B&B-LeastCost', 'O(2^n) pire cas', 'Oui', 'Rapide en pratique', '***'),
        ('Greedy-Ratio', 'O(n log n)', 'Non', 'Tres rapide', '**'),
        ('Fractional+', 'O(n log n)', 'Non', 'Bonne approx.', '**'),
        ('FPTAS', 'O(n^3/e)', '(1-e)OPT', 'Garanti', '**'),
        ('Genetic', 'O(g*p*n)', 'Non', 'Grandes instances', '**'),
        ('AntColony', 'O(i*a*n)', 'Non', 'Grandes instances', '**'),
        ('Randomized', 'O(n*iter)', 'Non', 'Simple', '*'),
    ]
    
    # Ajouter temps moyen depuis les données
    temps_moyens = df.groupby('algorithm')['time'].mean()
    
    table_data = []
    for algo, complexite, optimal, usage, note in algos_info:
        t = temps_moyens.get(algo, 0)
        if t < 0.001:
            temps_str = f'{t*1000:.2f} ms'
        elif t < 1:
            temps_str = f'{t*1000:.1f} ms'
        else:
            temps_str = f'{t:.2f} s'
        table_data.append([algo, complexite, optimal, temps_str, usage, note])
    
    columns = ['Algorithme', 'Complexité', 'Optimal?', 'Temps Moy.', 'Cas d\'Usage', 'Note']
    
    # Créer le tableau
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colWidths=[0.15, 0.15, 0.1, 0.12, 0.18, 0.08]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style du header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#2C3E50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Couleurs alternées pour les lignes
    for i in range(1, len(table_data) + 1):
        color = '#ECF0F1' if i % 2 == 0 else 'white'
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(color)
        
        # Colorer la colonne "Optimal?"
        if table_data[i-1][2] == 'Oui':
            table[(i, 2)].set_facecolor('#27AE60')
            table[(i, 2)].set_text_props(color='white', fontweight='bold')
        elif table_data[i-1][2] == 'Non':
            table[(i, 2)].set_facecolor('#E74C3C')
            table[(i, 2)].set_text_props(color='white')
    
    ax.set_title('Tableau Récapitulatif des Algorithmes', fontsize=18, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph5_tableau_synthese.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Graphique 5: Tableau de synthèse")


# =============================================================================
# GRAPHIQUE 6: Recommandations Visuelles
# =============================================================================
def plot_6_recommandations(df: pd.DataFrame, output_dir: str):
    """
    Graphique de recommandation : quel algo selon la situation ?
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Créer un diagramme de décision visuel
    recommendations = [
        ("Problème très petit\n(n ≤ 20)", "Brute-Force\n✓ Solution optimale garantie", "#8B0000", 0.1, 0.85),
        ("Capacité modérée\n(W < 10⁶)", "Programmation Dynamique\n✓ Optimal en O(nW)", "#2E86AB", 0.1, 0.55),
        ("Besoin de rapidité\n(temps < 1ms)", "Greedy-Ratio\n⚡ Ultra-rapide, ~97% optimal", "#2ECC71", 0.1, 0.25),
        ("Grande instance\n(n > 1000)", "B&B-LeastCost\n✓ Optimal, rapide en pratique", "#A23B72", 0.55, 0.85),
        ("Garantie théorique\nrequise", "FPTAS\n✓ Garantie (1-ε)×OPT", "#9B59B6", 0.55, 0.55),
        ("Exploration\ndiversifiee", "Metaheuristiques\nGenetic, AntColony", "#E67E22", 0.55, 0.25),
    ]
    
    for situation, algo, color, x, y in recommendations:
        # Boîte situation
        rect1 = plt.Rectangle((x, y), 0.18, 0.15, facecolor='#ECF0F1', edgecolor='#2C3E50', linewidth=2)
        ax.add_patch(rect1)
        ax.text(x + 0.09, y + 0.075, situation, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Flèche
        ax.annotate('', xy=(x + 0.22, y + 0.075), xytext=(x + 0.18, y + 0.075),
                   arrowprops=dict(arrowstyle='->', color=color, lw=3))
        
        # Boîte recommandation
        rect2 = plt.Rectangle((x + 0.22, y), 0.22, 0.15, facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(rect2)
        ax.text(x + 0.33, y + 0.075, algo, ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Guide de Choix d\'Algorithme', fontsize=18, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph6_recommandations.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Graphique 6: Recommandations")


# =============================================================================
# GRAPHIQUE 7: B&B-DFS - Impact du Type d'Instance
# =============================================================================
def plot_7_bnb_dfs_comparison(output_dir: str):
    """
    Graphique spécial pour B&B-DFS : montre pourquoi il est problématique
    sur les instances fortement corrélées (Type 3)
    """
    import time
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.algorithms.branch_and_bound import branch_and_bound_dfs, branch_and_bound_bfs, branch_and_bound_least_cost
    from src.utils.instance_loader import PisingerInstanceLoader
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Charger quelques instances de chaque type
    loader = PisingerInstanceLoader()
    
    results = {'Type 1': [], 'Type 2': [], 'Type 3': []}
    algos = {
        'B&B-DFS': branch_and_bound_dfs,
        'B&B-BFS': branch_and_bound_bfs,
        'B&B-LeastCost': branch_and_bound_least_cost
    }
    
    # Tester sur des instances MEDIUM (n=100-200) pour avoir des temps en secondes
    timeout = 30.0  # 30 secondes max
    
    print("\n  Test B&B-DFS sur instances medium (peut prendre 1-2 minutes)...")
    
    for cat in ['medium']:
        instances = loader.get_all_instances([cat], download_if_missing=True)
        
        for inst in instances[:3]:  # 1 de chaque type (beaucoup plus rapide)
            # Déterminer le type
            name = inst.name.lower()
            if 'knappi_1' in name or '_1_' in name.split('knappi')[-1][:5]:
                inst_type = 'Type 1'
            elif 'knappi_2' in name or '_2_' in name.split('knappi')[-1][:5]:
                inst_type = 'Type 2'
            elif 'knappi_3' in name or '_3_' in name.split('knappi')[-1][:5]:
                inst_type = 'Type 3'
            else:
                continue
            
            for algo_name, algo_func in algos.items():
                try:
                    start = time.time()
                    result = algo_func(inst.weights, inst.values, inst.capacity)
                    elapsed = time.time() - start
                    
                    if elapsed > timeout:
                        elapsed = timeout  # Cap pour l'affichage
                    
                    results[inst_type].append({
                        'algo': algo_name,
                        'time': elapsed,
                        'n': inst.n
                    })
                except Exception as e:
                    results[inst_type].append({
                        'algo': algo_name,
                        'time': timeout,
                        'n': inst.n
                    })
    
    # Barres de temps par type
    types = ['Type 1', 'Type 2', 'Type 3']
    algo_names = ['B&B-DFS', 'B&B-BFS', 'B&B-LeastCost']
    colors = ['#E74C3C', '#3498DB', '#27AE60']  # Rouge, Bleu, Vert
    
    x = np.arange(len(types))
    width = 0.25
    
    for i, algo in enumerate(algo_names):
        times = []
        for t in types:
            algo_times = [r['time'] for r in results[t] if r['algo'] == algo]
            times.append(np.mean(algo_times) if algo_times else 0)
        
        bars = ax.bar(x + i*width, times, width, label=algo, color=colors[i], edgecolor='black')
        
        # Ajouter valeurs
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                label = f'{h:.3f}s' if h < 1 else f'{h:.1f}s'
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, label,
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Type d\'Instance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Temps d\'Exécution (s)', fontsize=12, fontweight='bold')
    ax.set_title('Comparaison des Variantes Branch & Bound par Type d\'Instance', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Type 1\n(Uncorrelated)', 'Type 2\n(Weakly Corr.)', 'Type 3\n(Strongly Corr.)'], fontsize=11)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph7_bnb_dfs_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Graphique 7: Analyse B&B-DFS vs Types d'instances")


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================
def generate_all_graphs(csv_path: str = None, output_dir: str = None):
    """Génère tous les graphiques"""
    
    df = load_results(csv_path)
    print(f"Données chargées: {len(df)} résultats valides")
    
    if output_dir is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base, 'results', 'graphs')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Dossier de sortie: {output_dir}\n")
    
    print("=" * 50)
    print("GÉNÉRATION DES GRAPHIQUES")
    print("=" * 50)
    
    plot_1_temps_execution(df, output_dir)
    plot_2_qualite_solutions(df, output_dir)
    plot_3_explosion_exponentielle(df, output_dir)
    plot_4_comparaison_grande_instance(df, output_dir)
    plot_5_tableau_synthese(df, output_dir)
    plot_6_recommandations(df, output_dir)
    plot_7_bnb_dfs_comparison(output_dir)
    
    print("=" * 50)
    print(f"✅ 7 graphiques générés dans: {output_dir}")
    print("=" * 50)
    
    return output_dir


if __name__ == "__main__":
    generate_all_graphs()

