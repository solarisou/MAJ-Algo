"""
Benchmark COMPLET - Tous les algorithmes sur instances Pisinger
Génère des graphiques dans le style de gen_graph_extended_v2.py
"""

import csv
import time
import os
import sys
from pathlib import Path

# Imports des algorithmes (à ajuster selon ton projet)
# Tu devras peut-être modifier ces imports selon ta structure
try:
    from dynamic_programming import knapsack_bottom_up, knapsack_top_down
    from generer_probleme_KnapsackProblem import generer_probleme_knapsack
    # Ajoute ici les imports de TES autres algorithmes
    # from greedy import greedy_algorithm_ratio, greedy_algorithm_value, greedy_algorithm_weight
    # from branch_and_bound import branch_and_bound_bfs, branch_and_bound_dfs
    # from genetic import genetic_knapsack
    # from ant_colony import ant_colony_knapsack
    # etc.
except ImportError as e:
    print(f" Erreur d'import: {e}")
    print("   Assure-toi que tous les modules d'algorithmes sont disponibles")


def lire_instance_pisinger(filepath):
    """
    Lit une instance Pisinger depuis un fichier
    
    Format:
    - Ligne 1: n (objets), c (capacité), z (optimal connu)
    - Lignes suivantes: valeur, poids
    
    Returns:
        dict: {nom, n, capacite, optimal_value, valeurs, poids}
    """
    with open(filepath, 'r') as f:
        lignes = f.readlines()
    
    # Première ligne
    premiere = lignes[0].strip().split()
    n = int(premiere[0])
    capacite = int(premiere[1])
    optimal_value = int(premiere[2]) if len(premiere) > 2 else None
    
    # Objets
    valeurs = []
    poids = []
    
    for i in range(1, min(n + 1, len(lignes))):
        ligne = lignes[i].strip().split()
        if len(ligne) >= 2:
            valeurs.append(int(ligne[0]))
            poids.append(int(ligne[1]))
    
    nom = Path(filepath).stem
    
    return {
        'nom': nom,
        'n': n,
        'capacite': capacite,
        'optimal_value': optimal_value,
        'valeurs': valeurs,
        'poids': poids,
        'type': extraire_type_instance(nom)
    }


def extraire_type_instance(nom_fichier):
    """Extrait le type d'instance depuis le nom"""
    if 'knapPI_1_' in nom_fichier:
        return 'Uncorrelated'
    elif 'knapPI_2_' in nom_fichier:
        return 'Weakly Correlated'
    elif 'knapPI_3_' in nom_fichier:
        return 'Strongly Correlated'
    elif 'knapPI_4_' in nom_fichier:
        return 'Inverse Strongly Correlated'
    elif 'knapPI_5_' in nom_fichier:
        return 'Almost Strongly Correlated'
    elif 'knapPI_6_' in nom_fichier:
        return 'Subset-Sum'
    elif 'knapPI_7_' in nom_fichier:
        return 'Uncorrelated Similar Weights'
    else:
        return 'Unknown'


def charger_instances_pisinger(repertoire='data/pisinger_instances', limite=None, taille_max=1000):
    """
    Charge les instances Pisinger depuis le répertoire
    
    Args:
        repertoire: Chemin vers le répertoire
        limite: Nombre max d'instances (None = toutes)
        taille_max: Taille max (n) des instances
    
    Returns:
        list: Liste de dictionnaires d'instances
    """
    if not os.path.exists(repertoire):
        print(f" Répertoire non trouvé: {repertoire}")
        print("   Place les instances Pisinger dans data/pisinger_instances/")
        return []
    
    fichiers = []
    for f in os.listdir(repertoire):
        filepath = os.path.join(repertoire, f)
        if os.path.isfile(filepath) and (f.startswith('knapPI_') or f.endswith('.txt')):
            fichiers.append(filepath)
    
    fichiers.sort()
    
    print(f"\n Trouvé {len(fichiers)} fichiers Pisinger")
    
    instances = []
    for filepath in fichiers:
        try:
            instance = lire_instance_pisinger(filepath)
            
            # Filtrer par taille
            if instance['n'] > taille_max:
                continue
            
            instances.append(instance)
            
            if limite and len(instances) >= limite:
                break
                
        except Exception as e:
            print(f" Erreur lecture {Path(filepath).name}: {e}")
    
    print(f" Chargé {len(instances)} instances (n ≤ {taille_max})")
    
    return instances


def benchmark_algorithme(algo_func, algo_nom, poids, valeurs, capacite, timeout=60):
    """
    Benchmark d'un algorithme sur une instance
    
    Returns:
        dict: {valeur, temps, iterations, erreur}
    """
    start = time.time()
    
    try:
        # Certains algorithmes retournent (valeur, objets), d'autres (valeur, objets, iterations)
        resultat = algo_func(poids, valeurs, capacite)
        
        temps = time.time() - start
        
        if temps > timeout:
            return {'valeur': None, 'temps': None, 'iterations': None, 'erreur': 'Timeout'}
        
        # Gérer différents formats de retour
        if isinstance(resultat, tuple):
            if len(resultat) == 3:
                valeur, objets, iterations = resultat
            elif len(resultat) == 2:
                valeur, objets = resultat
                iterations = None
            else:
                valeur = resultat[0]
                iterations = None
        else:
            valeur = resultat
            iterations = None
        
        return {
            'valeur': valeur,
            'temps': temps,
            'iterations': iterations,
            'erreur': None
        }
        
    except Exception as e:
        return {
            'valeur': None,
            'temps': None,
            'iterations': None,
            'erreur': str(e)
        }


def executer_benchmark_complet(instances, algorithmes, output_csv='benchmark_all_algorithms.csv'):
    """
    Exécute le benchmark complet
    
    Args:
        instances: Liste d'instances Pisinger
        algorithmes: Liste de tuples (fonction, nom)
        output_csv: Fichier de sortie
    
    Returns:
        list: Résultats
    """
    print("\n" + "="*80)
    print("BENCHMARK COMPLET - TOUS ALGORITHMES SUR INSTANCES PISINGER")
    print("="*80)
    
    print(f"\nInstances: {len(instances)}")
    print(f"Algorithmes: {len(algorithmes)}")
    print(f"Total tests: {len(instances) * len(algorithmes)}")
    
    resultats = []
    
    for idx_inst, instance in enumerate(instances, 1):
        print(f"\n[{idx_inst}/{len(instances)}] Instance: {instance['nom']} (n={instance['n']}, C={instance['capacite']})")
        
        if instance['optimal_value']:
            print(f"              Optimal connu: {instance['optimal_value']}")
        
        for algo_func, algo_nom in algorithmes:
            print(f"  → {algo_nom:<30}", end=" ", flush=True)
            
            result = benchmark_algorithme(
                algo_func, algo_nom,
                instance['poids'], instance['valeurs'], instance['capacite']
            )
            
            if result['erreur']:
                print(f" {result['erreur']}")
            elif result['valeur'] is not None:
                # Calculer gap
                if instance['optimal_value']:
                    gap = instance['optimal_value'] - result['valeur']
                    gap_pct = (gap / instance['optimal_value'] * 100) if instance['optimal_value'] > 0 else 0
                    ratio_optimal = (result['valeur'] / instance['optimal_value'] * 100) if instance['optimal_value'] > 0 else 0
                else:
                    gap = None
                    gap_pct = None
                    ratio_optimal = None
                
                # Affichage
                temps_str = f"{result['temps']*1000:.1f}ms" if result['temps'] < 1 else f"{result['temps']:.2f}s"
                
                if ratio_optimal is not None:
                    if ratio_optimal >= 99.99:
                        print(f" Optimal ({result['valeur']}) en {temps_str}")
                    else:
                        print(f" {ratio_optimal:.2f}% ({result['valeur']}) en {temps_str}")
                else:
                    print(f" {result['valeur']} en {temps_str}")
                
                # Sauvegarder résultat
                resultats.append({
                    'instance': instance['nom'],
                    'instance_type': instance['type'],
                    'n': instance['n'],
                    'capacite': instance['capacite'],
                    'algorithme': algo_nom,
                    'valeur_obtenue': result['valeur'],
                    'valeur_optimale': instance['optimal_value'],
                    'gap': gap,
                    'gap_percent': gap_pct,
                    'ratio_optimal': ratio_optimal,
                    'temps_s': result['temps'],
                    'temps_ms': result['temps'] * 1000 if result['temps'] else None,
                    'iterations': result['iterations'],
                    'erreur': result['erreur']
                })
            else:
                print(" Échec")
    
    # Sauvegarder CSV
    print(f"\n Sauvegarde dans {output_csv}...")
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'instance', 'instance_type', 'n', 'capacite',
            'algorithme', 'valeur_obtenue', 'valeur_optimale',
            'gap', 'gap_percent', 'ratio_optimal',
            'temps_s', 'temps_ms', 'iterations', 'erreur'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(resultats)
    
    print(f" {len(resultats)} résultats sauvegardés")
    
    return resultats


def main():
    """Fonction principale"""
    
    # CONFIGURATION - À ADAPTER SELON TON PROJET
    # ==========================================
    
    # 1. Définir les algorithmes à tester
    #    Format: (fonction, nom_affichage)
    algorithmes = [
        (knapsack_bottom_up, "DP Bottom-Up"),
        (knapsack_top_down, "DP Top-Down"),
        # Ajoute ici TES autres algorithmes:
        # (greedy_algorithm_ratio, "Greedy Ratio"),
        # (greedy_algorithm_value, "Greedy Value"),
        # (greedy_algorithm_weight, "Greedy Weight"),
        # (branch_and_bound_bfs, "Branch & Bound BFS"),
        # (genetic_knapsack, "Genetic Algorithm"),
        # (ant_colony_knapsack, "Ant Colony"),
        # etc.
    ]
    
    # 2. Charger instances Pisinger
    instances = charger_instances_pisinger(
        repertoire='data/pisinger_instances',
        limite=20,  # Limite pour test rapide - mettre None pour tout
        taille_max=500  # Éviter les instances trop grandes
    )
    
    if not instances:
        print("\n Aucune instance chargée!")
        print("\nPour récupérer les instances Pisinger:")
        print("  1. git clone https://github.com/dnlfm/knapsack-01-instances.git")
        print("  2. cp knapsack-01-instances/pisinger_instances_01_KP/large_scale/* data/pisinger_instances/")
        return
    
    # 3. Exécuter benchmark
    resultats = executer_benchmark_complet(
        instances,
        algorithmes,
        output_csv='benchmark_all_algorithms.csv'
    )
    
    # 4. Générer graphiques
    print("\n" + "="*80)
    print("GÉNÉRATION DES GRAPHIQUES")
    print("="*80)
    print("\nPour générer les graphiques dans ton style:")
    print("  python generate_graphs_multi_algo.py")
    print("\n(Script de visualisation créé séparément)")


if __name__ == "__main__":
    main()
