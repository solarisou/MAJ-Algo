#!/usr/bin/env python3
"""
Point d'entrée principal du projet Knapsack
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          PROJET ALGORITHMES - PROBLÈME DU SAC À DOS              ║
║                     (0/1 Knapsack Problem)                       ║
╚══════════════════════════════════════════════════════════════════╝

Ce projet implémente et compare différents algorithmes pour résoudre
le problème du sac à dos 0/1, en utilisant les instances de benchmark
Pisinger.

OPTIONS:
  1. Télécharger les instances de test (Pisinger)
  2. Exécuter le benchmark rapide
  3. Exécuter le benchmark complet
  4. Tester un algorithme spécifique
  5. Générer les graphiques d'analyse
  6. Générer des problèmes aléatoires
  7. Résoudre un problème réel (exemple)
  8. Quitter
""")
    
    try:
        choice = input("Votre choix (1-8): ").strip()
    except KeyboardInterrupt:
        print("\n\nAu revoir!")
        return
    
    if choice == '1':
        from download_instances import main as download_main
        download_main()
    
    elif choice == '2':
        from src.benchmark import run_quick_benchmark
        run_quick_benchmark()
    
    elif choice == '3':
        from src.benchmark import run_full_benchmark
        run_full_benchmark()
    
    elif choice == '4':
        test_single_algorithm()
    
    elif choice == '5':
        generate_visualizations()
    
    elif choice == '6':
        generate_random_problems()
    
    elif choice == '7':
        run_real_world_example()
    
    elif choice == '8':
        print("Au revoir!")
    
    else:
        print("Choix invalide.")


def generate_visualizations():
    """Génère les graphiques d'analyse des résultats"""
    import os
    
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    results_file = os.path.join(results_dir, 'benchmark_results.csv')
    
    if not os.path.exists(results_file):
        print(f"\n[ERREUR] Fichier de résultats non trouvé: {results_file}")
        print("   Veuillez d'abord exécuter un benchmark (option 2 ou 3).")
        return
    
    from src.visualizations import generate_all_visualizations
    
    output_dir = os.path.join(results_dir, 'graphs')
    generate_all_visualizations(results_file, output_dir)
    
    print(f"\n[OK] Les graphiques ont été générés dans: {output_dir}")
    print("\nGraphiques disponibles:")
    print("  01. Comparaison des temps d'exécution")
    print("  02. Gap par rapport à l'optimal")
    print("  03. Analyse de scalabilité")
    print("  04. Trade-off qualité vs temps")
    print("  05. Heatmap de performance (gap)")
    print("  06. Heatmap des temps d'exécution")
    print("  07. Boxplot distribution des gaps")
    print("  08. Boxplot distribution des temps")
    print("  09. Classement multi-critères")
    print("  10. Comparaison par catégorie")
    print("  11. Résumé des solutions")


def test_single_algorithm():
    """Teste un algorithme sur une petite instance"""
    print("\nAlgorithmes disponibles:")
    print("  1. Brute-Force")
    print("  2. Programmation Dynamique (Bottom-Up)")
    print("  3. Programmation Dynamique (Top-Down)")
    print("  4. Greedy (Ratio)")
    print("  5. Branch and Bound (BFS)")
    print("  6. Branch and Bound (DFS)")
    print("  7. FPTAS")
    print("  8. Algorithme Génétique")
    print("  9. Colonie de Fourmis")
    print(" 10. Randomisé")
    
    try:
        choice = input("\nChoisir un algorithme (1-10): ").strip()
    except KeyboardInterrupt:
        return
    
    # Exemple de test
    weights = [12, 2, 1, 1, 4, 8, 6, 3, 5, 7]
    values = [4, 2, 1, 2, 10, 3, 5, 4, 6, 8]
    capacity = 20
    
    print(f"\nInstance de test:")
    print(f"   n = {len(weights)}")
    print(f"   Capacité = {capacity}")
    print(f"   Poids = {weights}")
    print(f"   Valeurs = {values}")
    
    algorithms = {
        '1': ('Brute-Force', lambda: __import__('src.algorithms.bruteforce', fromlist=['bruteforce_knapsack']).bruteforce_knapsack(values, weights, capacity)),
        '2': ('DP Bottom-Up', lambda: __import__('src.algorithms.dynamic_programming', fromlist=['knapsack_bottom_up']).knapsack_bottom_up(weights, values, capacity)),
        '3': ('DP Top-Down', lambda: __import__('src.algorithms.dynamic_programming', fromlist=['knapsack_top_down']).knapsack_top_down(weights, values, capacity)),
        '4': ('Greedy Ratio', lambda: __import__('src.algorithms.greedy', fromlist=['greedy_algorithm_ratio']).greedy_algorithm_ratio(weights, values, capacity)),
        '5': ('B&B BFS', lambda: __import__('src.algorithms.branch_and_bound', fromlist=['branch_and_bound_bfs']).branch_and_bound_bfs(weights, values, capacity)),
        '6': ('B&B DFS', lambda: __import__('src.algorithms.branch_and_bound', fromlist=['branch_and_bound_dfs']).branch_and_bound_dfs(weights, values, capacity)),
        '7': ('FPTAS', lambda: __import__('src.algorithms.fptas', fromlist=['fptas_knapsack']).fptas_knapsack(weights, values, capacity, 0.1)),
        '8': ('Génétique', lambda: __import__('src.algorithms.genetic', fromlist=['genetic_knapsack']).genetic_knapsack(weights, values, capacity)),
        '9': ('Colonie Fourmis', lambda: __import__('src.algorithms.ant_colony', fromlist=['ant_colony_knapsack']).ant_colony_knapsack(weights, values, capacity)),
        '10': ('Randomisé', lambda: __import__('src.algorithms.randomized', fromlist=['randomized_knapsack']).randomized_knapsack(weights, values, capacity)),
    }
    
    if choice not in algorithms:
        print("Choix invalide")
        return
    
    name, func = algorithms[choice]
    
    import time
    start = time.time()
    result = func()
    elapsed = time.time() - start
    
    print(f"\nAlgorithme: {name}")
    print(f"   Résultat: {result}")
    print(f"   Temps: {elapsed:.6f}s")


def generate_random_problems():
    """Génère des problèmes aléatoires avec le générateur"""
    import os
    from src.utils.problem_generator import (
        KnapsackProblemGenerator, 
        DistributionType,
        generate_test_suite
    )
    
    print("\nOptions de génération:")
    print("  1. Générer une instance unique")
    print("  2. Générer une suite de tests complète")
    
    try:
        choice = input("\nVotre choix (1-2): ").strip()
    except KeyboardInterrupt:
        return
    
    if choice == '1':
        print("\nDistributions disponibles:")
        distributions = [
            ("1", "uniform", "Uniforme"),
            ("2", "correlated", "Corrélée"),
            ("3", "inverse_correlated", "Inversement corrélée"),
            ("4", "subset_sum", "Subset Sum"),
            ("5", "almost_strongly_correlated", "Presque fortement corrélée"),
        ]
        for code, key, name in distributions:
            print(f"  {code}. {name}")
        
        dist_choice = input("\nType de distribution (1-5): ").strip()
        dist_map = {d[0]: d[1] for d in distributions}
        dist = dist_map.get(dist_choice, "uniform")
        
        try:
            n = int(input("Nombre d'objets (ex: 50): ").strip())
        except ValueError:
            n = 50
        
        gen = KnapsackProblemGenerator(seed=42)
        instance = gen.generate(
            n=n,
            distribution=DistributionType(dist),
            capacity_ratio=0.5
        )
        
        print(f"\n[OK] Instance générée:")
        print(f"   n = {instance['n']}")
        print(f"   Capacité = {instance['capacity']}")
        print(f"   Distribution = {instance['distribution']}")
        print(f"   Poids total = {instance['total_weight']}")
        print(f"   Valeur totale = {instance['total_value']}")
        print(f"   Premiers poids = {instance['weights'][:5]}...")
        print(f"   Premières valeurs = {instance['values'][:5]}...")
        
        save = input("\nSauvegarder l'instance? (o/n): ").strip().lower()
        if save == 'o':
            output_dir = os.path.join(os.path.dirname(__file__), 'data', 'generated')
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"custom_{dist}_{n}.txt")
            gen.save_instance(instance, filepath)
            print(f"   Sauvegardé: {filepath}")
    
    elif choice == '2':
        output_dir = os.path.join(os.path.dirname(__file__), 'data', 'generated')
        print(f"\nGénération de la suite de tests dans: {output_dir}")
        files = generate_test_suite(output_dir)
        print(f"\n[OK] {len(files)} fichiers générés!")


def run_real_world_example():
    """Exécute l'exemple de problème réel"""
    print("\nExécution de l'exemple de problème réel...")
    print("   (Optimisation de budget marketing)\n")
    
    from real_world_example import run_real_world_example as run_example
    run_example()


if __name__ == "__main__":
    main()
