"""
Tests unitaires pour les algorithmes Knapsack
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.bruteforce import bruteforce_knapsack
from src.algorithms.dynamic_programming import knapsack_bottom_up, knapsack_top_down
from src.algorithms.greedy import greedy_algorithm_ratio
from src.algorithms.branch_and_bound import branch_and_bound_bfs, branch_and_bound_dfs
from src.algorithms.fptas import fptas_knapsack
from src.algorithms.genetic import genetic_knapsack
from src.algorithms.ant_colony import ant_colony_knapsack
from src.algorithms.randomized import randomized_knapsack


def test_small_instance():
    """Test sur une petite instance avec solution connue"""
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 8
    
    # Solution optimale connue: objets 1 et 2 (indices 1, 2)
    # Valeur = 4 + 5 = 9, Poids = 3 + 4 = 7
    expected_value = 9
    
    print("=" * 60)
    print("TEST INSTANCE SIMPLE")
    print(f"Poids: {weights}")
    print(f"Valeurs: {values}")
    print(f"Capacité: {capacity}")
    print(f"Valeur optimale attendue: {expected_value}")
    print("=" * 60)
    
    results = {}
    
    # Brute-Force
    val, items = bruteforce_knapsack(values, weights, capacity)
    results['Brute-Force'] = val
    print(f"Brute-Force: val={val}, items={items}")
    
    # DP Bottom-Up
    val, items = knapsack_bottom_up(weights, values, capacity)
    results['DP-BU'] = val
    print(f"DP Bottom-Up: val={val}, items={items}")
    
    # DP Top-Down
    val, items = knapsack_top_down(weights, values, capacity)
    results['DP-TD'] = val
    print(f"DP Top-Down: val={val}, items={items}")
    
    # Branch and Bound
    items, val, wt = branch_and_bound_bfs(weights, values, capacity)
    results['B&B-BFS'] = val
    print(f"B&B BFS: val={val}, items={items}")
    
    items, val, wt = branch_and_bound_dfs(weights, values, capacity)
    results['B&B-DFS'] = val
    print(f"B&B DFS: val={val}, items={items}")
    
    # Greedy
    items, val, wt = greedy_algorithm_ratio(weights, values, capacity)
    results['Greedy'] = val
    print(f"Greedy: val={val}, items={items}")
    
    # FPTAS
    items, val, wt = fptas_knapsack(weights, values, capacity, eps=0.1)
    results['FPTAS'] = val
    print(f"FPTAS: val={val}, items={items}")
    
    # Vérification
    print("\n" + "-" * 40)
    print("VERIFICATION")
    all_passed = True
    
    for algo, val in results.items():
        if algo in ['Brute-Force', 'DP-BU', 'DP-TD', 'B&B-BFS', 'B&B-DFS']:
            # Algorithmes exacts
            if val == expected_value:
                print(f"[OK] {algo}: PASS")
            else:
                print(f"[FAIL] {algo}: FAIL (got {val}, expected {expected_value})")
                all_passed = False
        else:
            # Algorithmes d'approximation
            ratio = val / expected_value
            if ratio >= 0.5:  # Au moins 50% de l'optimal
                print(f"[OK] {algo}: PASS (ratio={ratio:.2f})")
            else:
                print(f"[WARN] {algo}: LOW (ratio={ratio:.2f})")
    
    return all_passed


def test_metaheuristics():
    """Test des métaheuristiques"""
    weights = [12, 2, 1, 1, 4, 8, 6, 3]
    values = [4, 2, 1, 2, 10, 3, 5, 4]
    capacity = 15
    
    print("\n" + "=" * 60)
    print("TEST MÉTAHEURISTIQUES")
    print("=" * 60)
    
    # Calculer l'optimal avec DP
    opt_val, _ = knapsack_bottom_up(weights, values, capacity)
    print(f"Valeur optimale (DP): {opt_val}")
    
    # Génétique
    items, val, wt = genetic_knapsack(weights, values, capacity)
    ratio = val / opt_val
    print(f"Génétique: val={val}, ratio={ratio:.2f}")
    
    # Colonie de fourmis
    items, val, wt = ant_colony_knapsack(weights, values, capacity)
    ratio = val / opt_val
    print(f"Colonie de Fourmis: val={val}, ratio={ratio:.2f}")
    
    # Randomisé
    items, val, wt = randomized_knapsack(weights, values, capacity)
    ratio = val / opt_val
    print(f"Randomisé: val={val}, ratio={ratio:.2f}")


def run_all_tests():
    """Exécute tous les tests"""
    print("\n" + "=" * 70)
    print("                    TESTS UNITAIRES KNAPSACK")
    print("=" * 70)
    
    test_small_instance()
    test_metaheuristics()
    
    print("\n" + "=" * 70)
    print("TESTS TERMINÉS")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
