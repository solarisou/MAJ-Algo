"""
Système de Benchmark pour les Algorithmes Knapsack
Utilise les instances Pisinger depuis GitHub
"""

import time
import os
import sys
import random
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass

# Ajouter le chemin pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.instance_loader import PisingerInstanceLoader, KnapsackInstance


def generate_tiny_instances() -> List[KnapsackInstance]:
    """Génère des petites instances (n=10,15,20) pour tester le Brute-Force"""
    random.seed(42)  # Reproductibilité
    instances = []
    
    for n in [10, 15, 20]:
        weights = [random.randint(1, 50) for _ in range(n)]
        values = [random.randint(10, 100) for _ in range(n)]
        capacity = sum(weights) // 3
        
        # Calculer l'optimal avec DP
        optimal = _compute_dp_optimal(weights, values, capacity)
        
        instances.append(KnapsackInstance(
            name=f"tiny_{n}",
            n=n,
            capacity=capacity,
            values=values,
            weights=weights,
            optimal_value=optimal,
            optimal_solution=None
        ))
    
    return instances


def load_generated_instances() -> List[KnapsackInstance]:
    """Charge les instances depuis le dossier data/generated/"""
    instances = []
    generated_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'generated')
    
    if not os.path.exists(generated_dir):
        print(f"[INFO] Dossier {generated_dir} non trouvé")
        return instances
    
    for filename in os.listdir(generated_dir):
        if not filename.endswith('.txt'):
            continue
        
        filepath = os.path.join(generated_dir, filename)
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Format: n, capacity, optimal_value, puis paires (weight, value)
            n = int(lines[0].strip())
            capacity = int(lines[1].strip())
            
            # Vérifier si la ligne 3 est l'optimal ou déjà un item
            line3 = lines[2].strip().split()
            if len(line3) == 1:
                # Nouveau format avec optimal
                optimal_value = int(line3[0])
                start_items = 3
            else:
                # Ancien format sans optimal - calculer avec DP
                optimal_value = None
                start_items = 2
            
            weights = []
            values = []
            
            for line in lines[start_items:start_items + n]:
                parts = line.strip().split()
                if len(parts) >= 2:
                    weights.append(int(parts[0]))
                    values.append(int(parts[1]))
            
            # Si pas d'optimal, le calculer avec DP
            if optimal_value is None:
                optimal_value = _compute_dp_optimal(weights, values, capacity)
            
            instances.append(KnapsackInstance(
                name=f"generated_{filename.replace('.txt', '')}",
                n=n,
                capacity=capacity,
                values=values,
                weights=weights,
                optimal_value=optimal_value,
                optimal_solution=None
            ))
            print(f"  [OK] Chargé: {filename} (n={n}, OPT={optimal_value})")
            
        except Exception as e:
            print(f"  [ERREUR] {filename}: {e}")
    
    return instances


def _compute_dp_optimal(weights, values, capacity):
    """Calcule l'optimal avec DP (pour les fichiers sans optimal)"""
    n = len(weights)
    dp = [0] * (capacity + 1)
    for i in range(n):
        w, v = weights[i], values[i]
        for c in range(capacity, w - 1, -1):
            dp[c] = max(dp[c], dp[c - w] + v)
    return dp[capacity]


from src.utils.solution_validator import validate_solution, compare_with_optimal

from src.algorithms.bruteforce import bruteforce_knapsack
from src.algorithms.dynamic_programming import knapsack_bottom_up, knapsack_top_down
from src.algorithms.greedy import greedy_algorithm_ratio
from src.algorithms.branch_and_bound import (
    branch_and_bound_bfs, 
    branch_and_bound_dfs,
    branch_and_bound_least_cost
)
from src.algorithms.fractional_approximation import (
    fractional_knapsack_approximation, 
    fractional_knapsack_with_full_item
)
from src.algorithms.fptas import fptas_knapsack
from src.algorithms.genetic import genetic_knapsack
from src.algorithms.ant_colony import ant_colony_knapsack
from src.algorithms.randomized import randomized_knapsack


@dataclass
class BenchmarkResult:
    """Résultat d'un benchmark pour un algorithme"""
    algorithm: str
    instance_name: str
    n: int
    capacity: int
    value: int
    weight: int
    time: float
    is_valid: bool
    optimal_value: Optional[int] = None
    gap_percent: Optional[float] = None
    error: Optional[str] = None


class KnapsackBenchmark:
    """Système de benchmark pour les algorithmes Knapsack"""
    
    def __init__(self, data_dir: str = "data/pisinger_instances"):
        self.loader = PisingerInstanceLoader(data_dir)
        self.results: List[BenchmarkResult] = []
        self.algorithms = self._init_algorithms()
    
    def _init_algorithms(self) -> Dict[str, Callable]:
        """Initialise les algorithmes à tester"""
        
        def bruteforce_wrapper(w, v, c):
            # Augmente la limite pour ne jamais lancer le bruteforce (sauf tiny)
            if len(w) > 10:
                return None, None, None
            val, items = bruteforce_knapsack(v, w, c)
            wt = sum(w[i] for i in items)
            return items, val, wt
        
        def dp_bu_wrapper(w, v, c):
            val, items = knapsack_bottom_up(w, v, c)
            wt = sum(w[i] for i in items)
            return items, val, wt
        
        def dp_td_wrapper(w, v, c):
            val, items = knapsack_top_down(w, v, c)
            wt = sum(w[i] for i in items)
            return items, val, wt
        
        def fptas_wrapper(w, v, c):
            return fptas_knapsack(w, v, c, eps=0.1)
        
        return {
            'Brute-Force': bruteforce_wrapper,
            'DP-BottomUp': dp_bu_wrapper,
            'DP-TopDown': dp_td_wrapper,
            'Greedy-Ratio': greedy_algorithm_ratio,
            'B&B-BFS': branch_and_bound_bfs,
            'B&B-LeastCost': branch_and_bound_least_cost,
            'Fractional': fractional_knapsack_approximation,
            'Fractional+': fractional_knapsack_with_full_item,
            'FPTAS': fptas_wrapper,
            'Genetic': genetic_knapsack,
            'AntColony': ant_colony_knapsack,
            'Randomized': randomized_knapsack,
        }
    
    def run_single(self, algo_name: str, instance: KnapsackInstance,
                   timeout: float = 60.0) -> BenchmarkResult:
        """Exécute un seul algorithme sur une instance"""
        
        if algo_name not in self.algorithms:
            return BenchmarkResult(
                algorithm=algo_name,
                instance_name=instance.name,
                n=instance.n,
                capacity=instance.capacity,
                value=0, weight=0, time=0,
                is_valid=False,
                error="Algorithme inconnu"
            )
        
        algo = self.algorithms[algo_name]
        
        try:
            start = time.time()
            items, value, weight = algo(
                instance.weights, 
                instance.values, 
                instance.capacity
            )
            elapsed = time.time() - start
            
            if items is None:
                return BenchmarkResult(
                    algorithm=algo_name,
                    instance_name=instance.name,
                    n=instance.n,
                    capacity=instance.capacity,
                    value=0, weight=0, time=elapsed,
                    is_valid=False,
                    error="Skipped (trop grand)"
                )
            
            is_valid, actual_wt, actual_val = validate_solution(
                items, instance.values, instance.weights, instance.capacity
            )
            
            gap = None
            if instance.optimal_value:
                _, gap = compare_with_optimal(value, instance.optimal_value)
            
            return BenchmarkResult(
                algorithm=algo_name,
                instance_name=instance.name,
                n=instance.n,
                capacity=instance.capacity,
                value=value,
                weight=weight,
                time=elapsed,
                is_valid=is_valid,
                optimal_value=instance.optimal_value,
                gap_percent=gap
            )
            
        except Exception as e:
            return BenchmarkResult(
                algorithm=algo_name,
                instance_name=instance.name,
                n=instance.n,
                capacity=instance.capacity,
                value=0, weight=0, time=0,
                is_valid=False,
                error=str(e)
            )
    
    def run_benchmark(self, categories: List[str] = None,
                      algorithms: List[str] = None,
                      download_if_missing: bool = True) -> List[BenchmarkResult]:
        """
        Exécute le benchmark complet.
        
        Args:
            categories: catégories d'instances ('tiny', 'small', 'medium', 'large', 'generated')
            algorithms: liste des algorithmes à tester
            download_if_missing: télécharger les instances manquantes
        """
        if categories is None:
            categories = ['small', 'medium']
        
        if algorithms is None:
            algorithms = list(self.algorithms.keys())
        
        # Charger les instances
        print("=" * 70)
        print("BENCHMARK KNAPSACK")
        print("=" * 70)
        
        instances = []
        
        # Ajouter les instances tiny si demandé
        if 'tiny' in categories:
            tiny = generate_tiny_instances()
            instances.extend(tiny)
            print(f"\n{len(tiny)} instances tiny générées pour Brute-Force")
            categories = [c for c in categories if c != 'tiny']
        
        # Ajouter les instances generated si demandé
        if 'generated' in categories:
            print("\nChargement des instances générées...")
            generated = load_generated_instances()
            instances.extend(generated)
            print(f"{len(generated)} instances generated chargées")
            categories = [c for c in categories if c != 'generated']
        
        # Charger les instances Pisinger (small, medium, large)
        if categories:
            pisinger_instances = self.loader.get_all_instances(
                categories, 
                download_if_missing=download_if_missing
            )
            instances.extend(pisinger_instances)
        
        if not instances:
            print("[ERREUR] Aucune instance trouvée!")
            return []
        
        print(f"\n{len(instances)} instances chargées au total")
        print(f"{len(algorithms)} algorithmes à tester\n")
        
        results = []
        
        for instance in instances:
            print(f"\nInstance: {instance.name} (n={instance.n})")
            print("-" * 50)
            
            for algo_name in algorithms:
                if algo_name not in self.algorithms:
                    continue
                
                result = self.run_single(algo_name, instance)
                results.append(result)
                
                # Affichage
                if result.error:
                    status = f"[SKIP] {result.error}"
                elif result.is_valid:
                    gap_str = f", gap={result.gap_percent:.2f}%" if result.gap_percent is not None else ""
                    status = f"[OK] val={result.value}, t={result.time:.4f}s{gap_str}"
                else:
                    status = f"[FAIL] INVALID"
                
                print(f"  {algo_name:15s}: {status}")
        
        self.results = results
        return results
    
    def save_results(self, filepath: str = None):
        """Sauvegarde les résultats au format CSV"""
        import csv
        
        # Chemin absolu dans le dossier Projet-algo/results/
        if filepath is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            filepath = os.path.join(base_dir, 'results', 'benchmark_results.csv')
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'algorithm', 'instance', 'n', 'capacity',
                'value', 'weight', 'time', 'is_valid',
                'optimal', 'gap_percent', 'error'
            ])
            
            for r in self.results:
                writer.writerow([
                    r.algorithm, r.instance_name, r.n, r.capacity,
                    r.value, r.weight, f"{r.time:.6f}", r.is_valid,
                    r.optimal_value or '', 
                    f"{r.gap_percent:.4f}" if r.gap_percent is not None else '',
                    r.error or ''
                ])
        
        print(f"\nRésultats sauvegardés: {filepath}")
    
    def print_summary(self):
        """Affiche un résumé des résultats"""
        if not self.results:
            print("Aucun résultat à afficher")
            return
        
        print("\n" + "=" * 70)
        print("RÉSUMÉ DU BENCHMARK")
        print("=" * 70)
        
        # Grouper par algorithme
        algo_stats = {}
        for r in self.results:
            if r.algorithm not in algo_stats:
                algo_stats[r.algorithm] = {
                    'count': 0, 'valid': 0, 'total_time': 0,
                    'gaps': []
                }
            
            stats = algo_stats[r.algorithm]
            if not r.error:
                stats['count'] += 1
                if r.is_valid:
                    stats['valid'] += 1
                stats['total_time'] += r.time
                if r.gap_percent is not None:
                    stats['gaps'].append(r.gap_percent)
        
        print(f"\n{'Algorithme':<20} {'Valides':<12} {'Temps Moy':<12} {'Gap Moy':<12}")
        print("-" * 56)
        
        for algo, stats in algo_stats.items():
            if stats['count'] > 0:
                avg_time = stats['total_time'] / stats['count']
                avg_gap = sum(stats['gaps']) / len(stats['gaps']) if stats['gaps'] else None
                
                gap_str = f"{avg_gap:.2f}%" if avg_gap is not None else "N/A"
                print(f"{algo:<20} {stats['valid']}/{stats['count']:<10} {avg_time:.4f}s      {gap_str}")


def run_quick_benchmark():
    """Exécute un benchmark rapide sur les petites instances avec tous les algorithmes"""
    benchmark = KnapsackBenchmark()
    
    # Petites instances seulement mais avec TOUS les algorithmes
    benchmark.run_benchmark(
        categories=['small']
        # Pas de filtre sur algorithms = tous les algorithmes
    )
    
    benchmark.print_summary()
    benchmark.save_results()


def run_full_benchmark():
    """Exécute le benchmark complet"""
    benchmark = KnapsackBenchmark()
    
    benchmark.run_benchmark(
        categories=['small', 'medium', 'large']
    )
    
    benchmark.print_summary()
    benchmark.save_results()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Knapsack")
    parser.add_argument('--quick', action='store_true', 
                        help="Benchmark rapide (small only)")
    parser.add_argument('--full', action='store_true',
                        help="Benchmark complet")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_benchmark()
    else:
        run_full_benchmark()
