"""
Algorithme de Colonie de Fourmis (ACO) pour le problème du Sac à Dos 0/1
Métaheuristique inspirée du comportement collectif des fourmis
"""

import random
from typing import List, Tuple


def ant_colony_knapsack(weights: List[int], values: List[int], capacity: int,
                        n_ants: int = 40, n_iterations: int = 200,
                        alpha: float = 1.0, beta: float = 3.0,
                        evaporation: float = 0.5, 
                        q: float = 1.0) -> Tuple[List[int], int, int]:
    """
    Algorithme de colonie de fourmis pour le knapsack 0/1.
    
    Args:
        weights: liste des poids
        values: liste des valeurs
        capacity: capacité du sac
        n_ants: nombre de fourmis
        n_iterations: nombre d'itérations
        alpha: influence des phéromones
        beta: influence heuristique
        evaporation: taux d'évaporation
        q: quantité de phéromone déposée
        
    Returns:
        Tuple (indices_sélectionnés, valeur_totale, poids_total)
    """
    n = len(weights)
    
    if n == 0:
        return [], 0, 0
    
    # Initialisation
    pheromone = [1.0 for _ in range(n)]
    heuristic = [values[i] / weights[i] if weights[i] > 0 else 0 
                 for i in range(n)]
    
    best_solution = None
    best_value = 0
    
    for _ in range(n_iterations):
        solutions = []
        
        # Chaque fourmi construit une solution
        for _ in range(n_ants):
            solution = [0] * n
            remaining = capacity
            
            items = list(range(n))
            random.shuffle(items)
            
            for i in items:
                if weights[i] > remaining:
                    continue
                
                tau = pheromone[i] ** alpha
                eta = heuristic[i] ** beta
                p = tau * eta
                
                if random.random() < (p / (1 + p)):
                    solution[i] = 1
                    remaining -= weights[i]
            
            solutions.append(solution)
        
        # Évaporation
        for i in range(n):
            pheromone[i] *= (1 - evaporation)
            if pheromone[i] < 1e-6:
                pheromone[i] = 1e-6
        
        # Dépôt de phéromones
        for sol in solutions:
            total_w = sum(weights[i] for i in range(n) if sol[i] == 1)
            if total_w > capacity:
                continue
            
            total_v = sum(values[i] for i in range(n) if sol[i] == 1)
            
            if total_v > best_value:
                best_value = total_v
                best_solution = sol[:]
            
            for i in range(n):
                if sol[i] == 1:
                    pheromone[i] += q * (total_v / (1 + total_w))
    
    if best_solution is None:
        return [], 0, 0
    
    selected = [i for i in range(n) if best_solution[i] == 1]
    total_weight = sum(weights[i] for i in selected)
    
    return selected, best_value, total_weight


# --- Exemple d'utilisation ---
if __name__ == "__main__":
    weights = [12, 2, 1, 1, 4]
    values = [4, 2, 1, 2, 10]
    capacity = 15
    
    print("=== Algorithme de Colonie de Fourmis ===")
    items, val, wt = ant_colony_knapsack(weights, values, capacity)
    print(f"Objets: {items}, Valeur: {val}, Poids: {wt}")
