"""
Algorithme de Colonie de Fourmis (ACO) pour le problème du Sac à Dos 0/1
Métaheuristique inspirée du comportement collectif des fourmis
"""

import random
from typing import List, Tuple


def ant_colony_knapsack(weights: List[int], values: List[int], capacity: int,
                        n_ants: int = None, n_iterations: int = None,
                        alpha: float = 1.0, beta: float = 5.0,
                        evaporation: float = 0.3, 
                        q: float = 100.0) -> Tuple[List[int], int, int]:
    """
    Algorithme de colonie de fourmis pour le knapsack 0/1.
    Parametres adaptes automatiquement a la taille du probleme.
    
    Args:
        weights: liste des poids
        values: liste des valeurs
        capacity: capacite du sac
        n_ants: nombre de fourmis (auto si None)
        n_iterations: nombre d'iterations (auto si None)
        alpha: influence des pheromones
        beta: influence heuristique (plus eleve = plus glouton)
        evaporation: taux d'evaporation
        q: quantite de pheromone deposee
        
    Returns:
        Tuple (indices_selectionnes, valeur_totale, poids_total)
    """
    n = len(weights)
    
    # Parametres adaptatifs - equilibre temps/qualite
    if n_ants is None:
        n_ants = min(50, max(15, n // 3))
    if n_iterations is None:
        n_iterations = min(100, max(30, n))
    
    if n == 0:
        return [], 0, 0
    
    # Initialisation
    pheromone = [1.0 for _ in range(n)]
    heuristic = [values[i] / weights[i] if weights[i] > 0 else 0 
                 for i in range(n)]
    
    best_solution = None
    best_value = 0
    
    for iteration in range(n_iterations):
        solutions = []
        
        # Chaque fourmi construit une solution
        for ant in range(n_ants):
            solution = [0] * n
            remaining = capacity
            
            # Trier par ratio valeur/poids avec un peu d'aleatoire
            items = list(range(n))
            # Premiere fourmi: greedy pur
            if ant == 0 and iteration == 0:
                items = sorted(items, key=lambda i: heuristic[i], reverse=True)
            else:
                random.shuffle(items)
            
            for i in items:
                if weights[i] > remaining:
                    continue
                
                tau = pheromone[i] ** alpha
                eta = (heuristic[i] + 0.001) ** beta
                
                # Probabilite plus agressive
                p = tau * eta
                total_p = p + 0.1
                
                if random.random() < (p / total_p):
                    solution[i] = 1
                    remaining -= weights[i]
            
            solutions.append(solution)
        
        # Evaporation
        for i in range(n):
            pheromone[i] *= (1 - evaporation)
            pheromone[i] = max(pheromone[i], 0.01)
        
        # Depot de pheromones (seulement les meilleures solutions)
        iteration_best_val = 0
        iteration_best_sol = None
        
        for sol in solutions:
            total_w = sum(weights[i] for i in range(n) if sol[i] == 1)
            if total_w > capacity:
                continue
            
            total_v = sum(values[i] for i in range(n) if sol[i] == 1)
            
            if total_v > iteration_best_val:
                iteration_best_val = total_v
                iteration_best_sol = sol
            
            if total_v > best_value:
                best_value = total_v
                best_solution = sol[:]
        
        # Depot renforce sur la meilleure solution de l'iteration
        if iteration_best_sol:
            for i in range(n):
                if iteration_best_sol[i] == 1:
                    pheromone[i] += q * iteration_best_val / best_value if best_value > 0 else q
    
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
