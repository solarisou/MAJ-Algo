"""
Algorithme Randomisé pour le problème du Sac à Dos 0/1
Approche probabiliste avec répétitions
"""

import random
from typing import List, Tuple


def randomized_knapsack(weights: List[int], values: List[int], capacity: int,
                        iterations: int = 500) -> Tuple[List[int], int, int]:
    """
    Algorithme randomisé pour le knapsack 0/1.
    
    Exécute un greedy dans un ordre aléatoire plusieurs fois
    et garde la meilleure solution.
    
    Args:
        weights: liste des poids
        values: liste des valeurs
        capacity: capacité du sac
        iterations: nombre d'itérations
        
    Returns:
        Tuple (indices_sélectionnés, valeur_totale, poids_total)
    """
    n = len(weights)
    
    if n == 0:
        return [], 0, 0
    
    best_value = 0
    best_selection = []
    
    for _ in range(iterations):
        # Ordre aléatoire des objets
        order = list(range(n))
        random.shuffle(order)
        
        total_weight = 0
        total_value = 0
        selection = []
        
        # Greedy dans l'ordre aléatoire
        for i in order:
            if total_weight + weights[i] <= capacity:
                total_weight += weights[i]
                total_value += values[i]
                selection.append(i)
        
        if total_value > best_value:
            best_value = total_value
            best_selection = selection
    
    best_weight = sum(weights[i] for i in best_selection)
    return best_selection, best_value, best_weight


def randomized_knapsack_weighted(weights: List[int], values: List[int], 
                                  capacity: int,
                                  iterations: int = 500) -> Tuple[List[int], int, int]:
    """
    Version améliorée avec probabilités pondérées par ratio.
    
    Les objets avec meilleur ratio ont plus de chances d'être choisis en premier.
    """
    n = len(weights)
    
    if n == 0:
        return [], 0, 0
    
    # Calculer les ratios
    ratios = [values[i] / weights[i] if weights[i] > 0 else 0 for i in range(n)]
    total_ratio = sum(ratios) or 1
    
    best_value = 0
    best_selection = []
    
    for _ in range(iterations):
        remaining = list(range(n))
        remaining_capacity = capacity
        
        total_value = 0
        selection = []
        
        while remaining and remaining_capacity > 0:
            # Calculer probabilités
            probs = [ratios[i] / total_ratio for i in remaining]
            
            # Choisir un objet selon les probabilités
            r = random.random()
            cumsum = 0
            chosen_idx = 0
            
            for idx, p in enumerate(probs):
                cumsum += p
                if r <= cumsum:
                    chosen_idx = idx
                    break
            
            i = remaining[chosen_idx]
            
            if weights[i] <= remaining_capacity:
                selection.append(i)
                total_value += values[i]
                remaining_capacity -= weights[i]
            
            remaining.pop(chosen_idx)
        
        if total_value > best_value:
            best_value = total_value
            best_selection = selection
    
    best_weight = sum(weights[i] for i in best_selection)
    return best_selection, best_value, best_weight


# --- Exemple d'utilisation ---
if __name__ == "__main__":
    weights = [12, 2, 1, 1, 4]
    values = [4, 2, 1, 2, 10]
    capacity = 15
    
    print("=== Algorithme Randomisé Simple ===")
    items, val, wt = randomized_knapsack(weights, values, capacity)
    print(f"Objets: {items}, Valeur: {val}, Poids: {wt}")
    
    print("\n=== Algorithme Randomisé Pondéré ===")
    items, val, wt = randomized_knapsack_weighted(weights, values, capacity)
    print(f"Objets: {items}, Valeur: {val}, Poids: {wt}")
