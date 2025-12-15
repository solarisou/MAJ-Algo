"""
Algorithmes d'Approximation basés sur le Knapsack Fractionnaire
"""

from typing import List, Tuple


def fractional_knapsack_approximation(weights: List[int], values: List[int],
                                      capacity: int) -> Tuple[List[int], int, int]:
    """
    Approximation basée sur la solution fractionnaire.
    
    Garantit un ratio d'approximation de 1/2.
    
    Args:
        weights: liste des poids
        values: liste des valeurs
        capacity: capacité du sac
        
    Returns:
        Tuple (indices_sélectionnés, valeur_totale, poids_total)
    """
    n = len(weights)
    
    if n == 0:
        return [], 0, 0
    
    # Objets avec indices et ratios
    items = [(i, values[i], weights[i], 
              values[i] / weights[i] if weights[i] > 0 else float('inf'))
             for i in range(n)]
    
    # Trier par ratio décroissant
    items.sort(key=lambda x: x[3], reverse=True)
    
    # Solution 1: Greedy basé sur ratio
    selected = []
    total_weight = 0
    total_value = 0
    
    for idx, val, wt, ratio in items:
        if total_weight + wt <= capacity:
            selected.append(idx)
            total_weight += wt
            total_value += val
    
    # Solution 2: Meilleur objet unique
    best_idx = max(range(n), key=lambda i: values[i] if weights[i] <= capacity else 0)
    
    if weights[best_idx] <= capacity:
        single_value = values[best_idx]
        single_weight = weights[best_idx]
    else:
        single_value = 0
        single_weight = 0
    
    # Retourner la meilleure solution
    if total_value >= single_value:
        return selected, total_value, total_weight
    else:
        return [best_idx] if single_value > 0 else [], single_value, single_weight


def fractional_knapsack_with_full_item(weights: List[int], values: List[int],
                                        capacity: int) -> Tuple[List[int], int, int]:
    """
    Amélioration: essaie chaque objet comme "spécial".
    
    Meilleur ratio d'approximation.
    
    Args:
        weights: liste des poids
        values: liste des valeurs
        capacity: capacité du sac
        
    Returns:
        Tuple (indices_sélectionnés, valeur_totale, poids_total)
    """
    n = len(weights)
    
    if n == 0:
        return [], 0, 0
    
    best = ([], 0, 0)
    
    items = [(i, values[i], weights[i],
              values[i] / weights[i] if weights[i] > 0 else float('inf'))
             for i in range(n)]
    items.sort(key=lambda x: x[3], reverse=True)
    
    # Essayer chaque objet comme spécial
    for special_idx in range(n):
        if weights[special_idx] > capacity:
            continue
        
        selected = [special_idx]
        total_weight = weights[special_idx]
        total_value = values[special_idx]
        
        for idx, val, wt, ratio in items:
            if idx == special_idx:
                continue
            if total_weight + wt <= capacity:
                selected.append(idx)
                total_weight += wt
                total_value += val
        
        if total_value > best[1]:
            best = (selected, total_value, total_weight)
    
    # Aussi essayer greedy pur
    selected = []
    total_weight = 0
    total_value = 0
    
    for idx, val, wt, ratio in items:
        if total_weight + wt <= capacity:
            selected.append(idx)
            total_weight += wt
            total_value += val
    
    if total_value > best[1]:
        best = (selected, total_value, total_weight)
    
    return best


# --- Exemple d'utilisation ---
if __name__ == "__main__":
    weights = [12, 2, 1, 1, 4]
    values = [4, 2, 1, 2, 10]
    capacity = 15
    
    print("=== Approximation Fractionnaire Simple ===")
    items, val, wt = fractional_knapsack_approximation(weights, values, capacity)
    print(f"Objets: {items}, Valeur: {val}, Poids: {wt}")
    
    print("\n=== Approximation Fractionnaire Améliorée ===")
    items, val, wt = fractional_knapsack_with_full_item(weights, values, capacity)
    print(f"Objets: {items}, Valeur: {val}, Poids: {wt}")
