"""
Algorithmes Gloutons (Greedy) pour le problème du Sac à Dos 0/1
Différentes stratégies de sélection
"""

from typing import List, Tuple


def greedy_algorithm_ratio(weights: List[int], values: List[int], 
                          capacity: int) -> Tuple[List[int], int, int]:
    """
    Algorithme glouton basé sur le ratio valeur/poids.
    
    Sélectionne les objets par ordre décroissant de ratio valeur/poids.
    Meilleure approximation parmi les méthodes gloutonnes.
    
    Complexité: O(n log n)
    
    Args:
        weights: liste des poids des objets
        values: liste des valeurs des objets
        capacity: capacité maximale du sac
        
    Returns:
        Tuple (indices_sélectionnés, valeur_totale, poids_total)
    """
    n = len(weights)
    
    if n == 0:
        return [], 0, 0
    
    # Créer liste d'indices avec leurs ratios
    items = [(i, values[i] / weights[i] if weights[i] > 0 else float('inf'))
             for i in range(n)]
    
    # Trier par ratio décroissant
    items.sort(key=lambda x: x[1], reverse=True)
    
    selected = []
    total_value = 0
    total_weight = 0
    
    for idx, ratio in items:
        if total_weight + weights[idx] <= capacity:
            selected.append(idx)
            total_weight += weights[idx]
            total_value += values[idx]
    
    return selected, total_value, total_weight


def greedy_algorithm_value(weights: List[int], values: List[int], 
                          capacity: int) -> Tuple[List[int], int, int]:
    """
    Algorithme glouton basé sur la valeur.
    
    Sélectionne les objets par ordre décroissant de valeur.
    
    Complexité: O(n log n)
    
    Args:
        weights: liste des poids des objets
        values: liste des valeurs des objets
        capacity: capacité maximale du sac
        
    Returns:
        Tuple (indices_sélectionnés, valeur_totale, poids_total)
    """
    n = len(weights)
    
    if n == 0:
        return [], 0, 0
    
    # Créer liste d'indices triés par valeur décroissante
    items = sorted(range(n), key=lambda i: values[i], reverse=True)
    
    selected = []
    total_value = 0
    total_weight = 0
    
    for idx in items:
        if total_weight + weights[idx] <= capacity:
            selected.append(idx)
            total_weight += weights[idx]
            total_value += values[idx]
    
    return selected, total_value, total_weight


def greedy_algorithm_weight(weights: List[int], values: List[int], 
                           capacity: int) -> Tuple[List[int], int, int]:
    """
    Algorithme glouton basé sur le poids.
    
    Sélectionne les objets par ordre croissant de poids.
    Maximise le nombre d'objets.
    
    Complexité: O(n log n)
    
    Args:
        weights: liste des poids des objets
        values: liste des valeurs des objets
        capacity: capacité maximale du sac
        
    Returns:
        Tuple (indices_sélectionnés, valeur_totale, poids_total)
    """
    n = len(weights)
    
    if n == 0:
        return [], 0, 0
    
    # Créer liste d'indices triés par poids croissant
    items = sorted(range(n), key=lambda i: weights[i])
    
    selected = []
    total_value = 0
    total_weight = 0
    
    for idx in items:
        if total_weight + weights[idx] <= capacity:
            selected.append(idx)
            total_weight += weights[idx]
            total_value += values[idx]
    
    return selected, total_value, total_weight


def greedy_best_of_three(weights: List[int], values: List[int], 
                         capacity: int) -> Tuple[List[int], int, int]:
    """
    Exécute les trois stratégies gloutonnes et retourne la meilleure.
    
    Args:
        weights: liste des poids des objets
        values: liste des valeurs des objets
        capacity: capacité maximale du sac
        
    Returns:
        Tuple (indices_sélectionnés, valeur_totale, poids_total)
    """
    results = [
        greedy_algorithm_ratio(weights, values, capacity),
        greedy_algorithm_value(weights, values, capacity),
        greedy_algorithm_weight(weights, values, capacity),
    ]
    
    # Retourner celle avec la meilleure valeur
    return max(results, key=lambda x: x[1])


# --- Exemple d'utilisation ---
if __name__ == "__main__":
    # Test
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50
    
    print("=== Greedy par Ratio Valeur/Poids ===")
    items, val, wt = greedy_algorithm_ratio(weights, values, capacity)
    print(f"Objets: {items}, Valeur: {val}, Poids: {wt}")
    
    print("\n=== Greedy par Valeur ===")
    items, val, wt = greedy_algorithm_value(weights, values, capacity)
    print(f"Objets: {items}, Valeur: {val}, Poids: {wt}")
    
    print("\n=== Greedy par Poids ===")
    items, val, wt = greedy_algorithm_weight(weights, values, capacity)
    print(f"Objets: {items}, Valeur: {val}, Poids: {wt}")
    
    print("\n=== Meilleur des Trois ===")
    items, val, wt = greedy_best_of_three(weights, values, capacity)
    print(f"Objets: {items}, Valeur: {val}, Poids: {wt}")
