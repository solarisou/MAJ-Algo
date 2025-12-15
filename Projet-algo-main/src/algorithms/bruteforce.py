"""
Algorithme Brute-Force pour le problème du Sac à Dos 0/1
Exploration exhaustive de toutes les combinaisons possibles
"""

from typing import List, Tuple


def bruteforce_knapsack(values: List[int], weights: List[int], 
                        capacity: int, index: int = 0) -> Tuple[int, List[int]]:
    """
    Résout le problème du sac à dos 0/1 par force brute (récursif).
    
    Complexité: O(2^n) - Ne pas utiliser pour n > 25
    
    Args:
        values: liste des valeurs des objets
        weights: liste des poids des objets
        capacity: capacité maximale du sac
        index: index actuel (pour la récursion)
        
    Returns:
        Tuple (valeur_maximale, liste_indices_sélectionnés)
    """
    # Cas de base: tous les objets traités ou capacité épuisée
    if index == len(values) or capacity == 0:
        return 0, []

    # CAS 1: Ne pas prendre l'objet actuel
    value_without, items_without = bruteforce_knapsack(values, weights, capacity, index + 1)

    # CAS 2: Prendre l'objet actuel (si possible)
    if weights[index] <= capacity:
        value_with, items_with = bruteforce_knapsack(
            values, weights, 
            capacity - weights[index], 
            index + 1
        )
        value_with += values[index]
        items_with = [index] + items_with
    else:
        value_with = -1
        items_with = []

    # Retourner la meilleure option
    if value_with > value_without:
        return value_with, items_with
    else:
        return value_without, items_without


def bruteforce_knapsack_iterative(values: List[int], weights: List[int], 
                                   capacity: int) -> Tuple[int, List[int]]:
    """
    Version itérative du brute-force utilisant les masques binaires.
    
    Plus efficace en mémoire mais même complexité temporelle O(2^n).
    
    Args:
        values: liste des valeurs des objets
        weights: liste des poids des objets
        capacity: capacité maximale du sac
        
    Returns:
        Tuple (valeur_maximale, liste_indices_sélectionnés)
    """
    n = len(values)
    best_value = 0
    best_items = []
    
    # Parcourir toutes les 2^n combinaisons
    for mask in range(1 << n):
        total_weight = 0
        total_value = 0
        items = []
        
        for i in range(n):
            if mask & (1 << i):
                total_weight += weights[i]
                total_value += values[i]
                items.append(i)
        
        # Vérifier si la solution est valide et meilleure
        if total_weight <= capacity and total_value > best_value:
            best_value = total_value
            best_items = items
    
    return best_value, best_items


# --- Exemple d'utilisation ---
if __name__ == "__main__":
    # Test
    values = [60, 100, 120]
    weights = [10, 20, 30]
    capacity = 50
    
    print("=== Brute-Force Récursif ===")
    val, items = bruteforce_knapsack(values, weights, capacity)
    print(f"Valeur maximale: {val}")
    print(f"Objets sélectionnés: {items}")
    
    print("\n=== Brute-Force Itératif ===")
    val, items = bruteforce_knapsack_iterative(values, weights, capacity)
    print(f"Valeur maximale: {val}")
    print(f"Objets sélectionnés: {items}")
