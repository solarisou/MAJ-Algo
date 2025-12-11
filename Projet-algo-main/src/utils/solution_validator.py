"""
Module de validation des solutions Knapsack
"""

from typing import List, Tuple


def validate_solution(selected_items: List[int], 
                     values: List[int], 
                     weights: List[int], 
                     capacity: int) -> Tuple[bool, int, int]:
    """
    Vérifie qu'une solution est valide et calcule sa valeur.
    
    Args:
        selected_items: Liste des indices des objets sélectionnés
        values: Liste des valeurs de tous les objets
        weights: Liste des poids de tous les objets
        capacity: Capacité du sac à dos
        
    Returns:
        Tuple (is_valid, total_weight, total_value)
    """
    if not selected_items:
        return True, 0, 0
    
    total_weight = sum(weights[i] for i in selected_items if i < len(weights))
    total_value = sum(values[i] for i in selected_items if i < len(values))
    is_valid = total_weight <= capacity
    
    return is_valid, total_weight, total_value


def calculate_solution_value(selected_items: List[int], 
                            values: List[int], 
                            weights: List[int], 
                            capacity: int) -> Tuple[int, int, bool]:
    """
    Calcule la valeur et le poids d'une solution.
    
    Args:
        selected_items: Liste des indices sélectionnés
        values: Liste des valeurs
        weights: Liste des poids
        capacity: Capacité
        
    Returns:
        Tuple (total_value, total_weight, is_feasible)
    """
    is_valid, total_weight, total_value = validate_solution(
        selected_items, values, weights, capacity
    )
    return total_value, total_weight, is_valid


def compare_with_optimal(found_value: int, 
                        optimal_value: int) -> Tuple[float, float]:
    """
    Compare une solution trouvée avec l'optimal.
    
    Args:
        found_value: Valeur de la solution trouvée
        optimal_value: Valeur optimale
        
    Returns:
        Tuple (ratio, gap_percent)
            - ratio: found_value / optimal_value (1.0 = optimal)
            - gap_percent: écart en pourcentage
    """
    if optimal_value == 0:
        return 1.0, 0.0
    
    ratio = found_value / optimal_value
    gap_percent = (optimal_value - found_value) / optimal_value * 100
    
    return ratio, gap_percent


def solution_indices_to_binary(indices: List[int], n: int) -> List[int]:
    """
    Convertit une liste d'indices en solution binaire.
    
    Args:
        indices: Liste des indices sélectionnés
        n: Nombre total d'objets
        
    Returns:
        Liste binaire de taille n
    """
    binary = [0] * n
    for i in indices:
        if 0 <= i < n:
            binary[i] = 1
    return binary


def solution_binary_to_indices(binary: List[int]) -> List[int]:
    """
    Convertit une solution binaire en liste d'indices.
    
    Args:
        binary: Liste binaire (0/1)
        
    Returns:
        Liste des indices où binary[i] == 1
    """
    return [i for i, x in enumerate(binary) if x == 1]
