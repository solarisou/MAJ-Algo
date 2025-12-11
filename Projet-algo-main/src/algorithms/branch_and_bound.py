"""
Algorithme Branch-and-Bound pour le problème du Sac à Dos 0/1
Implémentations BFS (Best-First Search) et DFS (Depth-First Search)
"""

import heapq
from typing import List, Tuple


class Node:
    """Nœud dans l'arbre de recherche Branch-and-Bound"""
    
    def __init__(self, level: int, value: int, weight: int, 
                 bound: float, items_selected: List[int]):
        self.level = level
        self.value = value
        self.weight = weight
        self.bound = bound
        self.items_selected = items_selected
    
    def __lt__(self, other):
        return self.bound > other.bound


def _fractional_bound(node: Node, n: int, capacity: int,
                      weights: List[int], values: List[int],
                      ratio_items: List[Tuple[int, float]]) -> float:
    """
    Calcule la borne supérieure avec la relaxation fractionnaire.
    """
    if node.weight >= capacity:
        return 0
    
    bound = node.value
    remaining = capacity - node.weight
    
    for idx, ratio in ratio_items:
        if idx <= node.level:
            continue
        
        if weights[idx] <= remaining:
            remaining -= weights[idx]
            bound += values[idx]
        else:
            bound += remaining * ratio
            break
    
    return bound


def branch_and_bound_knapsack(weights: List[int], values: List[int],
                               capacity: int, use_best_first: bool = True) -> Tuple[List[int], int, int]:
    """
    Branch-and-Bound pour le knapsack 0/1.
    
    Args:
        weights: liste des poids
        values: liste des valeurs
        capacity: capacité du sac
        use_best_first: True pour BFS, False pour DFS
        
    Returns:
        Tuple (indices_sélectionnés, valeur_totale, poids_total)
    """
    n = len(weights)
    
    if n == 0:
        return [], 0, 0
    
    # Ratios triés par ordre décroissant
    ratio_items = [(i, values[i] / weights[i] if weights[i] > 0 else float('inf'))
                   for i in range(n)]
    ratio_items.sort(key=lambda x: x[1], reverse=True)
    
    best_value = 0
    best_items = []
    
    # File de priorité ou pile
    if use_best_first:
        queue = []
        heapq.heappush(queue, Node(-1, 0, 0, float('inf'), []))
    else:
        queue = [Node(-1, 0, 0, float('inf'), [])]
    
    while queue:
        if use_best_first:
            current = heapq.heappop(queue)
        else:
            current = queue.pop()
        
        if current.bound <= best_value:
            continue
        
        if current.level == n - 1:
            continue
        
        next_level = current.level + 1
        
        # Branche: inclure l'objet
        if current.weight + weights[next_level] <= capacity:
            new_value = current.value + values[next_level]
            new_weight = current.weight + weights[next_level]
            new_items = current.items_selected + [next_level]
            
            if new_value > best_value:
                best_value = new_value
                best_items = new_items
            
            node_with = Node(next_level, new_value, new_weight, 0, new_items)
            node_with.bound = _fractional_bound(node_with, n, capacity,
                                                 weights, values, ratio_items)
            
            if node_with.bound > best_value:
                if use_best_first:
                    heapq.heappush(queue, node_with)
                else:
                    queue.append(node_with)
        
        # Branche: exclure l'objet
        node_without = Node(next_level, current.value, current.weight,
                           0, current.items_selected)
        node_without.bound = _fractional_bound(node_without, n, capacity,
                                                weights, values, ratio_items)
        
        if node_without.bound > best_value:
            if use_best_first:
                heapq.heappush(queue, node_without)
            else:
                queue.append(node_without)
    
    total_weight = sum(weights[i] for i in best_items)
    return best_items, best_value, total_weight


def branch_and_bound_bfs(weights: List[int], values: List[int],
                         capacity: int) -> Tuple[List[int], int, int]:
    """
    Branch-and-Bound avec Best-First Search.
    Explore les nœuds les plus prometteurs en premier.
    """
    return branch_and_bound_knapsack(weights, values, capacity, use_best_first=True)


def branch_and_bound_dfs(weights: List[int], values: List[int],
                         capacity: int) -> Tuple[List[int], int, int]:
    """
    Branch-and-Bound avec Depth-First Search.
    Explore en profondeur d'abord.
    """
    return branch_and_bound_knapsack(weights, values, capacity, use_best_first=False)


# =============================================================================
# VERSION 2: Branch-and-Bound avec Least Cost (LC) et borne Martello-Toth
# =============================================================================

def _compute_martello_toth_bound(level: int, current_value: int, current_weight: int,
                                   capacity: int, weights: List[int], values: List[int],
                                   sorted_indices: List[int]) -> float:
    """
    Borne supérieure de Martello-Toth (U2).
    Plus serrée que la simple relaxation fractionnaire.
    
    Utilise une combinaison de:
    - Relaxation fractionnaire standard
    - Amélioration par considération de l'objet critique
    """
    n = len(weights)
    remaining_capacity = capacity - current_weight
    bound = current_value
    
    if remaining_capacity <= 0:
        return current_value
    
    # Trouver l'objet critique (premier objet qui ne rentre pas)
    total_weight = 0
    critical_index = -1
    
    for i in range(level + 1, n):
        idx = sorted_indices[i]
        if total_weight + weights[idx] <= remaining_capacity:
            total_weight += weights[idx]
            bound += values[idx]
        else:
            critical_index = i
            break
    
    if critical_index == -1:
        return bound  # Tous les objets rentrent
    
    critical_idx = sorted_indices[critical_index]
    remaining_after = remaining_capacity - total_weight
    
    # Borne U1: relaxation fractionnaire classique
    u1 = bound + (remaining_after * values[critical_idx]) / weights[critical_idx]
    
    # Borne U2: Martello-Toth - considère remplacer le dernier objet pris
    u2 = bound
    if critical_index > level + 1 and critical_index < n:
        # Valeur si on prend l'objet critique à la place du dernier
        prev_idx = sorted_indices[critical_index - 1] if critical_index > 0 else -1
        if prev_idx >= 0:
            space_if_remove_prev = remaining_after + weights[prev_idx]
            if weights[critical_idx] <= space_if_remove_prev:
                u2 = bound - values[prev_idx] + values[critical_idx]
    
    return max(u1, u2)


class LCNode:
    """Nœud pour Least Cost Branch-and-Bound avec coût estimé"""
    
    def __init__(self, level: int, value: int, weight: int, 
                 bound: float, cost: float, path: List[int], decisions: List[bool]):
        self.level = level
        self.value = value
        self.weight = weight
        self.bound = bound
        self.cost = cost  # -bound pour min-heap (maximisation)
        self.path = path
        self.decisions = decisions
    
    def __lt__(self, other):
        return self.cost < other.cost


def branch_and_bound_least_cost(weights: List[int], values: List[int],
                                 capacity: int) -> Tuple[List[int], int, int]:
    """
    Branch-and-Bound avec stratégie Least Cost (LC) et borne Martello-Toth.
    
    Différences avec les versions BFS/DFS standard:
    1. Utilise la borne Martello-Toth (U2) plus serrée
    2. Stratégie Least-Cost: explore par coût croissant (-bound)
    3. Élagage plus agressif grâce à la borne améliorée
    
    Cette version est généralement plus efficace pour les grandes instances
    car elle élimine plus de branches.
    
    Complexité: O(2^n) pire cas, mais souvent bien meilleur en pratique
    
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
    
    # Trier par ratio décroissant
    indices_with_ratio = [(i, values[i] / weights[i] if weights[i] > 0 else float('inf'))
                          for i in range(n)]
    indices_with_ratio.sort(key=lambda x: x[1], reverse=True)
    sorted_indices = [i for i, _ in indices_with_ratio]
    
    # Créer mapping inverse pour reconstruire la solution
    index_map = {sorted_indices[i]: i for i in range(n)}
    
    best_value = 0
    best_items = []
    
    # Initialisation avec borne initiale
    initial_bound = _compute_martello_toth_bound(-1, 0, 0, capacity, weights, values, sorted_indices)
    
    # Min-heap avec coût = -bound
    pq = []
    heapq.heappush(pq, LCNode(-1, 0, 0, initial_bound, -initial_bound, [], []))
    
    nodes_explored = 0
    nodes_pruned = 0
    
    while pq:
        current = heapq.heappop(pq)
        nodes_explored += 1
        
        # Élagage: si la borne est inférieure à la meilleure solution
        if current.bound <= best_value:
            nodes_pruned += 1
            continue
        
        # Nœud feuille
        if current.level == n - 1:
            if current.value > best_value:
                best_value = current.value
                best_items = current.path.copy()
            continue
        
        next_level = current.level + 1
        next_original_idx = sorted_indices[next_level]
        
        # Branche 1: INCLURE l'objet (explorer en premier si prometteur)
        if current.weight + weights[next_original_idx] <= capacity:
            new_value = current.value + values[next_original_idx]
            new_weight = current.weight + weights[next_original_idx]
            new_path = current.path + [next_original_idx]
            new_decisions = current.decisions + [True]
            
            if new_value > best_value:
                best_value = new_value
                best_items = new_path.copy()
            
            new_bound = _compute_martello_toth_bound(
                next_level, new_value, new_weight, capacity, weights, values, sorted_indices
            )
            
            if new_bound > best_value:
                heapq.heappush(pq, LCNode(
                    next_level, new_value, new_weight, new_bound, -new_bound,
                    new_path, new_decisions
                ))
            else:
                nodes_pruned += 1
        
        # Branche 2: EXCLURE l'objet
        new_decisions = current.decisions + [False]
        new_bound = _compute_martello_toth_bound(
            next_level, current.value, current.weight, capacity, weights, values, sorted_indices
        )
        
        if new_bound > best_value:
            heapq.heappush(pq, LCNode(
                next_level, current.value, current.weight, new_bound, -new_bound,
                current.path.copy(), new_decisions
            ))
        else:
            nodes_pruned += 1
    
    total_weight = sum(weights[i] for i in best_items)
    return best_items, best_value, total_weight


def branch_and_bound_iterative_deepening(weights: List[int], values: List[int],
                                          capacity: int) -> Tuple[List[int], int, int]:
    """
    Branch-and-Bound avec Iterative Deepening (IDDFS).
    
    Combine les avantages de BFS (optimalité) et DFS (espace mémoire).
    Explore l'arbre niveau par niveau avec profondeur croissante.
    
    Particulièrement utile pour les instances où la mémoire est limitée.
    
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
    
    # Trier par ratio
    indices_with_ratio = [(i, values[i] / weights[i] if weights[i] > 0 else float('inf'))
                          for i in range(n)]
    indices_with_ratio.sort(key=lambda x: x[1], reverse=True)
    sorted_indices = [i for i, _ in indices_with_ratio]
    
    best_value = 0
    best_items = []
    
    def dfs_limited(level: int, current_value: int, current_weight: int,
                    current_items: List[int], max_depth: int) -> None:
        nonlocal best_value, best_items
        
        if level > max_depth or level >= n:
            return
        
        # Élagage par borne
        bound = _compute_martello_toth_bound(
            level - 1, current_value, current_weight, capacity, 
            weights, values, sorted_indices
        )
        if bound <= best_value:
            return
        
        idx = sorted_indices[level]
        
        # Inclure l'objet
        if current_weight + weights[idx] <= capacity:
            new_value = current_value + values[idx]
            new_items = current_items + [idx]
            
            if new_value > best_value:
                best_value = new_value
                best_items = new_items.copy()
            
            dfs_limited(level + 1, new_value, current_weight + weights[idx],
                       new_items, max_depth)
        
        # Exclure l'objet
        dfs_limited(level + 1, current_value, current_weight, 
                   current_items, max_depth)
    
    # Iterative deepening
    for depth in range(n + 1):
        dfs_limited(0, 0, 0, [], depth)
    
    total_weight = sum(weights[i] for i in best_items)
    return best_items, best_value, total_weight


# --- Exemple d'utilisation ---
if __name__ == "__main__":
    weights = [12, 2, 1, 1, 4]
    values = [4, 2, 1, 2, 10]
    capacity = 15
    
    print("=== Branch-and-Bound (BFS) ===")
    items, val, wt = branch_and_bound_bfs(weights, values, capacity)
    print(f"Objets: {items}, Valeur: {val}, Poids: {wt}")
    
    print("\n=== Branch-and-Bound (DFS) ===")
    items, val, wt = branch_and_bound_dfs(weights, values, capacity)
    print(f"Objets: {items}, Valeur: {val}, Poids: {wt}")
