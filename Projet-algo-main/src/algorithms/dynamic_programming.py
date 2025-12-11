"""
Algorithmes de Programmation Dynamique pour le problème du Sac à Dos 0/1
Implémentations Bottom-Up (itératif) et Top-Down (mémoïsation)
"""

from typing import List, Tuple


def knapsack_bottom_up(weights: List[int], values: List[int], 
                       capacity: int) -> Tuple[int, List[int]]:
    """
    Algorithme de programmation dynamique Bottom-Up (itératif).
    
    Complexité temporelle: O(n * W)
    Complexité spatiale: O(n * W)
    
    Args:
        weights: liste des poids des objets
        values: liste des valeurs des objets
        capacity: capacité maximale du sac
        
    Returns:
        Tuple (valeur_maximale, liste_indices_sélectionnés)
    """
    n = len(weights)
    
    if n == 0 or capacity == 0:
        return 0, []
    
    # Création de la table DP
    # dp[i][w] = valeur maximale avec les i premiers objets et capacité w
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    # Remplissage de la table (bottom-up)
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Option 1: ne pas prendre l'objet i-1
            dp[i][w] = dp[i-1][w]
            
            # Option 2: prendre l'objet i-1 si possible
            if weights[i-1] <= w:
                value_with = dp[i-1][w - weights[i-1]] + values[i-1]
                dp[i][w] = max(dp[i][w], value_with)
    
    # Valeur maximale
    max_value = dp[n][capacity]
    
    # Reconstruction de la solution
    selected_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected_items.append(i-1)
            w -= weights[i-1]
    
    selected_items.reverse()
    
    return max_value, selected_items


def knapsack_bottom_up_optimized(weights: List[int], values: List[int], 
                                  capacity: int) -> Tuple[int, List[int]]:
    """
    Version optimisée en espace du DP Bottom-Up.
    
    Utilise seulement O(W) espace au lieu de O(n * W).
    
    Args:
        weights: liste des poids des objets
        values: liste des valeurs des objets
        capacity: capacité maximale du sac
        
    Returns:
        Tuple (valeur_maximale, liste_indices_sélectionnés)
    """
    n = len(weights)
    
    if n == 0 or capacity == 0:
        return 0, []
    
    # Une seule ligne pour le DP
    dp = [0] * (capacity + 1)
    # Garder trace des décisions pour reconstruction
    keep = [[False] * (capacity + 1) for _ in range(n)]
    
    for i in range(n):
        # Parcourir de droite à gauche pour éviter d'utiliser un objet plusieurs fois
        for w in range(capacity, weights[i] - 1, -1):
            if dp[w - weights[i]] + values[i] > dp[w]:
                dp[w] = dp[w - weights[i]] + values[i]
                keep[i][w] = True
    
    # Reconstruction
    selected_items = []
    w = capacity
    for i in range(n - 1, -1, -1):
        if keep[i][w]:
            selected_items.append(i)
            w -= weights[i]
    
    selected_items.reverse()
    
    return dp[capacity], selected_items


def knapsack_top_down(weights: List[int], values: List[int], 
                      capacity: int) -> Tuple[int, List[int]]:
    """
    Algorithme de programmation dynamique Top-Down (récursif avec mémoïsation).
    
    Complexité temporelle: O(n * W)
    Complexité spatiale: O(n * W)
    
    Args:
        weights: liste des poids des objets
        values: liste des valeurs des objets
        capacity: capacité maximale du sac
        
    Returns:
        Tuple (valeur_maximale, liste_indices_sélectionnés)
    """
    n = len(weights)
    
    if n == 0 or capacity == 0:
        return 0, []
    
    # Table de mémoïsation
    memo = [[-1 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    def solve(i: int, w: int) -> int:
        """Fonction récursive avec mémoïsation"""
        if i == n or w == 0:
            return 0
        
        if memo[i][w] != -1:
            return memo[i][w]
        
        # Option 1: ne pas prendre l'objet
        value_without = solve(i + 1, w)
        
        # Option 2: prendre l'objet si possible
        value_with = 0
        if weights[i] <= w:
            value_with = values[i] + solve(i + 1, w - weights[i])
        
        memo[i][w] = max(value_without, value_with)
        return memo[i][w]
    
    max_value = solve(0, capacity)
    
    # Reconstruction de la solution
    selected_items = []
    i, w = 0, capacity
    
    while i < n and w > 0:
        value_without = memo[i+1][w] if i+1 <= n else 0
        value_with = 0
        
        if weights[i] <= w:
            value_with = values[i] + (memo[i+1][w - weights[i]] if i+1 <= n else 0)
        
        if weights[i] <= w and value_with > value_without:
            selected_items.append(i)
            w -= weights[i]
        
        i += 1
    
    return max_value, selected_items


# --- Exemple d'utilisation ---
if __name__ == "__main__":
    # Test
    weights = [12, 2, 1, 1, 4]
    values = [4, 2, 1, 2, 10]
    capacity = 15
    
    print("=== Programmation Dynamique Bottom-Up ===")
    val, items = knapsack_bottom_up(weights, values, capacity)
    print(f"Valeur maximale: {val}")
    print(f"Objets sélectionnés: {items}")
    
    print("\n=== Programmation Dynamique Bottom-Up (Optimisé) ===")
    val, items = knapsack_bottom_up_optimized(weights, values, capacity)
    print(f"Valeur maximale: {val}")
    print(f"Objets sélectionnés: {items}")
    
    print("\n=== Programmation Dynamique Top-Down ===")
    val, items = knapsack_top_down(weights, values, capacity)
    print(f"Valeur maximale: {val}")
    print(f"Objets sélectionnés: {items}")
