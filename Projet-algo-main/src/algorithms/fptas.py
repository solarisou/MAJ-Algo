"""
FPTAS (Fully Polynomial-Time Approximation Scheme) pour le Knapsack 0/1
Garantit une solution à (1-ε) de l'optimale en temps polynomial
"""

import math
from typing import List, Tuple


def fptas_knapsack(weights: List[int], values: List[int], 
                   capacity: int, eps: float = None) -> Tuple[List[int], int, int]:
    """
    FPTAS pour le knapsack 0/1.
    
    Garantit une solution ≥ (1-ε) * OPT en temps O(n³/ε).
    
    Args:
        weights: liste des poids (entiers)
        values: liste des valeurs (entiers)
        capacity: capacité du sac (entier)
        eps: tolérance d'approximation (auto si None)
        
    Returns:
        Tuple (indices_sélectionnés, valeur_totale, poids_total)
    """
    n = len(weights)
    
    # Epsilon adaptatif selon la taille (montre le compromis precision/temps)
    if eps is None:
        if n <= 100:
            eps = 0.05   # Très précis pour petites instances
        elif n <= 300:
            eps = 0.1    # Bon compromis
        else:
            eps = 0.15   # Légèrement plus rapide pour n>300
    
    assert 0 < eps < 1, "eps doit être dans (0, 1)"
    
    if n == 0:
        return [], 0, 0
    
    V_max = max(values)
    
    if V_max == 0:
        return [], 0, 0
    
    # Facteur d'échelle
    K = eps * V_max / n
    if K <= 0:
        K = 1e-12
    
    # Valeurs mises à l'échelle (entiers)
    v_scaled = [int(math.floor(v / K)) for v in values]
    
    # Éviter le cas où tout devient 0
    if all(vs == 0 for vs in v_scaled):
        v_scaled = [1] * n
    
    V_prime = sum(v_scaled)
    
    # DP: dp[val] = poids minimum pour atteindre valeur "val"
    INF = 10**30
    dp = [INF] * (V_prime + 1)
    prev_item = [-1] * (V_prime + 1)
    prev_val = [-1] * (V_prime + 1)
    
    dp[0] = 0
    
    # Remplissage DP
    for i in range(n):
        vi = v_scaled[i]
        wi = weights[i]
        
        for val in range(V_prime, vi - 1, -1):
            if dp[val - vi] + wi < dp[val]:
                dp[val] = dp[val - vi] + wi
                prev_item[val] = i
                prev_val[val] = val - vi
    
    # Trouver la meilleure valeur réalisable
    best_val_scaled = 0
    for val in range(V_prime + 1):
        if dp[val] <= capacity and val > best_val_scaled:
            best_val_scaled = val
    
    # Reconstruction de la solution
    selected = []
    cur = best_val_scaled
    
    while cur > 0 and prev_item[cur] != -1:
        i = prev_item[cur]
        selected.append(i)
        cur = prev_val[cur]
    
    selected.reverse()
    
    total_weight = sum(weights[i] for i in selected)
    total_value = sum(values[i] for i in selected)
    
    return selected, total_value, total_weight


# --- Exemple d'utilisation ---
if __name__ == "__main__":
    weights = [12, 2, 1, 1, 4]
    values = [4, 2, 1, 2, 10]
    capacity = 15
    
    for eps in [0.1, 0.2, 0.5]:
        print(f"\n=== FPTAS (ε = {eps}) ===")
        items, val, wt = fptas_knapsack(weights, values, capacity, eps)
        print(f"Objets: {items}, Valeur: {val}, Poids: {wt}")
