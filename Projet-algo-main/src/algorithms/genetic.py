"""
Algorithme Génétique pour le problème du Sac à Dos 0/1
Approche métaheuristique basée sur l'évolution
"""

import random
from typing import List, Tuple


def genetic_knapsack(weights: List[int], values: List[int], capacity: int,
                     pop_size: int = 80, generations: int = 300,
                     mutation_rate: float = 0.02, 
                     tournament_size: int = 4) -> Tuple[List[int], int, int]:
    """
    Algorithme génétique pour le knapsack 0/1.
    
    Args:
        weights: liste des poids
        values: liste des valeurs
        capacity: capacité du sac
        pop_size: taille de la population
        generations: nombre de générations
        mutation_rate: taux de mutation
        tournament_size: taille du tournoi de sélection
        
    Returns:
        Tuple (indices_sélectionnés, valeur_totale, poids_total)
    """
    n = len(weights)
    
    if n == 0:
        return [], 0, 0
    
    def fitness(individual: List[int]) -> int:
        """Calcule la fitness d'un individu"""
        total_w = sum(weights[i] for i in range(n) if individual[i] == 1)
        if total_w > capacity:
            return 0
        return sum(values[i] for i in range(n) if individual[i] == 1)
    
    def random_individual() -> List[int]:
        """Crée un individu aléatoire"""
        return [random.randint(0, 1) for _ in range(n)]
    
    def repair(individual: List[int]) -> List[int]:
        """Répare un individu invalide"""
        while True:
            total_w = sum(weights[i] for i in range(n) if individual[i] == 1)
            if total_w <= capacity:
                break
            ones = [i for i in range(n) if individual[i] == 1]
            if not ones:
                break
            individual[random.choice(ones)] = 0
        return individual
    
    def tournament_selection(pop: List[List[int]]) -> List[int]:
        """Sélection par tournoi"""
        best = None
        for _ in range(tournament_size):
            ind = random.choice(pop)
            if best is None or fitness(ind) > fitness(best):
                best = ind
        return best
    
    def crossover(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        """Croisement en un point"""
        if n <= 1:
            return p1[:], p2[:]
        point = random.randint(1, n - 1)
        c1 = p1[:point] + p2[point:]
        c2 = p2[:point] + p1[point:]
        return c1, c2
    
    def mutate(individual: List[int]) -> None:
        """Mutation bit-flip"""
        for i in range(n):
            if random.random() < mutation_rate:
                individual[i] = 1 - individual[i]
    
    # Initialisation
    population = [repair(random_individual()) for _ in range(pop_size)]
    best_global = None
    
    # Évolution
    for _ in range(generations):
        new_population = []
        
        # Élitisme
        best = max(population, key=fitness)
        if best_global is None or fitness(best) > fitness(best_global):
            best_global = best[:]
        new_population.append(best_global[:])
        
        # Reproduction
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            child1, child2 = crossover(parent1, parent2)
            
            mutate(child1)
            mutate(child2)
            
            repair(child1)
            repair(child2)
            
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
        
        population = new_population
    
    # Extraire la solution
    selected = [i for i in range(n) if best_global[i] == 1]
    total_weight = sum(weights[i] for i in selected)
    total_value = sum(values[i] for i in selected)
    
    return selected, total_value, total_weight


# --- Exemple d'utilisation ---
if __name__ == "__main__":
    weights = [12, 2, 1, 1, 4]
    values = [4, 2, 1, 2, 10]
    capacity = 15
    
    print("=== Algorithme Génétique ===")
    items, val, wt = genetic_knapsack(weights, values, capacity)
    print(f"Objets: {items}, Valeur: {val}, Poids: {wt}")
