"""
Algorithme Génétique pour le problème du Sac à Dos 0/1
Approche métaheuristique basée sur l'évolution
"""

import random
from typing import List, Tuple


def genetic_knapsack(weights: List[int], values: List[int], capacity: int,
                     pop_size: int = None, generations: int = None,
                     mutation_rate: float = 0.05, 
                     tournament_size: int = 5,
                     elitism: int = None) -> Tuple[List[int], int, int]:
    """
    Algorithme genetique pour le knapsack 0/1.
    Parametres adaptes automatiquement a la taille du probleme.
    
    Args:
        weights: liste des poids
        values: liste des valeurs
        capacity: capacite du sac
        pop_size: taille de la population (auto si None)
        generations: nombre de generations (auto si None)
        mutation_rate: taux de mutation
        tournament_size: taille du tournoi de selection
        elitism: nombre d'elites conservees (auto si None)
        
    Returns:
        Tuple (indices_selectionnes, valeur_totale, poids_total)
    """
    n = len(weights)
    
    # Parametres adaptatifs - equilibre temps/qualite
    if pop_size is None:
        pop_size = min(80, max(30, n))
    if generations is None:
        generations = min(100, max(50, n))
    if elitism is None:
        elitism = max(2, pop_size // 10)
    
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
    
    # Initialisation avec solution greedy pour demarrer
    population = []
    
    # Ajouter une solution greedy comme point de depart
    greedy_sol = [0] * n
    items_sorted = sorted(range(n), key=lambda i: values[i]/weights[i] if weights[i] > 0 else 0, reverse=True)
    remaining = capacity
    for i in items_sorted:
        if weights[i] <= remaining:
            greedy_sol[i] = 1
            remaining -= weights[i]
    population.append(greedy_sol)
    
    # Remplir avec des individus aleatoires
    while len(population) < pop_size:
        population.append(repair(random_individual()))
    
    best_global = max(population, key=fitness)
    
    # Evolution
    for gen in range(generations):
        new_population = []
        
        # Elitisme: garder les meilleurs
        sorted_pop = sorted(population, key=fitness, reverse=True)
        for i in range(elitism):
            new_population.append(sorted_pop[i][:])
        
        # Mettre a jour le meilleur global
        if fitness(sorted_pop[0]) > fitness(best_global):
            best_global = sorted_pop[0][:]
        
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
