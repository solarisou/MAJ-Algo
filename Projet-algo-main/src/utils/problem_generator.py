"""
Générateur de Problèmes de Sac à Dos (Knapsack Problem Generator)

Ce module permet de générer automatiquement des instances de problèmes
du sac à dos avec différentes distributions et caractéristiques.

Distributions supportées:
- Uniforme: poids et valeurs uniformément distribués
- Corrélée: valeurs corrélées aux poids
- Inversement corrélée: valeurs inversement corrélées aux poids
- Sous-ensemble somme: cas spécial difficile
- Presque fortement corrélée: variante réaliste
"""

import random
import numpy as np
from typing import List, Tuple, Optional, Dict
from enum import Enum


class DistributionType(Enum):
    """Types de distributions pour la génération de problèmes"""
    UNIFORM = "uniform"
    CORRELATED = "correlated"
    INVERSE_CORRELATED = "inverse_correlated"
    SUBSET_SUM = "subset_sum"
    ALMOST_STRONGLY_CORRELATED = "almost_strongly_correlated"
    MULTIPLE_SUBSET_SUM = "multiple_subset_sum"
    PROFIT_CEILING = "profit_ceiling"
    CIRCLE = "circle"


class KnapsackProblemGenerator:
    """
    Générateur de problèmes du Sac à Dos
    
    Permet de créer des instances de test avec différentes caractéristiques
    pour évaluer les performances des algorithmes.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialise le générateur.
        
        Args:
            seed: Graine pour la reproductibilité (optionnel)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.seed = seed
    
    def generate(self, n: int, 
                 distribution: DistributionType = DistributionType.UNIFORM,
                 weight_range: Tuple[int, int] = (1, 100),
                 value_range: Tuple[int, int] = (1, 100),
                 capacity_ratio: float = 0.5,
                 correlation_factor: float = 10.0) -> Dict:
        """
        Génère une instance de problème du sac à dos.
        
        Args:
            n: Nombre d'objets
            distribution: Type de distribution
            weight_range: (min, max) pour les poids
            value_range: (min, max) pour les valeurs
            capacity_ratio: Ratio de la capacité par rapport à la somme des poids
            correlation_factor: Facteur de corrélation (pour distributions corrélées)
            
        Returns:
            Dictionnaire contenant weights, values, capacity et metadata
        """
        
        if distribution == DistributionType.UNIFORM:
            weights, values = self._generate_uniform(n, weight_range, value_range)
        
        elif distribution == DistributionType.CORRELATED:
            weights, values = self._generate_correlated(n, weight_range, correlation_factor)
        
        elif distribution == DistributionType.INVERSE_CORRELATED:
            weights, values = self._generate_inverse_correlated(n, weight_range, correlation_factor)
        
        elif distribution == DistributionType.SUBSET_SUM:
            weights, values = self._generate_subset_sum(n, weight_range)
        
        elif distribution == DistributionType.ALMOST_STRONGLY_CORRELATED:
            weights, values = self._generate_almost_strongly_correlated(n, weight_range, correlation_factor)
        
        elif distribution == DistributionType.MULTIPLE_SUBSET_SUM:
            weights, values = self._generate_multiple_subset_sum(n, weight_range)
        
        elif distribution == DistributionType.PROFIT_CEILING:
            weights, values = self._generate_profit_ceiling(n, weight_range)
        
        elif distribution == DistributionType.CIRCLE:
            weights, values = self._generate_circle(n, weight_range)
        
        else:
            weights, values = self._generate_uniform(n, weight_range, value_range)
        
        # Calculer la capacité
        total_weight = sum(weights)
        capacity = max(1, int(total_weight * capacity_ratio))
        
        return {
            'weights': weights,
            'values': values,
            'capacity': capacity,
            'n': n,
            'distribution': distribution.value,
            'weight_range': weight_range,
            'value_range': value_range,
            'capacity_ratio': capacity_ratio,
            'total_weight': total_weight,
            'total_value': sum(values)
        }
    
    def _generate_uniform(self, n: int, weight_range: Tuple[int, int],
                          value_range: Tuple[int, int]) -> Tuple[List[int], List[int]]:
        """Distribution uniforme: poids et valeurs indépendants"""
        weights = [random.randint(weight_range[0], weight_range[1]) for _ in range(n)]
        values = [random.randint(value_range[0], value_range[1]) for _ in range(n)]
        return weights, values
    
    def _generate_correlated(self, n: int, weight_range: Tuple[int, int],
                             correlation_factor: float) -> Tuple[List[int], List[int]]:
        """
        Distribution corrélée: valeur = poids + bruit
        Les objets lourds ont tendance à avoir plus de valeur
        """
        weights = [random.randint(weight_range[0], weight_range[1]) for _ in range(n)]
        values = [max(1, int(w + random.uniform(-correlation_factor, correlation_factor))) 
                  for w in weights]
        return weights, values
    
    def _generate_inverse_correlated(self, n: int, weight_range: Tuple[int, int],
                                      correlation_factor: float) -> Tuple[List[int], List[int]]:
        """
        Distribution inversement corrélée: objets légers = plus de valeur
        Difficile pour les algorithmes gloutons basés sur le ratio
        """
        weights = [random.randint(weight_range[0], weight_range[1]) for _ in range(n)]
        max_w = max(weights)
        values = [max(1, int(max_w - w + correlation_factor + random.uniform(-5, 5))) 
                  for w in weights]
        return weights, values
    
    def _generate_subset_sum(self, n: int, 
                              weight_range: Tuple[int, int]) -> Tuple[List[int], List[int]]:
        """
        Subset Sum: valeur = poids
        Cas particulier où trouver la solution optimale est NP-difficile
        """
        weights = [random.randint(weight_range[0], weight_range[1]) for _ in range(n)]
        values = weights.copy()  # Valeur = Poids
        return weights, values
    
    def _generate_almost_strongly_correlated(self, n: int, weight_range: Tuple[int, int],
                                              correlation_factor: float) -> Tuple[List[int], List[int]]:
        """
        Presque fortement corrélée: valeur ≈ poids + constante
        Instance réaliste et difficile
        """
        weights = [random.randint(weight_range[0], weight_range[1]) for _ in range(n)]
        constant = weight_range[1] // 10
        values = [max(1, w + constant + random.randint(-3, 3)) for w in weights]
        return weights, values
    
    def _generate_multiple_subset_sum(self, n: int,
                                       weight_range: Tuple[int, int]) -> Tuple[List[int], List[int]]:
        """
        Multiple Subset Sum: objets groupés par valeur
        """
        num_groups = max(2, n // 10)
        group_values = [random.randint(1, 10) for _ in range(num_groups)]
        
        weights = []
        values = []
        
        for i in range(n):
            group = i % num_groups
            w = random.randint(weight_range[0], weight_range[1])
            weights.append(w)
            values.append(w * group_values[group])
        
        return weights, values
    
    def _generate_profit_ceiling(self, n: int,
                                  weight_range: Tuple[int, int]) -> Tuple[List[int], List[int]]:
        """
        Profit Ceiling: valeur = plafond(poids / d)
        """
        d = max(1, (weight_range[1] - weight_range[0]) // 3)
        weights = [random.randint(weight_range[0], weight_range[1]) for _ in range(n)]
        values = [max(1, (w + d - 1) // d) for w in weights]
        return weights, values
    
    def _generate_circle(self, n: int,
                          weight_range: Tuple[int, int]) -> Tuple[List[int], List[int]]:
        """
        Distribution en cercle: points sur un cercle dans l'espace (poids, valeur)
        """
        center = (weight_range[0] + weight_range[1]) // 2
        radius = (weight_range[1] - weight_range[0]) // 2
        
        weights = []
        values = []
        
        for i in range(n):
            angle = 2 * np.pi * i / n
            w = max(1, int(center + radius * np.cos(angle)))
            v = max(1, int(center + radius * np.sin(angle)))
            weights.append(w)
            values.append(v)
        
        return weights, values
    
    def generate_batch(self, configs: List[Dict]) -> List[Dict]:
        """
        Génère plusieurs instances avec différentes configurations.
        
        Args:
            configs: Liste de configurations
            
        Returns:
            Liste d'instances générées
        """
        instances = []
        for config in configs:
            instance = self.generate(**config)
            instances.append(instance)
        return instances
    
    def generate_difficulty_series(self, base_n: int = 10, 
                                    num_instances: int = 5,
                                    growth_factor: float = 2.0) -> List[Dict]:
        """
        Génère une série d'instances de difficulté croissante.
        
        Args:
            base_n: Nombre d'objets initial
            num_instances: Nombre d'instances à générer
            growth_factor: Facteur de croissance de la taille
            
        Returns:
            Liste d'instances de difficulté croissante
        """
        instances = []
        n = base_n
        
        for i in range(num_instances):
            instance = self.generate(
                n=int(n),
                distribution=DistributionType.UNIFORM,
                capacity_ratio=0.5
            )
            instance['difficulty_level'] = i + 1
            instances.append(instance)
            n *= growth_factor
        
        return instances
    
    def save_instance(self, instance: Dict, filepath: str) -> None:
        """
        Sauvegarde une instance dans un fichier texte.
        
        Format:
        n capacity
        w1 v1
        w2 v2
        ...
        """
        with open(filepath, 'w') as f:
            f.write(f"{instance['n']} {instance['capacity']}\n")
            for w, v in zip(instance['weights'], instance['values']):
                f.write(f"{w} {v}\n")
            # Métadonnées en commentaires
            f.write(f"# Distribution: {instance['distribution']}\n")
            f.write(f"# Total weight: {instance['total_weight']}\n")
            f.write(f"# Total value: {instance['total_value']}\n")
    
    @staticmethod
    def load_instance(filepath: str) -> Dict:
        """
        Charge une instance depuis un fichier.
        """
        weights = []
        values = []
        
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            parts = first_line.split()
            n, capacity = int(parts[0]), int(parts[1])
            
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    weights.append(int(parts[0]))
                    values.append(int(parts[1]))
        
        return {
            'weights': weights,
            'values': values,
            'capacity': capacity,
            'n': len(weights)
        }


def generate_test_suite(output_dir: str = "data/generated", seed: int = 42) -> List[str]:
    """
    Génère une suite complète de tests pour l'évaluation.
    
    Args:
        output_dir: Répertoire de sortie
        seed: Graine pour reproductibilité
        
    Returns:
        Liste des chemins des fichiers générés
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    generator = KnapsackProblemGenerator(seed=seed)
    generated_files = []
    
    # Configurations de test
    test_configs = [
        # Petites instances (pour brute-force)
        {'n': 10, 'distribution': DistributionType.UNIFORM, 'name': 'small_uniform'},
        {'n': 15, 'distribution': DistributionType.CORRELATED, 'name': 'small_correlated'},
        {'n': 12, 'distribution': DistributionType.SUBSET_SUM, 'name': 'small_subset'},
        
        # Moyennes instances
        {'n': 50, 'distribution': DistributionType.UNIFORM, 'name': 'medium_uniform'},
        {'n': 50, 'distribution': DistributionType.CORRELATED, 'name': 'medium_correlated'},
        {'n': 50, 'distribution': DistributionType.INVERSE_CORRELATED, 'name': 'medium_inverse'},
        {'n': 100, 'distribution': DistributionType.ALMOST_STRONGLY_CORRELATED, 'name': 'medium_asc'},
        
        # Grandes instances
        {'n': 200, 'distribution': DistributionType.UNIFORM, 'name': 'large_uniform'},
        {'n': 500, 'distribution': DistributionType.CORRELATED, 'name': 'large_correlated'},
        {'n': 1000, 'distribution': DistributionType.UNIFORM, 'name': 'xlarge_uniform'},
        
        # Cas difficiles
        {'n': 100, 'distribution': DistributionType.SUBSET_SUM, 'name': 'hard_subset'},
        {'n': 100, 'distribution': DistributionType.CIRCLE, 'name': 'hard_circle'},
    ]
    
    for config in test_configs:
        name = config.pop('name')
        instance = generator.generate(**config)
        
        filepath = os.path.join(output_dir, f"{name}.txt")
        generator.save_instance(instance, filepath)
        generated_files.append(filepath)
        print(f"[OK] Genere: {filepath} (n={instance['n']}, distribution={instance['distribution']})")
    
    return generated_files


# --- Exemple d'utilisation ---
if __name__ == "__main__":
    print("=== Générateur de Problèmes du Sac à Dos ===\n")
    
    # Créer un générateur avec seed fixe
    gen = KnapsackProblemGenerator(seed=42)
    
    # Exemple 1: Instance uniforme
    print("1. Distribution Uniforme:")
    instance = gen.generate(n=20, distribution=DistributionType.UNIFORM)
    print(f"   n={instance['n']}, capacity={instance['capacity']}")
    print(f"   Poids: {instance['weights'][:5]}...")
    print(f"   Valeurs: {instance['values'][:5]}...")
    
    # Exemple 2: Instance corrélée
    print("\n2. Distribution Corrélée:")
    instance = gen.generate(n=20, distribution=DistributionType.CORRELATED)
    print(f"   n={instance['n']}, capacity={instance['capacity']}")
    print(f"   Ratios v/w: {[round(v/w, 2) for v, w in zip(instance['values'][:5], instance['weights'][:5])]}")
    
    # Exemple 3: Série de difficulté croissante
    print("\n3. Série de difficulté croissante:")
    series = gen.generate_difficulty_series(base_n=10, num_instances=4)
    for inst in series:
        print(f"   Niveau {inst['difficulty_level']}: n={inst['n']}")
    
    # Générer suite de tests
    print("\n4. Génération de la suite de tests complète:")
    generate_test_suite()
