"""
Module pour télécharger et charger les instances Pisinger depuis GitHub
Repository: https://github.com/dnlfm/knapsack-01-instances

Format Pisinger:
- Ligne 1: n capacity (nombre d'objets et capacité)
- Lignes 2 à n+1: value weight (valeur et poids de chaque objet)  
- Dernière ligne: solution optimale binaire (0/1 pour chaque objet)
"""

import os
import requests
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class KnapsackInstance:
    """Représente une instance du problème Knapsack"""
    name: str
    n: int
    capacity: int
    values: List[int]
    weights: List[int]
    optimal_value: Optional[int] = None
    optimal_solution: Optional[List[int]] = None
    
    def __str__(self):
        return f"KnapsackInstance(name={self.name}, n={self.n}, capacity={self.capacity})"


# URLs de base pour les instances Pisinger
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/dnlfm/knapsack-01-instances/main/pisinger_instances_01_KP/large_scale"

# Liste des instances disponibles par catégorie
PISINGER_INSTANCES = {
    'small': [
        'knapPI_1_100_1000_1',
        'knapPI_2_100_1000_1',
        'knapPI_3_100_1000_1',
    ],
    'medium': [
        'knapPI_1_200_1000_1',
        'knapPI_2_200_1000_1',
        'knapPI_3_200_1000_1',
        'knapPI_1_500_1000_1',
        'knapPI_2_500_1000_1',
        'knapPI_3_500_1000_1',
    ],
    'large': [
        'knapPI_1_1000_1000_1',
        'knapPI_2_1000_1000_1',
        'knapPI_3_1000_1000_1',
        'knapPI_1_2000_1000_1',
        'knapPI_2_2000_1000_1',
        'knapPI_3_2000_1000_1',
    ],
    'very_large': [
        'knapPI_1_5000_1000_1',
        'knapPI_2_5000_1000_1',
        'knapPI_3_5000_1000_1',
        'knapPI_1_10000_1000_1',
        'knapPI_2_10000_1000_1',
        'knapPI_3_10000_1000_1',
    ]
}


class PisingerInstanceLoader:
    """Classe pour charger et gérer les instances Pisinger"""
    
    def __init__(self, data_dir: str = "data/pisinger_instances"):
        """
        Initialise le loader.
        
        Args:
            data_dir: Répertoire où stocker les instances téléchargées
        """
        self.data_dir = data_dir
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Crée les répertoires nécessaires"""
        for category in ['small', 'medium', 'large', 'very_large']:
            path = os.path.join(self.data_dir, category)
            os.makedirs(path, exist_ok=True)
    
    def download_instance(self, instance_name: str, category: str = None) -> bool:
        """
        Télécharge une instance depuis GitHub.
        
        Args:
            instance_name: Nom de l'instance (ex: 'knapPI_1_100_1000_1')
            category: Catégorie de l'instance (small, medium, large, very_large)
            
        Returns:
            True si succès, False sinon
        """
        # Déterminer la catégorie si non spécifiée
        if category is None:
            category = self._guess_category(instance_name)
        
        url = f"{GITHUB_RAW_BASE}/{instance_name}"
        filepath = os.path.join(self.data_dir, category, f"{instance_name}.txt")
        
        # Ne pas retélécharger si le fichier existe déjà
        if os.path.exists(filepath):
            print(f"[OK] Instance déjà présente: {instance_name}")
            return True
        
        try:
            print(f"Telechargement: {instance_name}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            print(f"[OK] Telecharge: {instance_name}")
            return True
            
        except requests.RequestException as e:
            print(f"[FAIL] Erreur telechargement {instance_name}: {e}")
            return False
    
    def _guess_category(self, instance_name: str) -> str:
        """Devine la catégorie d'une instance à partir de son nom"""
        for category, instances in PISINGER_INSTANCES.items():
            if instance_name in instances:
                return category
        
        # Deviner à partir du nom (knapPI_X_SIZE_...)
        parts = instance_name.split('_')
        if len(parts) >= 3:
            try:
                size = int(parts[2])
                if size <= 100:
                    return 'small'
                elif size <= 500:
                    return 'medium'
                elif size <= 2000:
                    return 'large'
                else:
                    return 'very_large'
            except ValueError:
                pass
        return 'medium'
    
    def download_all_instances(self, categories: List[str] = None) -> Dict[str, bool]:
        """
        Télécharge toutes les instances des catégories spécifiées.
        
        Args:
            categories: Liste des catégories à télécharger. Si None, télécharge tout.
            
        Returns:
            Dictionnaire {instance_name: success}
        """
        if categories is None:
            categories = list(PISINGER_INSTANCES.keys())
        
        results = {}
        for category in categories:
            if category not in PISINGER_INSTANCES:
                print(f"[WARN] Catégorie inconnue: {category}")
                continue
            
            print(f"\nTelechargement categorie: {category}")
            print("-" * 40)
            
            for instance_name in PISINGER_INSTANCES[category]:
                results[instance_name] = self.download_instance(instance_name, category)
        
        return results
    
    def load_instance(self, filepath: str) -> KnapsackInstance:
        """
        Charge une instance depuis un fichier local.
        
        Args:
            filepath: Chemin vers le fichier d'instance
            
        Returns:
            KnapsackInstance contenant les données
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # Première ligne: n et capacité
        first_line = lines[0].split()
        n = int(first_line[0])
        capacity = int(first_line[1])
        
        values = []
        weights = []
        
        # Lignes des objets (format: valeur poids)
        for i in range(1, n + 1):
            if i < len(lines):
                parts = lines[i].split()
                if len(parts) >= 2:
                    values.append(int(parts[0]))
                    weights.append(int(parts[1]))
        
        # Dernière ligne: solution optimale (si présente)
        optimal_solution = None
        optimal_value = None
        
        if len(lines) > n + 1:
            # La solution peut être sur plusieurs lignes, concaténer
            solution_lines = ' '.join(lines[n + 1:])
            solution_parts = solution_lines.split()
            try:
                optimal_solution = [int(x) for x in solution_parts]
                # Calculer la valeur optimale à partir de la solution
                if len(optimal_solution) == n:
                    optimal_value = sum(v * s for v, s in zip(values, optimal_solution))
            except ValueError:
                pass
        
        name = os.path.basename(filepath).replace('.txt', '')
        
        return KnapsackInstance(
            name=name,
            n=n,
            capacity=capacity,
            values=values,
            weights=weights,
            optimal_value=optimal_value,
            optimal_solution=optimal_solution
        )
    
    def load_instance_by_name(self, instance_name: str, 
                              download_if_missing: bool = True) -> Optional[KnapsackInstance]:
        """
        Charge une instance par son nom.
        
        Args:
            instance_name: Nom de l'instance
            download_if_missing: Télécharger si l'instance n'existe pas localement
            
        Returns:
            KnapsackInstance ou None si échec
        """
        category = self._guess_category(instance_name)
        filepath = os.path.join(self.data_dir, category, f"{instance_name}.txt")
        
        if not os.path.exists(filepath):
            if download_if_missing:
                if not self.download_instance(instance_name, category):
                    return None
            else:
                print(f"[FAIL] Instance non trouvee: {filepath}")
                return None
        
        return self.load_instance(filepath)
    
    def list_local_instances(self) -> Dict[str, List[str]]:
        """
        Liste toutes les instances disponibles localement.
        
        Returns:
            Dictionnaire {category: [instance_names]}
        """
        instances = {}
        for category in ['small', 'medium', 'large', 'very_large']:
            path = os.path.join(self.data_dir, category)
            if os.path.exists(path):
                files = [f.replace('.txt', '') for f in os.listdir(path) if f.endswith('.txt')]
                if files:
                    instances[category] = sorted(files)
        return instances
    
    def get_all_instances(self, categories: List[str] = None,
                          download_if_missing: bool = True) -> List[KnapsackInstance]:
        """
        Récupère toutes les instances des catégories spécifiées.
        
        Args:
            categories: Liste des catégories
            download_if_missing: Télécharger les instances manquantes
            
        Returns:
            Liste de KnapsackInstance
        """
        if categories is None:
            categories = ['small', 'medium', 'large']
        
        instances = []
        for category in categories:
            if category not in PISINGER_INSTANCES:
                continue
            
            for instance_name in PISINGER_INSTANCES[category]:
                instance = self.load_instance_by_name(
                    instance_name, 
                    download_if_missing=download_if_missing
                )
                if instance:
                    instances.append(instance)
        
        return instances


def load_instance(filepath: str) -> Tuple[int, int, List[int], List[int], Optional[int]]:
    """
    Fonction utilitaire pour charger une instance (compatibilité avec l'ancien code).
    
    Args:
        filepath: Chemin vers le fichier d'instance
        
    Returns:
        Tuple (n, capacity, values, weights, optimal_value)
    """
    loader = PisingerInstanceLoader()
    instance = loader.load_instance(filepath)
    return (instance.n, instance.capacity, instance.values, 
            instance.weights, instance.optimal_value)


def download_pisinger_instances(categories: List[str] = None, 
                                data_dir: str = "data/pisinger_instances") -> bool:
    """
    Télécharge les instances Pisinger depuis GitHub.
    
    Args:
        categories: Liste des catégories à télécharger ('small', 'medium', 'large', 'very_large')
        data_dir: Répertoire de destination
        
    Returns:
        True si toutes les instances ont été téléchargées avec succès
    """
    loader = PisingerInstanceLoader(data_dir)
    results = loader.download_all_instances(categories)
    return all(results.values())


# --- Script principal ---
if __name__ == "__main__":
    print("="*60)
    print("TÉLÉCHARGEMENT DES INSTANCES PISINGER")
    print("Source: https://github.com/dnlfm/knapsack-01-instances")
    print("="*60)
    
    # Télécharger toutes les instances (sauf very_large qui sont très grandes)
    loader = PisingerInstanceLoader()
    loader.download_all_instances(['small', 'medium', 'large'])
    
    print("\n" + "="*60)
    print("INSTANCES DISPONIBLES")
    print("="*60)
    
    for category, instances in loader.list_local_instances().items():
        print(f"\n{category.upper()}:")
        for name in instances:
            inst = loader.load_instance_by_name(name, download_if_missing=False)
            if inst:
                opt_str = f", optimal={inst.optimal_value}" if inst.optimal_value else ""
                print(f"   - {name}: n={inst.n}, C={inst.capacity}{opt_str}")
