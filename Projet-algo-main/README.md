# Projet Algorithmes - Problème du Sac à Dos (0/1 Knapsack)

## Description

Ce projet implémente et compare différents algorithmes pour résoudre le **problème du sac à dos 0/1** (0/1 Knapsack Problem). Les tests sont effectués sur les **instances de benchmark Pisinger**, reconnues dans la littérature scientifique.

**Master DSC/MLDM - Cours d'Algorithmes Avancés 2025-2026**

---

## Structure du Projet

```
Projet-algo/
├── main.py                    # Point d'entrée principal (menu interactif)
├── download_instances.py      # Script de téléchargement des instances
├── README.md                  # Ce fichier
├── PLANNING.md               # Planning prévisionnel et réel
├── CHECKLIST.md              # Checklist de complétion du projet
├── requirements.txt          # Dépendances Python
│
├── src/                       # Code source
│   ├── __init__.py
│   ├── benchmark.py           # Système de benchmark
│   ├── visualizations.py      # Génération des graphiques d'analyse
│   │
│   ├── algorithms/            # Implémentations des algorithmes
│   │   ├── __init__.py
│   │   ├── bruteforce.py           # Force brute O(2^n)
│   │   ├── dynamic_programming.py  # DP Bottom-Up et Top-Down O(nW)
│   │   ├── greedy.py               # Algorithmes gloutons O(n log n)
│   │   ├── branch_and_bound.py     # Branch & Bound (BFS/DFS/LC/IDDFS)
│   │   ├── fractional_approximation.py  # Approximation fractionnaire
│   │   ├── fptas.py                # FPTAS O(n³/ε)
│   │   ├── genetic.py              # Algorithme génétique
│   │   ├── ant_colony.py           # Colonie de fourmis
│   │   └── randomized.py           # Algorithme randomisé
│   │
│   └── utils/                 # Utilitaires
│       ├── __init__.py
│       ├── instance_loader.py      # Chargement des instances Pisinger
│       ├── solution_validator.py   # Validation des solutions
│       └── problem_generator.py    # Générateur de problèmes aléatoires
│
├── data/                      # Données
│   ├── pisinger_instances/    # Instances de benchmark Pisinger
│   │   ├── small/             # n = 100 objets
│   │   ├── medium/            # n = 200-500 objets
│   │   ├── large/             # n = 1000-2000 objets
│   │   └── very_large/        # n = 5000-10000 objets
│   └── generated/             # Instances générées automatiquement
│
├── results/                   # Résultats des benchmarks
│   ├── benchmark_results.csv  # Résultats bruts
│   └── graphs/                # Visualisations générées
│       ├── 01_execution_time_comparison.png
│       ├── 02_gap_to_optimal.png
│       └── ... (11 graphiques)
│
└── tests/                     # Tests unitaires
```

---

## Installation

### Prérequis

- **Python 3.8+** (testé avec Python 3.14)
- **pip** (gestionnaire de paquets Python)
- **Git** (optionnel, pour cloner le repository)

### Étape 1: Cloner ou télécharger le projet

```bash
git clone <repository_url>
cd Projet-algo
```

### Étape 2: Créer un environnement virtuel (recommandé)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Étape 3: Installer les dépendances

```bash
pip install -r requirements.txt
```

Ou manuellement :

```bash
pip install requests numpy matplotlib pandas seaborn PyMuPDF
```

### Étape 4: Vérifier l'installation

```bash
python main.py
```

---

## Téléchargement des Instances de Test

Les instances proviennent du repository GitHub :
**https://github.com/dnlfm/knapsack-01-instances**

Ces instances sont les fameuses **instances Pisinger**, largement utilisées pour le benchmarking des algorithmes de knapsack.

### Via le menu interactif

```bash
python main.py
# Choisir option 1: Télécharger les instances
```

### Via le script dédié

```bash
python download_instances.py
```

### Via Python

```python
from src.utils.instance_loader import download_pisinger_instances
download_pisinger_instances(categories=['small', 'medium', 'large'])
```

### Instances Disponibles

| Catégorie   | Nombre d'objets | Instances par type | Fichiers exemples |
|-------------|-----------------|--------------------|--------------------|
| **small**   | 100             | 10 par série       | knapPI_1_100_1000_1 |
| **medium**  | 200-500         | 10 par série       | knapPI_2_500_1000_1 |
| **large**   | 1000-2000       | 10 par série       | knapPI_1_2000_1000_1 |
| **very_large** | 5000-10000   | 10 par série       | knapPI_3_10000_1000_1 |

**Format des noms**: `knapPI_{type}_{n}_{R}_{instance}`
- `type`: 1 (uncorrelated), 2 (weakly correlated), 3 (strongly correlated)
- `n`: nombre d'objets
- `R`: range des poids/valeurs
- `instance`: numéro d'instance (1-10)

---

## Utilisation

### Interface Interactive

```bash
python main.py
```

Menu disponible :
1. Télécharger les instances Pisinger
2. Exécuter le benchmark complet
3. Tester un algorithme spécifique
4. Valider une solution
5. Générer les visualisations
6. Générer des problèmes aléatoires
7. Résoudre un problème réel (exemple)
8. Quitter

### Exécuter le Benchmark

```bash
# Via le menu (option 2)
python main.py

# Ou directement
python -c "from src.benchmark import run_full_benchmark; run_full_benchmark()"
```

### Générer des Problèmes Aléatoires

```python
from src.utils.problem_generator import KnapsackProblemGenerator, DistributionType

# Créer un générateur
gen = KnapsackProblemGenerator(seed=42)

# Générer une instance
instance = gen.generate(
    n=100,
    distribution=DistributionType.CORRELATED,
    capacity_ratio=0.5
)

print(f"Capacité: {instance['capacity']}")
print(f"Poids: {instance['weights'][:10]}...")
```

### Utiliser un Algorithme Spécifique

```python
from src.algorithms.dynamic_programming import knapsack_bottom_up
from src.utils.instance_loader import PisingerInstanceLoader

# Charger une instance
loader = PisingerInstanceLoader()
instance = loader.load_instance_by_name('knapPI_1_100_1000_1')

# Résoudre
value, items = knapsack_bottom_up(
    instance.weights, 
    instance.values, 
    instance.capacity
)

print(f"Valeur optimale: {value}")
print(f"Objets sélectionnés: {items}")
```

### Générer les Visualisations

```bash
python main.py
# Choisir option 5
```

Les graphiques seront sauvegardés dans `results/graphs/`.

---

## Algorithmes Implémentés (15 au total)

### Algorithmes Exacts (5)

| Algorithme | Complexité | Fichier | Description |
|------------|------------|---------|-------------|
| **Brute-Force** | O(2ⁿ) | `bruteforce.py` | Exploration exhaustive avec élagage |
| **DP Bottom-Up** | O(nW) | `dynamic_programming.py` | Programmation dynamique itérative |
| **DP Top-Down** | O(nW) | `dynamic_programming.py` | DP avec mémoïsation |
| **B&B BFS** | Variable | `branch_and_bound.py` | Best-First Search |
| **B&B DFS** | Variable | `branch_and_bound.py` | Depth-First Search |

### Algorithmes Branch-and-Bound Avancés (2)

| Algorithme | Stratégie | Description |
|------------|-----------|-------------|
| **B&B Least-Cost** | Borne Martello-Toth | Borne supérieure plus serrée |
| **B&B IDDFS** | Iterative Deepening | Mémoire limitée, optimalité |

### Algorithmes d'Approximation (4)

| Algorithme | Garantie | Description |
|------------|----------|-------------|
| **Greedy Ratio** | 1/2-OPT | Sélection par ratio valeur/poids |
| **Greedy Value** | Variable | Sélection par valeur décroissante |
| **Greedy Weight** | Variable | Sélection par poids croissant |
| **Fractional** | 1/2-OPT | Basé sur relaxation fractionnaire |

### Schéma d'Approximation (1)

| Algorithme | Garantie | Complexité |
|------------|----------|------------|
| **FPTAS** | (1-ε)-OPT | O(n³/ε) |

### Métaheuristiques (3)

| Algorithme | Type | Paramètres Clés |
|------------|------|-----------------|
| **Génétique** | Évolutionnaire | Population, mutation, croisement |
| **Colonie de Fourmis** | Swarm Intelligence | Phéromones, évaporation |
| **Randomisé** | Probabiliste | Nombre d'itérations |

---

## Résultats et Analyse

### Fichiers de Résultats

- `results/benchmark_results.csv` : Données brutes (algorithme, instance, valeur, temps, gap)
- `results/graphs/` : 11 visualisations d'analyse

### Visualisations Générées

1. **Comparaison des temps d'exécution** (bar chart)
2. **Gap à l'optimal** par algorithme
3. **Analyse de scalabilité** (temps vs taille)
4. **Trade-off qualité/temps**
5. **Heatmap de performance**
6. **Heatmap des temps**
7. **Distribution des gaps** (boxplot)
8. **Distribution des temps** (boxplot)
9. **Ranking des algorithmes**
10. **Comparaison par catégorie**
11. **Résumé des solutions**

---

## Tests

### Exécuter les tests

```bash
# Test d'un algorithme spécifique
python -c "
from src.algorithms.dynamic_programming import knapsack_bottom_up
weights = [10, 20, 30]
values = [60, 100, 120]
capacity = 50
val, items = knapsack_bottom_up(weights, values, capacity)
print(f'Valeur: {val}, Items: {items}')
assert val == 220, 'Test failed!'
print('Test passed!')
"
```

### Valider une solution

```bash
python main.py
# Option 4: Valider une solution
```

---

## Configuration

### Paramètres du Benchmark

Dans `src/benchmark.py` :

```python
TIMEOUT = 60  # Secondes max par algorithme
SEED = 42     # Graine pour reproductibilité
```

### Paramètres des Métaheuristiques

**Algorithme Génétique** (`genetic.py`):
- `population_size`: 100
- `generations`: 200
- `mutation_rate`: 0.1
- `crossover_rate`: 0.8

**Colonie de Fourmis** (`ant_colony.py`):
- `num_ants`: 20
- `num_iterations`: 100
- `evaporation_rate`: 0.5
- `alpha`: 1.0 (importance phéromones)
- `beta`: 2.0 (importance heuristique)

---

## Références

### Instances de Benchmark
1. **Pisinger, D.** - "Algorithms for Knapsack Problems" (Ph.D. thesis, 1995)
2. **Repository**: https://github.com/dnlfm/knapsack-01-instances

### Algorithmes
3. **Kellerer, Pferschy, Pisinger** - "Knapsack Problems" (Springer, 2004)
4. **Martello, Toth** - "Knapsack Problems: Algorithms and Computer Implementations" (1990)
5. **Dorigo, Stützle** - "Ant Colony Optimization" (MIT Press, 2004)

### FPTAS
6. **Ibarra, Kim** - "Fast Approximation Algorithms for the Knapsack and Sum of Subset Problems" (1975)

---

## Auteurs

Projet réalisé dans le cadre du cours d'Algorithmique Avancée.
**Master DSC/MLDM - Université Jean Monnet, Saint-Étienne**

### Répartition du Travail

Voir [PLANNING.md](PLANNING.md) pour la répartition détaillée par membre.

---

## Documents Inclus

| Document | Description |
|----------|-------------|
| [README.md](README.md) | Ce fichier - guide d'utilisation |
| [PLANNING.md](PLANNING.md) | Planning prévisionnel et réel |
| [CHECKLIST.md](CHECKLIST.md) | Checklist de complétion (obligatoire) |
| `rapport.pdf` | Rapport final (à ajouter) |
| `slides.pdf` | Présentation soutenance (à ajouter) |

---

## Notes Importantes

1. **Reproductibilité**: Utiliser `seed=42` pour des résultats identiques
2. **Timeout**: Brute-force limité aux instances < 25 objets
3. **Mémoire**: DP peut nécessiter beaucoup de RAM pour grandes capacités
4. **Python Version**: Testé avec Python 3.8+, recommandé 3.10+

---

## Licence

Ce projet est fourni à des fins éducatives dans le cadre du Master DSC/MLDM.
