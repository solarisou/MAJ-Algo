# Checklist de Complétion du Projet

## Project Completion Checklist

Cette checklist doit être complétée et incluse dans le rapport. Marquez chaque élément et fournissez des commentaires ou observations pertinents.

| V/X | Exigence | Commentaires/Observations |
|-----|----------|---------------------------|
| [OK] | **Avez-vous relu votre rapport ?** | Rapport relu et corrigé par tous les membres du groupe. Vérification orthographique et grammaticale effectuée. |
| [OK] | **Avez-vous présenté l'objectif global de votre travail ?** | L'objectif est clairement défini dans l'introduction : implémenter et comparer différents algorithmes pour le problème du sac à dos 0/1. |
| [OK] | **Avez-vous présenté les principes de toutes les méthodes/algorithmes utilisés ?** | Chaque algorithme est décrit avec sa complexité, ses avantages et inconvénients. Les algorithmes non vus en cours (Ant Colony, FPTAS, B&B Least-Cost) sont détaillés avec références. |
| [OK] | **Avez-vous cité correctement les références des méthodes/algorithmes qui ne sont pas les vôtres ?** | Références bibliographiques incluses pour : Pisinger instances, Martello-Toth bound, algorithmes génétiques, Ant Colony Optimization (Dorigo). |
| [OK] | **Avez-vous inclus tous les détails de votre setup expérimental pour reproduire les résultats ?** | Configuration détaillée : Python 3.14, specs machine, paramètres des algorithmes, seed pour reproductibilité (42), timeout (60s). |
| [OK] | **Avez-vous fourni des courbes, résultats numériques et barres d'erreur ?** | 11 visualisations générées : comparaison temps, gap à l'optimum, heatmaps, boxplots avec écart-type, analyse de scalabilité. |
| [OK] | **Avez-vous commenté et interprété les différents résultats présentés ?** | Chaque graphique accompagné d'une analyse. Synthèse des forces/faiblesses de chaque algorithme. |
| [OK] | **Avez-vous inclus toutes les données, code, instructions d'installation et d'exécution ?** | README complet, requirements.txt, structure de projet claire, données Pisinger téléchargeables automatiquement. |
| [OK] | **Avez-vous conçu le code de manière unifiée pour faciliter l'ajout de nouvelles méthodes ?** | Architecture modulaire : chaque algorithme dans un fichier séparé, interface commune (weights, values, capacity) -> (items, value, weight). |
| [OK] | **Avez-vous vérifié que les résultats des différentes expériences sont comparables ?** | Même instances de test, même timeout, même seed, exécution séquentielle sur même machine. |
| [OK] | **Avez-vous suffisamment commenté votre code ?** | Docstrings pour toutes les fonctions, commentaires explicatifs pour les parties complexes, type hints Python. |
| [OK] | **Avez-vous ajouté une documentation complète du code fourni ?** | Documentation dans README.md, docstrings détaillées, exemples d'utilisation dans chaque module. |
| [OK] | **Avez-vous fourni le planning prévisionnel et le planning final, et discuté des aspects organisationnels ?** | PLANNING.md contient planning prévu (06/11) vs réel, écarts justifiés, répartition du travail. |
| [OK] | **Avez-vous décrit précisément les algorithmes non vus en cours ?** | Détails pour : FPTAS (schéma d'approximation), Ant Colony (phéromones, évaporation), B&B Least-Cost (borne Martello-Toth), Iterative Deepening. |
| [OK] | **Avez-vous fourni le pourcentage de charge de travail entre les membres du groupe ?** | Distribution : Membre 1 (20%), Membre 2 (20%), Membre 3 (25%), Membre 4 (20%), Membre 5 (15%). Voir PLANNING.md. |
| [OK] | **Avez-vous envoyé le travail à temps ?** | Soumission avant le 16/12/2025 22:00. Archive .zip organisée avec code, données, rapport. |

---

## Suggested Guideline for Grading Scale - Auto-évaluation

### Minimum Requirements (6/20) [OK]

| Exigence | Statut | Commentaire |
|----------|--------|-------------|
| Algorithme Greedy (0/1) implémenté et fonctionnel | [OK] | 3 variantes : ratio, valeur, poids |
| Programmation dynamique implémentée et fonctionnelle | [OK] | Bottom-Up + Top-Down avec mémoïsation |
| Évaluation rigoureuse sur données artificielles | [OK] | Générateur avec 8 distributions différentes |
| Rapport présenté proprement | [OK] | Structure claire, figures, tableaux |
| Code source présenté proprement | [OK] | Commenté, typé, modulaire |

### Good Level (10/20) [OK]

| Exigence | Statut | Commentaire |
|----------|--------|-------------|
| Chaque membre a programmé une approche différente | [OK] | 5 membres, 12 algorithmes |
| Brute-force implémenté | [OK] | Avec élagage précoce |
| Branch-and-bound implémenté | [OK] | 4 versions : BFS, DFS, Least-Cost, IDDFS |
| Approche Greedy implémentée | [OK] | 3 variantes comptant comme 1 méthode |
| Programmation dynamique implémentée | [OK] | 2 versions + optimisation espace |
| Évaluations expérimentales rigoureuses | [OK] | Timeout, répétitions, statistiques |
| Au moins 4 problèmes de benchmarks testés | [OK] | Instances Pisinger (100-10000 items) |
| Code et rapport propres | [OK] | Architecture claire |
| Questions checklist répondues | [OK] | Ce document |

### Very Good Level (12/20) [OK]

| Exigence | Statut | Commentaire |
|----------|--------|-------------|
| Exigences 6/20 et 10/20 satisfaites | [OK] | Voir ci-dessus |
| Algorithmes évalués sur grande partie des benchmarks | [OK] | 5 catégories Pisinger |
| Setup expérimental clair avec commentaires | [OK] | Analyse détaillée des résultats |
| Une méthode additionnelle (total > taille groupe) | [OK] | 12 algorithmes > 5 membres |

### Excellent Level (14/20) [OK]

| Exigence | Statut | Commentaire |
|----------|--------|-------------|
| Exigences jusqu'à 12/20 satisfaites | [OK] | Voir ci-dessus |
| Deux nouvelles méthodes implémentées | [OK] | FPTAS, Ant Colony, B&B variants |
| Plus grande partie des benchmarks traitée | [OK] | Toutes tailles Pisinger |
| Setup expérimental très clair avec commentaires approfondis | [OK] | 11 graphiques d'analyse |
| Code extrêmement propre et clair | [OK] | Type hints, docstrings |
| Rapport extrêmement propre et clair | [OK] | Structure IEEE-like |

### Outstanding Level (16/20) [OK]

| Exigence | Statut | Commentaire |
|----------|--------|-------------|
| Tous les algorithmes de la liste programmés | [OK] | Brute-force, B&B (4 versions), Greedy (3), DP (2), FPTAS, Fractional, Genetic, Ant Colony, Randomized |
| Partie importante des benchmarks externes traitée | [OK] | Instances Pisinger complètes |
| Analyse de quelle méthode est efficace pour quelles instances | [OK] | Voir visualisations et rapport |
| Modélisation d'un problème réel en Knapsack | [OK] | Exemple : allocation de budget marketing |
| Problème réel résolu avec les algorithmes | [OK] | Voir `real_world_example.py` |
| Interface graphique (optionnel) | [--] | Non prioritaire |

---

## Résumé

| Niveau | Score Visé | Atteint |
|--------|-----------|---------|
| Minimum (6/20) | 6 | [OK] |
| Good (10/20) | 10 | [OK] |
| Very Good (12/20) | 12 | [OK] |
| Excellent (14/20) | 14 | [OK] |
| Outstanding (16/20) | 16 | [OK] |

**Score auto-évalué : 16/20** (hors qualité de la soutenance orale)

---

## Algorithmes Implémentés (Récapitulatif)

### Liste Complète

1. [OK] **Brute-Force** - Exploration exhaustive
2. [OK] **Branch-and-Bound BFS** - Best-First Search
3. [OK] **Branch-and-Bound DFS** - Depth-First Search  
4. [OK] **Branch-and-Bound Least-Cost** - Borne Martello-Toth
5. [OK] **Branch-and-Bound IDDFS** - Iterative Deepening
6. [OK] **Greedy Ratio** - Sélection par ratio v/w
7. [OK] **Greedy Value** - Sélection par valeur
8. [OK] **Greedy Weight** - Sélection par poids croissant
9. [OK] **DP Bottom-Up** - Programmation dynamique itérative
10. [OK] **DP Top-Down** - Avec mémoïsation
11. [OK] **FPTAS** - Fully Polynomial Time Approximation Scheme
12. [OK] **Fractional Approximation** - 2-approximation basée sur relaxation
13. [OK] **Genetic Algorithm** - Méta-heuristique évolutionnaire
14. [OK] **Ant Colony** - Optimisation par colonies de fourmis
15. [OK] **Randomized** - Approche probabiliste

**Total : 15 algorithmes** (12 algorithmes principaux + variantes)

---

**RAPPEL IMPORTANT** : Cette checklist complète DOIT être incluse dans le rapport final. L'absence de cette checklist commentée peut entraîner une division de la note par 2.
