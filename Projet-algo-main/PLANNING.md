# Planning du Projet - Sac à Dos (Knapsack Problem)

## Planning Prévisionnel (Établi le 06/11/2025)

| Semaine | Dates | Tâches Prévues | Responsable | Temps Estimé |
|---------|-------|----------------|-------------|--------------|
| **S1** | 06-12 Nov | Recherche biblio, compréhension algorithmes | Tous | 10h |
| **S2** | 13-19 Nov | Implémentation Brute-Force + Greedy | Membre 1 | 8h |
| **S2** | 13-19 Nov | Implémentation Dynamic Programming | Membre 2 | 8h |
| **S3** | 20-26 Nov | Implémentation Branch-and-Bound v1 | Membre 3 | 10h |
| **S3** | 20-26 Nov | Implémentation FPTAS + Fractional | Membre 4 | 10h |
| **S4** | 27 Nov-03 Déc | Implémentation Genetic/Ant Colony | Membre 5 | 10h |
| **S4** | 27 Nov-03 Déc | Implémentation Randomized + B&B v2 | Membre 1 | 10h |
| **S5** | 04-10 Déc | Tests et benchmarks | Tous | 15h |
| **S6** | 11-16 Déc | Rédaction rapport + visualisations | Tous | 20h |
| **S7** | 17-19 Déc | Préparation soutenance | Tous | 10h |

**Total estimé:** ~100h de travail collectif

---

## Planning Réel (Actualisé)

| Semaine | Dates | Tâches Réalisées | Problèmes Rencontrés | Temps Réel |
|---------|-------|------------------|---------------------|------------|
| **S1** | 06-12 Nov | [OK] Recherche biblio complète | - | 12h |
| **S2** | 13-19 Nov | [OK] Brute-Force, Greedy (3 variants) | - | 9h |
| **S2** | 13-19 Nov | [OK] DP Bottom-Up + Top-Down | - | 10h |
| **S3** | 20-26 Nov | [OK] B&B BFS + DFS | Difficultés avec les bornes | 12h |
| **S3** | 20-26 Nov | [OK] FPTAS + Approximation fractionnelle | - | 8h |
| **S4** | 27 Nov-03 Déc | [OK] Genetic + Ant Colony | Paramétrage difficile | 14h |
| **S4** | 27 Nov-03 Déc | [OK] Randomized | - | 6h |
| **S5** | 04-10 Déc | [OK] B&B Least-Cost | - | 8h |
| **S5** | 04-10 Déc | [OK] Intégration instances Pisinger | Formats différents | 10h |
| **S6** | 11-16 Déc | [OK] Benchmark complet + visualisations | - | 15h |
| **S6** | 11-16 Déc | [OK] Générateur de problèmes | - | 5h |
| **S6** | 11-16 Déc | [EN COURS] Rédaction rapport | En cours | 10h |
| **S7** | 17-19 Déc | [A FAIRE] Préparation soutenance | - | - |

**Total réel:** ~119h de travail collectif

---

## Comparaison Prévu vs Réel

### Écarts et Justifications

| Aspect | Prévu | Réel | Écart | Justification |
|--------|-------|------|-------|---------------|
| Temps total | 100h | 119h | +19h | Ajout de méthodes supplémentaires |
| Branch-and-Bound | 1 version | 4 versions | +3 | Volonté d'explorer différentes stratégies |
| Benchmarks | Basiques | Pisinger + générés | ++ | Meilleure évaluation |
| Visualisations | Non prévu | 11 graphiques | ++ | Analyse approfondie |

### Difficultés Rencontrées

1. **Paramétrage Génétique/Ant Colony** (S4)
   - Problème: Convergence prématurée
   - Solution: Ajustement des taux de mutation et phéromones

2. **Intégration Pisinger** (S5)
   - Problème: Format d'instance différent
   - Solution: Création d'un loader universel

3. **Comparaison équitable** (S5-S6)
   - Problème: Timeout pour brute-force sur grandes instances
   - Solution: Limites de temps et catégorisation des instances

---

## Répartition du Travail

### Distribution par Membre

| Membre | Algorithmes Implémentés | Autres Contributions | % Travail |
|--------|------------------------|---------------------|-----------|
| **Membre 1** | Brute-Force, Randomized | Tests unitaires | 20% |
| **Membre 2** | DP (Bottom-Up, Top-Down) | Loader instances | 20% |
| **Membre 3** | B&B (BFS, DFS, LC) | Benchmark system | 25% |
| **Membre 4** | FPTAS, Fractional Approx | Visualisations | 20% |
| **Membre 5** | Genetic, Ant Colony | Rapport, Documentation | 15% |

**Note:** Les pourcentages reflètent l'effort global incluant implémentation, tests et documentation.

### Réunions de Suivi

| Date | Sujet | Participants | Décisions |
|------|-------|--------------|-----------|
| 08/11 | Kick-off | Tous | Répartition initiale |
| 15/11 | Point S2 | Tous | Validation premiers algos |
| 22/11 | Point S3 | Tous | Ajout méthodes B&B |
| 29/11 | Point S4 | Tous | Focus meta-heuristiques |
| 06/12 | Point S5 | Tous | Stratégie benchmarks |
| 13/12 | Point S6 | Tous | Review rapport |

---

## Jalons (Milestones)

### M1: Algorithmes de Base (19/11) [OK]
- [x] Brute-Force fonctionnel
- [x] Greedy (3 variantes)
- [x] Dynamic Programming

### M2: Algorithmes Avancés (03/12) [OK]
- [x] Branch-and-Bound
- [x] FPTAS
- [x] Meta-heuristiques

### M3: Évaluation (10/12) [OK]
- [x] Benchmark complet
- [x] Instances Pisinger
- [x] Visualisations

### M4: Livraison (16/12) [EN COURS]
- [x] Code complet
- [x] Documentation
- [ ] Rapport final
- [ ] Slides soutenance

---

## Procédures de Test

### Tests Unitaires
- Chaque algorithme testé sur instances connues
- Validation solution optimale (quand calculable)

### Tests d'Intégration
- Comparaison inter-algorithmes
- Cohérence des résultats

### Tests de Performance
- Mesure temps d'exécution
- Instances de taille croissante
- Timeout: 60s pour grandes instances

### Validation Benchmarks
- Instances Pisinger avec solution optimale connue
- Calcul du gap à l'optimum
