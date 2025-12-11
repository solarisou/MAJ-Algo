"""
Exemple de Problème Réel : Allocation de Budget Marketing

Ce module démontre comment le problème du sac à dos peut être utilisé
pour résoudre un problème d'optimisation réel en entreprise.

Scénario:
---------
Une entreprise dispose d'un budget marketing limité et doit choisir
parmi plusieurs campagnes publicitaires. Chaque campagne a:
- Un coût (poids)
- Un retour sur investissement attendu (valeur)

L'objectif est de maximiser le ROI total tout en respectant le budget.

Ce type de problème se retrouve dans:
- Allocation de budget publicitaire
- Sélection de projets d'investissement
- Optimisation de portefeuille
- Planification de production
- Gestion de ressources IT
"""

import sys
import os

# Ajouter le chemin du projet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class MarketingCampaign:
    """Représente une campagne marketing"""
    name: str
    cost: int  # En milliers d'euros
    expected_roi: int  # Retour attendu en milliers d'euros
    duration_weeks: int
    target_audience: str
    risk_level: str  # low, medium, high
    
    def __repr__(self):
        return f"{self.name} (Coût: {self.cost}k€, ROI: {self.expected_roi}k€)"


class MarketingBudgetOptimizer:
    """
    Optimiseur de budget marketing utilisant les algorithmes du sac à dos.
    
    Démontre l'application pratique du knapsack 0/1 à un problème métier réel.
    """
    
    def __init__(self, budget: int):
        """
        Args:
            budget: Budget total disponible en milliers d'euros
        """
        self.budget = budget
        self.campaigns: List[MarketingCampaign] = []
    
    def add_campaign(self, campaign: MarketingCampaign) -> None:
        """Ajoute une campagne à la liste des options"""
        self.campaigns.append(campaign)
    
    def get_knapsack_inputs(self) -> Tuple[List[int], List[int], int]:
        """
        Convertit le problème métier en format knapsack.
        
        Returns:
            (weights, values, capacity) pour les algorithmes du sac à dos
        """
        weights = [c.cost for c in self.campaigns]
        values = [c.expected_roi for c in self.campaigns]
        return weights, values, self.budget
    
    def solve_with_dp(self) -> Dict:
        """
        Résout avec la programmation dynamique (optimal).
        """
        from src.algorithms.dynamic_programming import knapsack_bottom_up
        
        weights, values, capacity = self.get_knapsack_inputs()
        max_roi, selected_indices = knapsack_bottom_up(weights, values, capacity)
        
        return self._format_solution(selected_indices, max_roi, "Programmation Dynamique")
    
    def solve_with_greedy(self) -> Dict:
        """
        Résout avec l'algorithme glouton (approximation rapide).
        """
        from src.algorithms.greedy import greedy_algorithm_ratio
        
        weights, values, capacity = self.get_knapsack_inputs()
        selected_indices, total_roi, total_cost = greedy_algorithm_ratio(weights, values, capacity)
        
        return self._format_solution(selected_indices, total_roi, "Greedy (Ratio)")
    
    def solve_with_genetic(self) -> Dict:
        """
        Résout avec l'algorithme génétique (méta-heuristique).
        """
        from src.algorithms.genetic import genetic_algorithm_knapsack
        
        weights, values, capacity = self.get_knapsack_inputs()
        selected_indices, total_roi, total_cost = genetic_algorithm_knapsack(
            weights, values, capacity,
            population_size=50,
            generations=100
        )
        
        return self._format_solution(selected_indices, total_roi, "Algorithme Génétique")
    
    def solve_with_branch_and_bound(self) -> Dict:
        """
        Résout avec Branch and Bound (optimal).
        """
        from src.algorithms.branch_and_bound import branch_and_bound_least_cost
        
        weights, values, capacity = self.get_knapsack_inputs()
        selected_indices, total_roi, total_cost = branch_and_bound_least_cost(weights, values, capacity)
        
        return self._format_solution(selected_indices, total_roi, "Branch & Bound (Least-Cost)")
    
    def _format_solution(self, selected_indices: List[int], total_roi: int, 
                         method: str) -> Dict:
        """Formate la solution pour l'affichage"""
        selected_campaigns = [self.campaigns[i] for i in selected_indices]
        total_cost = sum(c.cost for c in selected_campaigns)
        
        return {
            'method': method,
            'selected_campaigns': selected_campaigns,
            'total_cost': total_cost,
            'total_roi': total_roi,
            'net_profit': total_roi - total_cost,
            'budget_used': f"{(total_cost / self.budget) * 100:.1f}%",
            'roi_percentage': f"{(total_roi / total_cost) * 100:.1f}%" if total_cost > 0 else "N/A"
        }
    
    def compare_all_methods(self) -> List[Dict]:
        """Compare toutes les méthodes de résolution"""
        results = []
        
        try:
            results.append(self.solve_with_dp())
        except Exception as e:
            print(f"DP failed: {e}")
        
        try:
            results.append(self.solve_with_greedy())
        except Exception as e:
            print(f"Greedy failed: {e}")
        
        try:
            results.append(self.solve_with_genetic())
        except Exception as e:
            print(f"Genetic failed: {e}")
        
        try:
            results.append(self.solve_with_branch_and_bound())
        except Exception as e:
            print(f"B&B failed: {e}")
        
        return results


def create_sample_campaigns() -> List[MarketingCampaign]:
    """
    Crée un ensemble réaliste de campagnes marketing.
    
    Basé sur des scénarios typiques d'une PME.
    """
    campaigns = [
        MarketingCampaign(
            name="Google Ads - Search",
            cost=15,
            expected_roi=45,
            duration_weeks=4,
            target_audience="Prospects actifs",
            risk_level="low"
        ),
        MarketingCampaign(
            name="Facebook/Instagram Ads",
            cost=12,
            expected_roi=30,
            duration_weeks=4,
            target_audience="18-35 ans",
            risk_level="medium"
        ),
        MarketingCampaign(
            name="LinkedIn B2B Campaign",
            cost=25,
            expected_roi=60,
            duration_weeks=6,
            target_audience="Décideurs entreprises",
            risk_level="medium"
        ),
        MarketingCampaign(
            name="Email Marketing Automation",
            cost=8,
            expected_roi=24,
            duration_weeks=8,
            target_audience="Clients existants",
            risk_level="low"
        ),
        MarketingCampaign(
            name="Influencer Partnership",
            cost=30,
            expected_roi=75,
            duration_weeks=2,
            target_audience="Gen Z",
            risk_level="high"
        ),
        MarketingCampaign(
            name="SEO Content Strategy",
            cost=20,
            expected_roi=40,
            duration_weeks=12,
            target_audience="Recherche organique",
            risk_level="low"
        ),
        MarketingCampaign(
            name="Podcast Sponsoring",
            cost=18,
            expected_roi=35,
            duration_weeks=4,
            target_audience="Professionnels tech",
            risk_level="medium"
        ),
        MarketingCampaign(
            name="Trade Show Exhibition",
            cost=40,
            expected_roi=100,
            duration_weeks=1,
            target_audience="B2B Industry",
            risk_level="medium"
        ),
        MarketingCampaign(
            name="YouTube Video Ads",
            cost=22,
            expected_roi=50,
            duration_weeks=4,
            target_audience="Mass market",
            risk_level="medium"
        ),
        MarketingCampaign(
            name="Retargeting Campaign",
            cost=10,
            expected_roi=32,
            duration_weeks=8,
            target_audience="Visiteurs site web",
            risk_level="low"
        ),
        MarketingCampaign(
            name="Print Magazine Ad",
            cost=35,
            expected_roi=45,
            duration_weeks=1,
            target_audience="Audience premium",
            risk_level="high"
        ),
        MarketingCampaign(
            name="Webinar Series",
            cost=5,
            expected_roi=15,
            duration_weeks=4,
            target_audience="Leads qualifiés",
            risk_level="low"
        ),
    ]
    
    return campaigns


def print_solution(solution: Dict) -> None:
    """Affiche une solution de manière formatée"""
    print(f"\n{'='*60}")
    print(f"Méthode: {solution['method']}")
    print(f"{'='*60}")
    
    print(f"\nRésumé Financier:")
    print(f"   Budget utilisé: {solution['total_cost']}k€ ({solution['budget_used']})")
    print(f"   ROI total attendu: {solution['total_roi']}k€")
    print(f"   Profit net estimé: {solution['net_profit']}k€")
    print(f"   Taux de retour: {solution['roi_percentage']}")
    
    print(f"\nCampagnes Sélectionnées ({len(solution['selected_campaigns'])}):")
    for i, campaign in enumerate(solution['selected_campaigns'], 1):
        ratio = campaign.expected_roi / campaign.cost
        print(f"   {i}. {campaign.name}")
        print(f"      Coût: {campaign.cost}k€ | ROI: {campaign.expected_roi}k€ | Ratio: {ratio:.2f}")
        print(f"      Durée: {campaign.duration_weeks} sem | Risque: {campaign.risk_level}")


def run_real_world_example():
    """
    Exécute l'exemple complet d'optimisation de budget marketing.
    """
    print("\n" + "="*70)
    print("EXEMPLE RÉEL: Optimisation de Budget Marketing")
    print("="*70)
    
    # Configuration
    BUDGET = 80  # 80 000 euros
    
    print(f"\nScénario:")
    print(f"   Une entreprise dispose d'un budget marketing de {BUDGET}k€")
    print(f"   Elle doit choisir parmi 12 campagnes publicitaires")
    print(f"   Objectif: Maximiser le retour sur investissement (ROI)")
    
    # Créer l'optimiseur
    optimizer = MarketingBudgetOptimizer(budget=BUDGET)
    
    # Ajouter les campagnes
    campaigns = create_sample_campaigns()
    for campaign in campaigns:
        optimizer.add_campaign(campaign)
    
    print(f"\nCampagnes Disponibles:")
    print("-" * 60)
    for i, c in enumerate(campaigns, 1):
        ratio = c.expected_roi / c.cost
        print(f"{i:2}. {c.name:<30} | Coût: {c.cost:3}k€ | ROI: {c.expected_roi:3}k€ | Ratio: {ratio:.2f}")
    
    print(f"\nRésolution avec différents algorithmes...")
    
    # Comparer les méthodes
    results = optimizer.compare_all_methods()
    
    # Afficher les résultats
    for solution in results:
        print_solution(solution)
    
    # Comparaison finale
    print("\n" + "="*70)
    print("COMPARAISON DES MÉTHODES")
    print("="*70)
    print(f"\n{'Méthode':<35} | {'Coût':>8} | {'ROI':>8} | {'Profit':>8}")
    print("-" * 65)
    
    for sol in results:
        print(f"{sol['method']:<35} | {sol['total_cost']:>6}k€ | {sol['total_roi']:>6}k€ | {sol['net_profit']:>6}k€")
    
    # Meilleure solution
    best = max(results, key=lambda x: x['total_roi'])
    print(f"\n[OK] Meilleure solution: {best['method']}")
    print(f"   ROI maximal: {best['total_roi']}k€ pour un investissement de {best['total_cost']}k€")
    
    return results


# === Autre exemple: Sélection de Projets IT ===

@dataclass
class ITProject:
    """Représente un projet IT"""
    name: str
    cost_man_days: int
    business_value: int
    priority: str


def create_it_projects() -> List[ITProject]:
    """Crée une liste de projets IT typiques"""
    return [
        ITProject("Migration Cloud AWS", 45, 120, "high"),
        ITProject("Refactoring Legacy", 30, 50, "medium"),
        ITProject("Mobile App v2", 60, 150, "high"),
        ITProject("API Gateway", 20, 40, "medium"),
        ITProject("Data Lake", 80, 200, "high"),
        ITProject("Security Audit", 15, 35, "high"),
        ITProject("CI/CD Pipeline", 25, 60, "medium"),
        ITProject("Monitoring Dashboard", 10, 25, "low"),
        ITProject("Customer Portal", 40, 90, "medium"),
        ITProject("AI Chatbot", 35, 70, "low"),
    ]


def run_it_portfolio_example():
    """
    Exemple d'optimisation de portefeuille de projets IT.
    """
    print("\n" + "="*70)
    print("EXEMPLE RÉEL 2: Sélection de Projets IT")
    print("="*70)
    
    CAPACITY = 150  # 150 jours-homme disponibles
    
    print(f"\nScénario:")
    print(f"   L'équipe IT dispose de {CAPACITY} jours-homme ce trimestre")
    print(f"   Elle doit sélectionner les projets à plus forte valeur")
    
    projects = create_it_projects()
    
    # Conversion en format knapsack
    weights = [p.cost_man_days for p in projects]
    values = [p.business_value for p in projects]
    
    print(f"\nProjets Disponibles:")
    for i, p in enumerate(projects, 1):
        ratio = p.business_value / p.cost_man_days
        print(f"{i:2}. {p.name:<25} | Coût: {p.cost_man_days:3}j | Valeur: {p.business_value:3} | Ratio: {ratio:.2f}")
    
    # Résolution
    from src.algorithms.dynamic_programming import knapsack_bottom_up
    
    max_value, selected = knapsack_bottom_up(weights, values, CAPACITY)
    
    print(f"\n[OK] Solution Optimale:")
    print(f"   Valeur business totale: {max_value}")
    print(f"   Jours-homme utilisés: {sum(weights[i] for i in selected)}/{CAPACITY}")
    
    print(f"\nProjets Sélectionnés:")
    for idx in selected:
        p = projects[idx]
        print(f"   - {p.name} (Coût: {p.cost_man_days}j, Valeur: {p.business_value})")


if __name__ == "__main__":
    # Exécuter l'exemple marketing
    run_real_world_example()
    
    print("\n" + "="*70)
    
    # Exécuter l'exemple IT
    run_it_portfolio_example()
    
    print("\n" + "="*70)
    print("Exemples terminés!")
    print("="*70)
    print("\nCes exemples démontrent comment le problème du sac à dos")
    print("s'applique à des décisions réelles d'entreprise:")
    print("  - Allocation de budget marketing")
    print("  - Sélection de projets IT")
    print("  - Optimisation de portefeuille")
    print("  - Et bien d'autres domaines...")
