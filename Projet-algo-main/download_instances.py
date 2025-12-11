#!/usr/bin/env python3
"""
Script pour télécharger les instances Pisinger depuis GitHub
Source: https://github.com/dnlfm/knapsack-01-instances
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.instance_loader import PisingerInstanceLoader, PISINGER_INSTANCES


def main():
    print("=" * 60)
    print("TÉLÉCHARGEMENT DES INSTANCES PISINGER")
    print("=" * 60)
    print("\nSource: https://github.com/dnlfm/knapsack-01-instances")
    print("        /pisinger_instances_01_KP/large_scale\n")
    
    loader = PisingerInstanceLoader()
    
    # Afficher les instances disponibles
    print("Instances disponibles:")
    for category, instances in PISINGER_INSTANCES.items():
        print(f"\n  [{category}] - {len(instances)} instances")
        for name in instances:
            print(f"    • {name}")
    
    # Choix utilisateur
    print("\n" + "-" * 60)
    print("Options de téléchargement:")
    print("  1. Small (100 objets) - Rapide")
    print("  2. Small + Medium (100-500 objets)")
    print("  3. Small + Medium + Large (100-2000 objets)")
    print("  4. Tout (inclut very_large jusqu'à 10000 objets)")
    print("  5. Quitter")
    
    try:
        choice = input("\nVotre choix (1-5): ").strip()
    except KeyboardInterrupt:
        print("\n\nAnnulé.")
        return
    
    categories_map = {
        '1': ['small'],
        '2': ['small', 'medium'],
        '3': ['small', 'medium', 'large'],
        '4': ['small', 'medium', 'large', 'very_large'],
    }
    
    if choice == '5':
        print("Au revoir!")
        return
    
    if choice not in categories_map:
        print("Choix invalide. Téléchargement des instances small par défaut.")
        choice = '1'
    
    categories = categories_map[choice]
    
    print(f"\nTéléchargement des catégories: {', '.join(categories)}")
    print("-" * 60)
    
    results = loader.download_all_instances(categories)
    
    # Résumé
    success = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    print(f"[OK] Téléchargées avec succès: {success}")
    print(f"[ECHEC] Échecs: {failed}")
    
    # Afficher les instances locales
    print("\nInstances disponibles localement:")
    local = loader.list_local_instances()
    
    for category, names in local.items():
        print(f"\n  [{category}]")
        for name in names:
            inst = loader.load_instance_by_name(name, download_if_missing=False)
            if inst:
                opt = f", optimal={inst.optimal_value}" if inst.optimal_value else ""
                print(f"    • {name}: n={inst.n}, C={inst.capacity}{opt}")


if __name__ == "__main__":
    main()
