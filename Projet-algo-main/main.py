#!/usr/bin/env python3
"""
Point d'entree principal du projet Knapsack
Lance l'interface graphique
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui import main as gui_main


if __name__ == "__main__":
    gui_main()
