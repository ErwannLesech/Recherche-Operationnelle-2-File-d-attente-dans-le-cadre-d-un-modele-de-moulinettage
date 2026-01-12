"""
Module de simulation des rushs Moulinette.

Ce module ne fait plus qu’appeler la moulinette avec un λ fixe.
Il génère maintenant dynamiquement une évolution temporelle
des arrivées λ(t) basée sur:

- Personas
- Modèles d’usage
- Mode rush
- Proximité deadline
- Intensité et durée du rush

Puis il délègue la simulation au MoulinetteSystem
qui renvoie un SimulationReport complet.

Auteurs: ERO2 Team Markov Moulinette Configurators - EPITA
"""

import numpy as np
from .moulinette_system import MoulinetteSystem, SimulationReport

class RushSimulator:
    """Simule des périodes de rush avec λ(t) évolutif."""

    def __init__(self, moulinette_system: MoulinetteSystem):
        self.moulinette_system = moulinette_system

    def generate_rush_profile(
        self,
        duration: float,
        base_rate: float,
        peak_multiplier: float = 3.0,
        rush_center: float = 0.7,
        rush_width: float = 0.2
    ):
        """Génère une fonction λ(t) représentant un rush (gaussienne)."""
        def λ(t):
            x = t / duration
            rush = np.exp(-((x - rush_center) ** 2) / (2 * rush_width ** 2))
            return base_rate * (1 + rush * (peak_multiplier - 1))
        return λ

    def run_rush(self, duration: float, base_rate: float) -> SimulationReport:
        profile = self.generate_rush_profile(duration, base_rate)
        return self.moulinette_system.simulate_evolving(profile, duration)