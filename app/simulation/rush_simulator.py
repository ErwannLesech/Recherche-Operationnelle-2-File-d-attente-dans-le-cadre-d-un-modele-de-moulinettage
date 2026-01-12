"""
Module de simulation du système de moulinette.

Ce module implémente le simulateur complet de la moulinette EPITA,
combinant les modèles de files d'attente avec les personas et patterns
d'utilisation.

Architecture de la moulinette (simplifiée):
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Arrivées   │────▶│   File 1    │────▶│  Runners    │
│  (tags git) │     │  (buffer)   │     │ (serveurs)  │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  Résultats  │
                                        │  (feedback) │
                                        └─────────────┘

Modélisation:
- File 1: M/M/c/K (buffer limité, plusieurs runners)
- Runners: Temps de service variable selon complexité du projet

Ce module permet de:
1. Simuler des périodes avec charge variable
2. Analyser les métriques de performance
3. Identifier les goulots d'étranglement
4. Recommander des ajustements d'infrastructure

Auteurs: ERO2 Team Markov Moulinette Configurators - EPITA
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

from ..models import MMcKQueue, MMcQueue, MM1Queue, MD1Queue
from ..models.base_queue import QueueMetrics, SimulationResults
from ..personas import Persona, PersonaFactory, StudentType
from ..personas.usage_patterns import UsagePattern, PatternFactory, AcademicCalendar
from app.config.server_config import ServerConfig
from .moulinette_system import MoulinetteSystem


@dataclass
class SimulationConfig:
    """
    Configuration d'une simulation.
    
    Définit les paramètres de la simulation:
    - Durée et granularité temporelle
    - Personas à inclure
    - Pattern d'utilisation
    - Configuration serveur
    """
    duration_hours: float = 24.0          # Durée de simulation
    time_step_minutes: float = 15.0       # Pas de temps pour analyse
    
    # Personas
    personas: Dict[StudentType, Persona] = field(
        default_factory=PersonaFactory.create_all_personas
    )
    
    # Pattern temporel
    usage_pattern: UsagePattern = field(
        default_factory=PatternFactory.create_default_pattern
    )
    
    # Configuration serveur
    server_config: ServerConfig = field(default_factory=ServerConfig)
    
    # Paramètres de simulation
    n_simulation_runs: int = 10           # Répétitions Monte Carlo
    seed: Optional[int] = 42              # Graine aléatoire
    
    # Options
    include_weekend: bool = False
    start_hour: int = 0
    start_day: int = 0  # 0=lundi
    
    # Deadline optionnelle
    deadline_at_hour: Optional[float] = None


@dataclass
class SimulationReport:
    """
    Rapport détaillé d'une simulation.
    
    Contient toutes les métriques et recommandations.
    """
    # Configuration
    config: SimulationConfig = None
    
    # Métriques théoriques
    theoretical_metrics: QueueMetrics = None
    
    # Résultats de simulation
    simulation_results: List[SimulationResults] = field(default_factory=list)
    
    # Métriques agrégées
    avg_waiting_time: float = 0.0
    std_waiting_time: float = 0.0
    avg_system_time: float = 0.0
    max_queue_length: int = 0
    avg_queue_length: float = 0.0
    rejection_rate: float = 0.0
    throughput: float = 0.0
    utilization: float = 0.0
    
    # Traces temporelles (moyennées)
    time_series: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Analyse des pics
    peak_hours: List[int] = field(default_factory=list)
    peak_load: float = 0.0
    
    # Recommandations
    recommendations: List[str] = field(default_factory=list)
    optimal_servers: int = 0
    estimated_cost: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit le rapport en dictionnaire."""
        return {
            'avg_waiting_time': self.avg_waiting_time,
            'std_waiting_time': self.std_waiting_time,
            'avg_system_time': self.avg_system_time,
            'max_queue_length': self.max_queue_length,
            'avg_queue_length': self.avg_queue_length,
            'rejection_rate': self.rejection_rate,
            'throughput': self.throughput,
            'utilization': self.utilization,
            'peak_hours': self.peak_hours,
            'peak_load': self.peak_load,
            'optimal_servers': self.optimal_servers,
            'estimated_cost': self.estimated_cost,
            'recommendations': self.recommendations,
        }


class RushSimulator:
    """
    Simulateur de périodes de rush utilisant MoulinetteSystem.
    """
    def __init__(self, moulinette_system: MoulinetteSystem):
        """
        Initialise le simulateur avec un système moulinette.

        Args:
            moulinette_system: Instance de MoulinetteSystem.
        """
        self.moulinette_system = moulinette_system

    def run(self, arrival_rate: float, duration: float):
        """
        Exécute la simulation complète via MoulinetteSystem.

        Args:
            arrival_rate: Taux d'arrivée moyen.
            duration: Durée de la simulation en heures.

        Returns:
            Résultats de simulation sous forme de métriques.
        """
        # Utilise la chaîne de queues configurée dans MoulinetteSystem
        return self.moulinette_system.simulate(arrival_rate, duration)
