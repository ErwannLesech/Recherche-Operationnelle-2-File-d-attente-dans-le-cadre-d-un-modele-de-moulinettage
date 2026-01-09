# Moulinette Simulator - Application de simulation de file d'attente
# Pour la modélisation du système de correction automatique EPITA

__version__ = "1.0.0"
__author__ = "ERO2 Team"

# Exports principaux
from .models import MM1Queue, MMcQueue, MMcKQueue, MD1Queue, MG1Queue
from .personas import PersonaFactory, StudentType, Persona
from .simulation import RushSimulator, MoulinetteSystem
from .optimization import CostOptimizer, ScalingAdvisor

__all__ = [
    # Modèles
    'MM1Queue',
    'MMcQueue', 
    'MMcKQueue',
    'MD1Queue',
    'MG1Queue',
    # Personas
    'PersonaFactory',
    'StudentType',
    'Persona',
    # Simulation
    'RushSimulator',
    'MoulinetteSystem',
    # Optimisation
    'CostOptimizer',
    'ScalingAdvisor',
]
