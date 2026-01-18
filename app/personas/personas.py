"""
Module de définition des personas utilisateurs.

Ce module modélise les différents types d'utilisateurs de la moulinette
avec leurs patterns de comportement spécifiques.

Contexte EPITA:
- Prépa (SUP/SPÉ): Travail régulier, soumissions réparties, moins de rush
- Ingénieur (ING1-5): Travail plus concentré, procrastination, rush de deadline

Les personas permettent de:
1. Simuler des patterns de charge réalistes
2. Prévoir les pics de demande
3. Dimensionner l'infrastructure selon les populations

Modélisation basée sur:
- Taux de soumission moyen par étudiant
- Variance du comportement (régularité)
- Patterns horaires (jour/nuit, weekend)
- Patterns de deadline (rush avant échéance)

Auteurs: ERO2 Team Markov Moulinette Configurators - EPITA
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable
import numpy as np


class StudentType(Enum):
    """Types d'étudiants à EPITA."""
    PREPA = "prepa"                # Prépa SUP/SPE (900 étudiants)
    INGENIEUR = "ingenieur"        # Ingénieurs (600 étudiants)
    ADMIN = "admin"                # Admin/Assistants (90 utilisateurs)


@dataclass
class Persona:
    """
    Représente un type d'utilisateur avec son comportement.
    
    Attributs comportementaux:
    - base_submission_rate: Taux de base de soumissions/heure
    - variance_coefficient: Régularité (0=constant, 1=très variable)
    - rush_multiplier: Facteur multiplicatif en période de rush
    - peak_hours: Heures de pic d'activité
    - weekend_factor: Réduction d'activité le weekend
    
    Distribution d'arrivée:
    Le taux d'arrivée effectif λ(t) varie selon:
    λ(t) = base_rate × hour_factor(t) × day_factor(t) × rush_factor(t)
    """
    name: str
    student_type: StudentType
    
    # Comportement de base
    base_submission_rate: float = 1.0  # soumissions/heure/étudiant
    variance_coefficient: float = 0.5  # C² de la variance
    
    # Patterns temporels
    peak_hours: List[int] = field(default_factory=lambda: [14, 15, 16, 21, 22, 23])
    off_peak_factor: float = 0.3  # Réduction hors heures de pointe
    weekend_factor: float = 0.4   # Réduction le weekend
    night_factor: float = 0.1     # Réduction la nuit (0h-8h)
    
    # Comportement de rush (deadline)
    rush_multiplier: float = 3.0  # Multiplication du taux en rush
    hours_before_deadline_rush: float = 24.0  # Début du rush
    procrastination_level: float = 0.5  # 0=régulier, 1=procrastinateur
    
    # Temps de traitement (impact sur μ)
    avg_test_complexity: float = 1.0  # Multiplicateur de temps de service
    
    # Population
    population_size: int = 100  # Nombre d'étudiants de ce type
    
    def get_arrival_rate(
        self,
        hour: int,
        is_weekend: bool = False,
        hours_to_deadline: Optional[float] = None
    ) -> float:
        """
        Calcule le taux d'arrivée effectif à un instant donné.
        
        λ(t) = population × base_rate × hour_factor × day_factor × rush_factor
        
        Args:
            hour: Heure de la journée (0-23)
            is_weekend: Si c'est le weekend
            hours_to_deadline: Heures avant la deadline (None si pas de deadline)
            
        Returns:
            Taux d'arrivée λ (soumissions/heure pour ce groupe)
        """
        rate = self.base_submission_rate * self.population_size
        
        # Facteur horaire
        if 0 <= hour < 8:
            rate *= self.night_factor
        elif hour in self.peak_hours:
            rate *= 1.0  # Plein régime
        else:
            rate *= self.off_peak_factor
        
        # Facteur weekend
        if is_weekend:
            rate *= self.weekend_factor
        
        # Facteur rush (deadline)
        if hours_to_deadline is not None and hours_to_deadline > 0:
            if hours_to_deadline <= self.hours_before_deadline_rush:
                # Modèle exponentiel de rush
                # Plus on approche de la deadline, plus le rush est intense
                rush_intensity = 1 - (hours_to_deadline / self.hours_before_deadline_rush)
                rush_intensity = rush_intensity ** (1 / (1 + self.procrastination_level))
                rate *= 1 + (self.rush_multiplier - 1) * rush_intensity
        
        return rate
    
    def generate_arrivals(
        self,
        duration_hours: float,
        start_hour: int = 0,
        is_weekend: bool = False,
        deadline_at_hour: Optional[float] = None,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Génère les temps d'arrivée pour une période donnée.
        
        Utilise un processus de Poisson non-homogène avec thinning.
        
        Args:
            duration_hours: Durée de la simulation
            start_hour: Heure de début
            is_weekend: Si c'est le weekend
            deadline_at_hour: Heure de la deadline (depuis le début)
            seed: Graine aléatoire
            
        Returns:
            Array des temps d'arrivée
        """
        rng = np.random.default_rng(seed)
        
        # Trouver le taux maximum pour le thinning
        max_rate = self.base_submission_rate * self.population_size * self.rush_multiplier
        
        # Générer avec processus homogène de taux max_rate
        n_arrivals_expected = int(max_rate * duration_hours * 1.5)
        interarrivals = rng.exponential(1/max_rate, n_arrivals_expected)
        arrival_times = np.cumsum(interarrivals)
        arrival_times = arrival_times[arrival_times <= duration_hours]
        
        # Thinning: accepter chaque arrivée avec probabilité λ(t)/max_rate
        accepted = []
        for t in arrival_times:
            current_hour = (start_hour + int(t)) % 24
            hours_to_deadline = deadline_at_hour - t if deadline_at_hour else None
            
            rate_t = self.get_arrival_rate(current_hour, is_weekend, hours_to_deadline)
            accept_prob = rate_t / max_rate
            
            if rng.random() < accept_prob:
                accepted.append(t)
        
        return np.array(accepted)
    
    def get_hourly_rates(
        self,
        is_weekend: bool = False,
        hours_to_deadline: float = None
    ) -> np.ndarray:
        """
        Retourne les taux horaires pour une journée entière.
        
        Args:
            is_weekend: Si c'est le weekend
            hours_to_deadline: Heures avant deadline à minuit
            
        Returns:
            Array de 24 valeurs (taux par heure)
        """
        rates = []
        for hour in range(24):
            htd = hours_to_deadline - (24 - hour) if hours_to_deadline else None
            if htd is not None and htd < 0:
                htd = None
            rates.append(self.get_arrival_rate(hour, is_weekend, htd))
        return np.array(rates)


class PersonaFactory:
    """Factory pour créer des personas pré-configurés."""
    
    @staticmethod
    def create_prepa(population: int = 900) -> Persona:
        """
        Crée un persona Prépa (SUP/SPE combinés).
        
        Caractéristiques:
        - 900 étudiants
        - Tag groupé lors des rendus (burst sur 15 min max)
        - Traitement plus long: 1 job/min/serveur
        - Arrivée quasi-simultanée lors des deadlines
        """
        return Persona(
            name="Prépa (SUP/SPE)",
            student_type=StudentType.PREPA,
            base_submission_rate=0.1,  # Faible hors deadline (tag uniquement au rendu)
            variance_coefficient=0.9,   # Très variable (burst)
            peak_hours=[14, 15, 16, 17],  # Heures de rendu typiques
            off_peak_factor=0.05,  # Très peu d'activité hors rendu
            weekend_factor=0.1,
            night_factor=0.01,
            rush_multiplier=1.0,  # Pas de rush progressif, burst direct
            hours_before_deadline_rush=0.0,
            procrastination_level=0.0,
            avg_test_complexity=1.0,  # Temps de traitement: 1 min/serveur
            population_size=population
        )
    
    @staticmethod
    def create_ingenieur(population: int = 600) -> Persona:
        """
        Crée un persona Ingénieur.
        
        Caractéristiques:
        - 600 étudiants
        - Tag régulier (~3 tags/heure en continu)
        - Traitement rapide: 4 jobs/min/serveur (0.25 min/job)
        - Flux continu toute la journée
        
        Calculs:
        - λ = 3 tags/h/user × 600 users = 1800 tags/h = 30 tags/min
        - μ = 4 jobs/min/serveur
        - Avec 20 serveurs: capacité = 80 jobs/min (> 30, donc stable)
        """
        return Persona(
            name="Ingénieur",
            student_type=StudentType.INGENIEUR,
            base_submission_rate=2.0,  # 3 tags/heure par étudiant
            variance_coefficient=0.3,   # Assez régulier
            peak_hours=[9, 10, 11, 14, 15, 16, 17, 20, 21, 22],  # Journée étendue
            off_peak_factor=0.5,  # Activité continue
            weekend_factor=0.4,
            night_factor=0.15,
            rush_multiplier=1.0,  # Pas de rush, flux continu
            hours_before_deadline_rush=0.0,
            procrastination_level=0.0,
            avg_test_complexity=0.25,  # Temps de traitement: 4 jobs/min = 0.25 min/job
            population_size=population
        )
    
    @staticmethod
    def create_admin(population: int = 90) -> Persona:
        """
        Crée un persona Admin/Assistants.
        
        Caractéristiques:
        - 90 utilisateurs
        - Tag ponctuel (~4/heure pour l'ensemble)
        - Traitement: 1 job/min/serveur
        - Usage très sporadique
        """
        return Persona(
            name="Admin/Assistants",
            student_type=StudentType.ADMIN,
            base_submission_rate=0.044,  # ~4 tags/h pour 90 users = 0.044/user/h
            variance_coefficient=0.2,
            peak_hours=[9, 10, 11, 14, 15, 16],  # Heures de bureau
            off_peak_factor=0.1,
            weekend_factor=0.02,
            night_factor=0.0,
            rush_multiplier=1.0,  # Pas de rush
            hours_before_deadline_rush=0.0,
            procrastination_level=0.0,
            avg_test_complexity=1.0,  # Temps de traitement: 1 min/serveur
            population_size=population
        )
    
    @staticmethod
    def create_all_personas() -> Dict[StudentType, Persona]:
        """Crée tous les personas avec populations par défaut."""
        return {
            StudentType.PREPA: PersonaFactory.create_prepa(),
            StudentType.INGENIEUR: PersonaFactory.create_ingenieur(),
            StudentType.ADMIN: PersonaFactory.create_admin(),
        }
    
    @staticmethod
    def get_total_population(personas: Dict[StudentType, Persona]) -> int:
        """Calcule la population totale."""
        return sum(p.population_size for p in personas.values())
    
    @staticmethod
    def get_combined_arrival_rate(
        personas: Dict[StudentType, Persona],
        hour: int,
        is_weekend: bool = False,
        hours_to_deadline: Optional[float] = None
    ) -> float:
        """
        Calcule le taux d'arrivée combiné de tous les personas.
        
        λ_total(t) = Σ λ_persona(t)
        
        Returns:
            Taux d'arrivée total
        """
        return sum(
            p.get_arrival_rate(hour, is_weekend, hours_to_deadline)
            for p in personas.values()
        )
