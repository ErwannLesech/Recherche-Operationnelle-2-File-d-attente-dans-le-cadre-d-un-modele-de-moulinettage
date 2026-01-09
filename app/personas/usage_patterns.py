"""
Module de patterns d'utilisation temporels.

Ce module définit les patterns d'utilisation de la moulinette
selon les périodes de l'année, jours de la semaine et heures.

Patterns identifiés:
1. Journaliers: Pics le soir (21h-23h), creux la nuit
2. Hebdomadaires: Plus faible le weekend, pic le dimanche soir
3. Annuels: Rush à chaque deadline de projet majeur

Ces patterns sont calibrés sur les observations du système EPITA
(données Grafana quand disponibles).

Auteurs: ERO2 Team Markov Moulinette Configurators - EPITA
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta


class TimeOfDay(Enum):
    """Périodes de la journée."""
    NIGHT = "night"           # 0h-6h
    EARLY_MORNING = "early"   # 6h-9h
    MORNING = "morning"       # 9h-12h
    AFTERNOON = "afternoon"   # 12h-18h
    EVENING = "evening"       # 18h-21h
    LATE_EVENING = "late"     # 21h-24h


class DayOfWeek(Enum):
    """Jours de la semaine."""
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


class AcademicPeriod(Enum):
    """Périodes académiques."""
    NORMAL = "normal"              # Période normale
    EXAM_WEEK = "exam_week"        # Semaine d'examens
    PROJECT_DEADLINE = "deadline"   # Deadline de projet
    VACATION = "vacation"           # Vacances
    RUSH_WEEK = "rush_week"        # Semaine de rush (soutenances)


@dataclass
class UsagePattern:
    """
    Définit un pattern d'utilisation temporel.
    
    Un pattern combine:
    - Facteurs horaires (24 valeurs)
    - Facteurs journaliers (7 valeurs)
    - Facteur de période académique
    
    Le taux effectif est:
    λ(t) = λ_base × hour_factor × day_factor × period_factor
    """
    name: str
    
    # Facteurs horaires (multiplicateurs pour chaque heure 0-23)
    hourly_factors: np.ndarray = field(
        default_factory=lambda: np.ones(24)
    )
    
    # Facteurs journaliers (lundi=0 à dimanche=6)
    daily_factors: np.ndarray = field(
        default_factory=lambda: np.ones(7)
    )
    
    # Facteurs de période académique
    period_factors: Dict[AcademicPeriod, float] = field(
        default_factory=lambda: {
            AcademicPeriod.NORMAL: 1.0,
            AcademicPeriod.EXAM_WEEK: 0.3,
            AcademicPeriod.PROJECT_DEADLINE: 3.0,
            AcademicPeriod.VACATION: 0.1,
            AcademicPeriod.RUSH_WEEK: 2.0,
        }
    )
    
    def get_factor(
        self,
        hour: int,
        day: int,
        period: AcademicPeriod = AcademicPeriod.NORMAL
    ) -> float:
        """
        Retourne le facteur multiplicatif total.
        
        Args:
            hour: Heure (0-23)
            day: Jour de la semaine (0=lundi, 6=dimanche)
            period: Période académique
            
        Returns:
            Facteur multiplicatif
        """
        h_factor = self.hourly_factors[hour % 24]
        d_factor = self.daily_factors[day % 7]
        p_factor = self.period_factors.get(period, 1.0)
        
        return h_factor * d_factor * p_factor
    
    def get_time_of_day(self, hour: int) -> TimeOfDay:
        """Détermine la période de la journée."""
        if 0 <= hour < 6:
            return TimeOfDay.NIGHT
        elif 6 <= hour < 9:
            return TimeOfDay.EARLY_MORNING
        elif 9 <= hour < 12:
            return TimeOfDay.MORNING
        elif 12 <= hour < 18:
            return TimeOfDay.AFTERNOON
        elif 18 <= hour < 21:
            return TimeOfDay.EVENING
        else:
            return TimeOfDay.LATE_EVENING


class PatternFactory:
    """Factory pour créer des patterns pré-configurés."""
    
    @staticmethod
    def create_default_pattern() -> UsagePattern:
        """
        Crée le pattern par défaut basé sur les observations.
        
        Heures de pointe: 14h-16h et 21h-23h
        Creux: 2h-8h
        Weekend réduit mais pic dimanche soir
        """
        # Facteurs horaires (normalisés autour de 1.0)
        hourly = np.array([
            0.05,  # 0h
            0.03,  # 1h
            0.02,  # 2h
            0.01,  # 3h
            0.01,  # 4h
            0.02,  # 5h
            0.05,  # 6h
            0.10,  # 7h
            0.20,  # 8h
            0.40,  # 9h
            0.60,  # 10h
            0.70,  # 11h
            0.50,  # 12h (pause déjeuner)
            0.60,  # 13h
            1.00,  # 14h (pic après-midi)
            1.00,  # 15h
            0.90,  # 16h
            0.70,  # 17h
            0.50,  # 18h (pause dîner)
            0.70,  # 19h
            0.90,  # 20h
            1.20,  # 21h (pic soir)
            1.30,  # 22h (pic maximal)
            0.80,  # 23h
        ])
        
        # Facteurs journaliers
        daily = np.array([
            1.0,   # Lundi
            1.0,   # Mardi
            1.0,   # Mercredi
            1.1,   # Jeudi (veille de weekend)
            0.9,   # Vendredi (début weekend)
            0.4,   # Samedi
            0.6,   # Dimanche (rush soir)
        ])
        
        return UsagePattern(
            name="Default EPITA Pattern",
            hourly_factors=hourly,
            daily_factors=daily
        )
    
    @staticmethod
    def create_deadline_pattern() -> UsagePattern:
        """
        Pattern de période de deadline.
        
        Intensification générale avec pic la nuit
        (nuits blanches avant deadline).
        """
        hourly = np.array([
            0.50,  # 0h - nuit blanche
            0.40,  # 1h
            0.30,  # 2h
            0.20,  # 3h
            0.15,  # 4h
            0.20,  # 5h
            0.30,  # 6h
            0.40,  # 7h
            0.50,  # 8h
            0.70,  # 9h
            0.80,  # 10h
            0.90,  # 11h
            0.70,  # 12h
            0.80,  # 13h
            1.00,  # 14h
            1.10,  # 15h
            1.20,  # 16h
            1.00,  # 17h
            0.80,  # 18h
            1.00,  # 19h
            1.30,  # 20h
            1.50,  # 21h
            1.80,  # 22h - pic maximal
            1.20,  # 23h
        ])
        
        daily = np.array([
            1.0,   # Lundi
            1.1,   # Mardi
            1.2,   # Mercredi
            1.3,   # Jeudi
            1.4,   # Vendredi
            1.2,   # Samedi
            1.5,   # Dimanche (rush deadline)
        ])
        
        return UsagePattern(
            name="Deadline Pattern",
            hourly_factors=hourly,
            daily_factors=daily,
            period_factors={
                AcademicPeriod.NORMAL: 1.0,
                AcademicPeriod.PROJECT_DEADLINE: 1.5,  # Déjà un pattern intense
            }
        )
    
    @staticmethod
    def create_exam_pattern() -> UsagePattern:
        """
        Pattern de période d'examens.
        
        Activité réduite car focus sur révisions.
        """
        hourly = np.array([
            0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0h-5h
            0.05, 0.10, 0.20, 0.30, 0.40, 0.30,  # 6h-11h
            0.20, 0.30, 0.40, 0.40, 0.30, 0.20,  # 12h-17h
            0.15, 0.20, 0.25, 0.30, 0.25, 0.10,  # 18h-23h
        ])
        
        daily = np.array([0.8, 0.8, 0.8, 0.8, 0.6, 0.3, 0.4])
        
        return UsagePattern(
            name="Exam Period Pattern",
            hourly_factors=hourly,
            daily_factors=daily
        )


@dataclass
class DeadlineEvent:
    """
    Représente une deadline de projet.
    
    Permet de modéliser le rush avant une échéance.
    """
    name: str
    deadline: datetime
    affected_student_types: List[str] = field(default_factory=list)
    intensity: float = 1.0  # Multiplicateur d'intensité
    rush_duration_hours: float = 48.0  # Durée du rush avant deadline
    
    def get_rush_factor(self, current_time: datetime) -> float:
        """
        Calcule le facteur de rush pour un instant donné.
        
        Le rush suit un modèle exponentiel croissant:
        factor = 1 + (max_factor - 1) * (1 - t/T)^α
        
        où t = temps restant, T = durée totale du rush
        
        Returns:
            Facteur multiplicatif (1.0 si hors rush)
        """
        time_to_deadline = (self.deadline - current_time).total_seconds() / 3600
        
        if time_to_deadline <= 0:
            return 0.5  # Après deadline, activité réduite
        
        if time_to_deadline > self.rush_duration_hours:
            return 1.0  # Pas encore en rush
        
        # Modèle de rush: intensité croissante exponentiellement
        progress = 1 - (time_to_deadline / self.rush_duration_hours)
        rush_factor = 1 + (self.intensity * 4) * (progress ** 2)
        
        return rush_factor


@dataclass 
class AcademicCalendar:
    """
    Calendrier académique avec deadlines et périodes.
    
    Permet de simuler une année entière avec les variations
    d'activité selon les événements.
    """
    deadlines: List[DeadlineEvent] = field(default_factory=list)
    exam_periods: List[Tuple[datetime, datetime]] = field(default_factory=list)
    vacation_periods: List[Tuple[datetime, datetime]] = field(default_factory=list)
    
    def get_period(self, dt: datetime) -> AcademicPeriod:
        """Détermine la période académique pour une date."""
        for start, end in self.vacation_periods:
            if start <= dt <= end:
                return AcademicPeriod.VACATION
        
        for start, end in self.exam_periods:
            if start <= dt <= end:
                return AcademicPeriod.EXAM_WEEK
        
        # Vérifier les deadlines
        for deadline in self.deadlines:
            time_to = (deadline.deadline - dt).total_seconds() / 3600
            if 0 < time_to <= deadline.rush_duration_hours:
                return AcademicPeriod.PROJECT_DEADLINE
        
        return AcademicPeriod.NORMAL
    
    def get_total_factor(
        self,
        dt: datetime,
        base_pattern: UsagePattern
    ) -> float:
        """
        Calcule le facteur total pour une date/heure.
        
        Combine pattern de base + deadline rushes.
        """
        hour = dt.hour
        day = dt.weekday()
        period = self.get_period(dt)
        
        base_factor = base_pattern.get_factor(hour, day, period)
        
        # Ajouter rush des deadlines
        rush_factor = 1.0
        for deadline in self.deadlines:
            rush_factor = max(rush_factor, deadline.get_rush_factor(dt))
        
        return base_factor * rush_factor
    
    @staticmethod
    def create_sample_semester() -> 'AcademicCalendar':
        """
        Crée un exemple de calendrier semestriel.
        
        Inclut quelques deadlines typiques d'EPITA.
        """
        # Date de référence: semestre de printemps 2026
        base = datetime(2026, 1, 15)
        
        deadlines = [
            DeadlineEvent(
                name="Mini-projet 1",
                deadline=base + timedelta(weeks=3),
                intensity=0.8
            ),
            DeadlineEvent(
                name="TP noté",
                deadline=base + timedelta(weeks=5),
                intensity=0.6
            ),
            DeadlineEvent(
                name="Projet majeur",
                deadline=base + timedelta(weeks=8),
                intensity=1.5,
                rush_duration_hours=72.0
            ),
            DeadlineEvent(
                name="Soutenance",
                deadline=base + timedelta(weeks=12),
                intensity=2.0,
                rush_duration_hours=96.0
            ),
        ]
        
        exam_periods = [
            (base + timedelta(weeks=6), base + timedelta(weeks=6, days=5)),
            (base + timedelta(weeks=14), base + timedelta(weeks=15)),
        ]
        
        vacation_periods = [
            (base + timedelta(weeks=7), base + timedelta(weeks=7, days=7)),
        ]
        
        return AcademicCalendar(
            deadlines=deadlines,
            exam_periods=exam_periods,
            vacation_periods=vacation_periods
        )
