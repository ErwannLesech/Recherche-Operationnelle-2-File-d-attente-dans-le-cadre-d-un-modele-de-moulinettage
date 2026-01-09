"""
Module de représentation du système de moulinette complet.

Ce module encapsule toute l'architecture de la moulinette EPITA
en un objet cohérent permettant:
- Configuration centralisée
- Simulation de bout en bout
- Analyse de performance
- Optimisation

Architecture modélisée:
══════════════════════════════════════════════════════════════
                        MOULINETTE EPITA
══════════════════════════════════════════════════════════════

  Étudiants                                      Résultats
  ┌──────┐                                       ┌──────┐
  │ SUP  │──┐                                ┌──▶│ Pass │
  └──────┘  │                                │   └──────┘
  ┌──────┐  │    ┌─────────┐   ┌─────────┐   │   ┌──────┐
  │ SPÉ  │──┼───▶│ Buffer  │──▶│ Runners │───┼──▶│ Fail │
  └──────┘  │    │ (Queue) │   │ (c srv) │   │   └──────┘
  ┌──────┐  │    └─────────┘   └─────────┘   │   ┌──────┐
  │ ING1 │──┤         K              μ       └──▶│ Blck │
  └──────┘  │                                    └──────┘
  ┌──────┐  │                                     
  │ ING2 │──┤    λ(t) varie selon:
  └──────┘  │    - Heure du jour
  ┌──────┐  │    - Jour de la semaine  
  │ ING3 │──┘    - Proximité deadline
  └──────┘       - Type d'étudiant

══════════════════════════════════════════════════════════════

Auteurs: ERO2 Team Markov Moulinette Configurators - EPITA
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
from enum import Enum

from ..models import MMcKQueue, MMcQueue
from ..models.base_queue import QueueMetrics
from ..personas import Persona, PersonaFactory, StudentType
from ..personas.usage_patterns import (
    UsagePattern, PatternFactory, AcademicCalendar, DeadlineEvent
)
from app.config.server_config import ServerConfig, ServerConfigDefaults, DEFAULT_SERVER_CONFIG


class ScalingMode(Enum):
    """Modes d'auto-scaling."""
    FIXED = "fixed"              # Nombre fixe de serveurs
    SCHEDULED = "scheduled"      # Scaling programmé
    REACTIVE = "reactive"        # Scaling basé sur la charge actuelle
    PREDICTIVE = "predictive"    # Scaling basé sur les prédictions


@dataclass
class ScalingPolicy:
    """
    Politique d'auto-scaling.
    
    Définit quand et comment ajuster le nombre de serveurs.
    """
    mode: ScalingMode = ScalingMode.FIXED
    
    # Seuils pour scaling réactif
    scale_up_threshold: float = 0.8      # ρ pour ajouter serveurs
    scale_down_threshold: float = 0.3    # ρ pour retirer serveurs
    
    # Paramètres
    scale_up_increment: int = 2          # Serveurs à ajouter
    scale_down_increment: int = 1        # Serveurs à retirer
    cooldown_minutes: float = 10.0       # Délai entre ajustements
    
    # Limites
    min_servers: int = 2
    max_servers: int = 20
    
    # Planification (pour mode SCHEDULED)
    scheduled_servers: Dict[int, int] = field(default_factory=dict)
    # Format: {heure: nb_serveurs}
    
    def get_target_servers(
        self,
        current_load: float,
        current_servers: int,
        hour: int
    ) -> int:
        """
        Détermine le nombre cible de serveurs.
        
        Args:
            current_load: Utilisation actuelle ρ
            current_servers: Nombre actuel de serveurs
            hour: Heure actuelle
            
        Returns:
            Nombre cible de serveurs
        """
        if self.mode == ScalingMode.FIXED:
            return current_servers
        
        elif self.mode == ScalingMode.SCHEDULED:
            return self.scheduled_servers.get(hour, current_servers)
        
        elif self.mode == ScalingMode.REACTIVE:
            if current_load > self.scale_up_threshold:
                target = min(
                    current_servers + self.scale_up_increment,
                    self.max_servers
                )
            elif current_load < self.scale_down_threshold:
                target = max(
                    current_servers - self.scale_down_increment,
                    self.min_servers
                )
            else:
                target = current_servers
            return target
        
        elif self.mode == ScalingMode.PREDICTIVE:
            # TODO: Implémenter prédiction ML
            return current_servers
        
        return current_servers


@dataclass
class MoulinetteConfig:
    """
    Configuration complète du système moulinette.
    """
    # Serveurs
    server_config: ServerConfig = field(default_factory=ServerConfig)
    
    # Scaling
    scaling_policy: ScalingPolicy = field(default_factory=ScalingPolicy)
    
    # Utilisateurs
    personas: Dict[StudentType, Persona] = field(
        default_factory=PersonaFactory.create_all_personas
    )
    
    # Patterns
    usage_pattern: UsagePattern = field(
        default_factory=PatternFactory.create_default_pattern
    )
    
    # Calendrier
    calendar: Optional[AcademicCalendar] = None
    
    # Paramètres globaux
    name: str = "Moulinette EPITA"
    description: str = "Système de correction automatique"


@dataclass
class SystemState:
    """
    État instantané du système.
    
    Capture l'état à un instant t pour monitoring.
    """
    timestamp: float = 0.0
    
    # File d'attente
    queue_length: int = 0
    n_in_service: int = 0
    n_total: int = 0
    
    # Serveurs
    active_servers: int = 0
    utilization: float = 0.0
    
    # Métriques
    arrival_rate: float = 0.0
    current_throughput: float = 0.0
    
    # Compteurs cumulatifs
    total_arrivals: int = 0
    total_served: int = 0
    total_rejected: int = 0
    
    # Temps moyens (fenêtre glissante)
    avg_waiting_time: float = 0.0
    avg_system_time: float = 0.0


class MoulinetteSystem:
    """
    Représentation complète du système de moulinette.
    
    Cette classe centralise:
    - La configuration du système
    - Les modèles de file d'attente
    - La simulation
    - L'analyse et l'optimisation
    
    Exemple:
        >>> moulinette = MoulinetteSystem()
        >>> moulinette.configure(n_servers=4, buffer_size=100)
        >>> 
        >>> # Simuler une journée avec deadline
        >>> results = moulinette.simulate_day(
        ...     deadline_hour=23,
        ...     start_hour=8
        ... )
        >>> 
        >>> # Obtenir recommandations
        >>> reco = moulinette.get_scaling_recommendations()
    """
    
    def __init__(self, config: Optional[MoulinetteConfig] = None):
        """
        Initialise le système.
        
        Args:
            config: Configuration (défaut si None)
        """
        self.config = config or MoulinetteConfig()
        self._queue_model: Optional[MMcKQueue] = None
        self._state_history: List[SystemState] = []
    
    def configure(
        self,
        n_servers: Optional[int] = None,
        buffer_size: Optional[int] = None,
        service_rate: Optional[float] = None,
        **kwargs
    ) -> 'MoulinetteSystem':
        """
        Configure le système (API fluide).
        
        Args:
            n_servers: Nombre de runners
            buffer_size: Taille du buffer
            service_rate: Taux de service par runner
            
        Returns:
            self pour chaînage
        """
        sc = self.config.server_config
        
        if n_servers is not None:
            sc.n_servers = n_servers
        if buffer_size is not None:
            sc.buffer_size = buffer_size
        if service_rate is not None:
            sc.service_rate = service_rate
        
        # Invalider le cache
        self._queue_model = None
        
        return self
    
    def set_personas(
        self,
        personas: Dict[StudentType, Persona]
    ) -> 'MoulinetteSystem':
        """Configure les personas."""
        self.config.personas = personas
        return self
    
    def add_deadline(
        self,
        name: str,
        hours_from_now: float,
        intensity: float = 1.0
    ) -> 'MoulinetteSystem':
        """Ajoute une deadline."""
        from datetime import datetime, timedelta
        
        if self.config.calendar is None:
            self.config.calendar = AcademicCalendar()
        
        deadline = DeadlineEvent(
            name=name,
            deadline=datetime.now() + timedelta(hours=hours_from_now),
            intensity=intensity
        )
        self.config.calendar.deadlines.append(deadline)
        
        return self
    
    def get_queue_model(
        self,
        arrival_rate: Optional[float] = None
    ) -> MMcKQueue:
        """
        Retourne le modèle de file d'attente.
        
        Args:
            arrival_rate: Taux d'arrivée (calcul automatique si None)
            
        Returns:
            Modèle M/M/c/K configuré
        """
        if arrival_rate is None:
            # Calculer le taux moyen
            arrival_rate = self._compute_average_arrival_rate()
        
        sc = self.config.server_config
        
        return MMcKQueue(
            lambda_rate=arrival_rate,
            mu_rate=sc.service_rate,
            c=sc.n_servers,
            K=sc.buffer_size
        )
    
    def _compute_average_arrival_rate(self) -> float:
        """Calcule le taux d'arrivée moyen global."""
        total = 0.0
        for persona in self.config.personas.values():
            # Taux moyen sur une journée
            daily_rates = persona.get_hourly_rates()
            total += np.mean(daily_rates)
        return total
    
    def get_theoretical_metrics(
        self,
        arrival_rate: Optional[float] = None
    ) -> QueueMetrics:
        """
        Calcule les métriques théoriques.
        
        Args:
            arrival_rate: Taux d'arrivée (auto si None)
            
        Returns:
            QueueMetrics théoriques
        """
        model = self.get_queue_model(arrival_rate)
        return model.compute_theoretical_metrics()
    
    def analyze_stability(self) -> Dict[str, Any]:
        """
        Analyse la stabilité du système.
        
        Vérifie si le système peut absorber la charge
        dans différents scénarios.
        
        Returns:
            Dict avec analyse de stabilité
        """
        sc = self.config.server_config
        capacity = sc.total_capacity
        
        # Calculer les taux pour différents scénarios
        scenarios = {
            'normal': self._compute_average_arrival_rate(),
            'peak': self._compute_peak_arrival_rate(),
            'deadline_rush': self._compute_deadline_rush_rate(),
        }
        
        results = {
            'capacity': capacity,
            'scenarios': {}
        }
        
        for name, lambda_rate in scenarios.items():
            rho = lambda_rate / capacity if capacity > 0 else float('inf')
            stable = rho < 1
            
            if stable:
                model = MMcKQueue(lambda_rate, sc.service_rate, sc.n_servers, sc.buffer_size)
                metrics = model.compute_theoretical_metrics()
                wq = metrics.Wq
                pk = metrics.Pk
            else:
                wq = float('inf')
                pk = 1.0
            
            results['scenarios'][name] = {
                'arrival_rate': lambda_rate,
                'utilization': min(rho, 1.0),
                'stable': stable,
                'avg_waiting_time': wq,
                'rejection_rate': pk,
                'servers_needed': int(np.ceil(lambda_rate / sc.service_rate * 1.3))
            }
        
        return results
    
    def _compute_peak_arrival_rate(self) -> float:
        """Calcule le taux d'arrivée aux heures de pointe."""
        total = 0.0
        for persona in self.config.personas.values():
            rates = persona.get_hourly_rates()
            total += np.max(rates)
        return total
    
    def _compute_deadline_rush_rate(self) -> float:
        """Calcule le taux d'arrivée en période de rush deadline."""
        total = 0.0
        for persona in self.config.personas.values():
            # Simuler rush: 2h avant deadline, heure de pointe
            rates = persona.get_hourly_rates(hours_to_deadline=2.0)
            total += np.max(rates)
        return total
    
    def get_scaling_recommendations(
        self,
        target_waiting_time: float = 0.1,  # 6 min max
        target_rejection_rate: float = 0.01,  # 1% max
        budget_per_hour: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Génère des recommandations de scaling.
        
        Args:
            target_waiting_time: Temps d'attente cible (heures)
            target_rejection_rate: Taux de rejet cible
            budget_per_hour: Budget horaire max
            
        Returns:
            Dict avec recommandations détaillées
        """
        sc = self.config.server_config
        
        # Taux de pic
        peak_lambda = self._compute_peak_arrival_rate()
        rush_lambda = self._compute_deadline_rush_rate()
        
        recommendations = {
            'current': {
                'servers': sc.n_servers,
                'capacity': sc.total_capacity,
                'hourly_cost': sc.get_hourly_cost()
            },
            'scenarios': {},
            'recommendations': []
        }
        
        # Analyser différents scénarios
        for scenario, lambda_rate in [
            ('normal', self._compute_average_arrival_rate()),
            ('peak', peak_lambda),
            ('rush', rush_lambda)
        ]:
            # Trouver le nombre optimal de serveurs
            optimal_c = self._find_optimal_servers(
                lambda_rate,
                target_waiting_time,
                target_rejection_rate
            )
            
            # Calculer les métriques avec optimal
            if optimal_c > 0:
                model = MMcKQueue(
                    lambda_rate, sc.service_rate, optimal_c, sc.buffer_size
                )
                metrics = model.compute_theoretical_metrics()
            else:
                metrics = None
            
            recommendations['scenarios'][scenario] = {
                'arrival_rate': lambda_rate,
                'optimal_servers': optimal_c,
                'current_utilization': lambda_rate / sc.total_capacity if sc.total_capacity > 0 else 0,
                'optimal_utilization': lambda_rate / (optimal_c * sc.service_rate) if optimal_c > 0 else 0,
                'estimated_waiting': metrics.Wq * 60 if metrics else float('inf'),  # minutes
                'estimated_rejection': metrics.Pk if metrics else 1.0,
                'hourly_cost': optimal_c * sc.cost_per_server_hour
            }
        
        # Générer recommandations textuelles
        rush_optimal = recommendations['scenarios']['rush']['optimal_servers']
        normal_optimal = recommendations['scenarios']['normal']['optimal_servers']
        
        if rush_optimal > sc.n_servers:
            recommendations['recommendations'].append(
                f"🔴 Insuffisant pour les rushs: Prévoir {rush_optimal} serveurs "
                f"(actuellement {sc.n_servers})"
            )
        
        if normal_optimal < sc.n_servers:
            savings = (sc.n_servers - normal_optimal) * sc.cost_per_server_hour
            recommendations['recommendations'].append(
                f"💰 Économie possible: Réduire à {normal_optimal} serveurs "
                f"en période normale (économie: {savings:.2f}€/h)"
            )
        
        if rush_optimal > normal_optimal + 2:
            recommendations['recommendations'].append(
                f"📊 Auto-scaling recommandé: Varier entre {normal_optimal} "
                f"et {rush_optimal} serveurs selon la charge"
            )
        
        # Politique de scaling suggérée
        recommendations['suggested_policy'] = {
            'mode': 'scheduled' if rush_optimal > normal_optimal + 2 else 'fixed',
            'min_servers': max(1, normal_optimal - 1),
            'max_servers': min(sc.max_servers, rush_optimal + 2),
            'scheduled': {
                # Heures creuses
                **{h: normal_optimal for h in range(2, 8)},
                # Heures normales
                **{h: normal_optimal for h in [8, 9, 10, 11, 12, 13]},
                # Heures de pointe
                **{h: int((normal_optimal + rush_optimal) / 2) for h in [14, 15, 16, 17]},
                # Soirée
                **{h: rush_optimal for h in [20, 21, 22, 23, 0, 1]},
            }
        }
        
        return recommendations
    
    def _find_optimal_servers(
        self,
        lambda_rate: float,
        target_wq: float,
        target_pk: float
    ) -> int:
        """Trouve le nombre optimal de serveurs."""
        sc = self.config.server_config
        mu = sc.service_rate
        K = sc.buffer_size
        
        # Minimum pour stabilité
        c_min = max(1, int(np.ceil(lambda_rate / mu)))
        
        for c in range(c_min, sc.max_servers + 1):
            try:
                model = MMcKQueue(lambda_rate, mu, c, K)
                metrics = model.compute_theoretical_metrics()
                
                if metrics.Wq <= target_wq and metrics.Pk <= target_pk:
                    return c
            except:
                continue
        
        return sc.max_servers
    
    def estimate_cost(
        self,
        hours: float = 24.0,
        include_scaling: bool = True
    ) -> Dict[str, float]:
        """
        Estime les coûts de fonctionnement.
        
        Args:
            hours: Durée en heures
            include_scaling: Inclure les variations de scaling
            
        Returns:
            Dict avec détail des coûts
        """
        sc = self.config.server_config
        
        if include_scaling and self.config.scaling_policy.mode != ScalingMode.FIXED:
            # Calculer coût avec scaling
            scheduled = self.config.scaling_policy.scheduled_servers
            if scheduled:
                avg_servers = np.mean(list(scheduled.values()))
            else:
                avg_servers = sc.n_servers
        else:
            avg_servers = sc.n_servers
        
        server_cost = avg_servers * hours * sc.cost_per_server_hour
        
        # Estimer les rejets
        stability = self.analyze_stability()
        avg_rejection = np.mean([
            s['rejection_rate'] 
            for s in stability['scenarios'].values()
        ])
        
        # Population totale
        total_students = sum(
            p.population_size for p in self.config.personas.values()
        )
        
        # Coût des rejets (approximatif)
        rejection_cost = avg_rejection * total_students * hours * 0.1
        
        return {
            'server_cost': server_cost,
            'rejection_cost': rejection_cost,
            'total_cost': server_cost + rejection_cost,
            'cost_per_hour': (server_cost + rejection_cost) / hours,
            'avg_servers': avg_servers
        }
    
    def create_heatmap_data(
        self,
        param1_name: str = 'n_servers',
        param1_range: List[Any] = None,
        param2_name: str = 'arrival_rate_factor',
        param2_range: List[Any] = None,
        metric: str = 'avg_waiting_time'
    ) -> Dict[str, Any]:
        """
        Génère les données pour une heatmap de paramètres.
        
        Utile pour visualiser l'impact des hyperparamètres
        sur les performances.
        
        Args:
            param1_name: Premier paramètre à varier
            param1_range: Valeurs du premier paramètre
            param2_name: Second paramètre à varier
            param2_range: Valeurs du second paramètre
            metric: Métrique à mesurer
            
        Returns:
            Dict avec données de heatmap
        """
        sc = self.config.server_config
        base_lambda = self._compute_average_arrival_rate()
        
        if param1_range is None:
            param1_range = list(range(1, 11))
        if param2_range is None:
            param2_range = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        
        # Matrice de résultats
        results = np.zeros((len(param1_range), len(param2_range)))
        
        for i, p1 in enumerate(param1_range):
            for j, p2 in enumerate(param2_range):
                # Configurer les paramètres
                if param1_name == 'n_servers':
                    c = p1
                else:
                    c = sc.n_servers
                
                if param2_name == 'arrival_rate_factor':
                    lambda_rate = base_lambda * p2
                else:
                    lambda_rate = base_lambda
                
                # Calculer la métrique
                try:
                    model = MMcKQueue(
                        lambda_rate, sc.service_rate, c, sc.buffer_size
                    )
                    metrics = model.compute_theoretical_metrics()
                    
                    if metric == 'avg_waiting_time':
                        results[i, j] = metrics.Wq * 60  # en minutes
                    elif metric == 'rejection_rate':
                        results[i, j] = metrics.Pk * 100  # en %
                    elif metric == 'utilization':
                        results[i, j] = metrics.rho * 100  # en %
                    else:
                        results[i, j] = getattr(metrics, metric, 0)
                except:
                    results[i, j] = np.nan
        
        return {
            'param1_name': param1_name,
            'param1_values': param1_range,
            'param2_name': param2_name,
            'param2_values': param2_range,
            'metric': metric,
            'data': results
        }
    
    def __repr__(self) -> str:
        sc = self.config.server_config
        n_students = sum(p.population_size for p in self.config.personas.values())
        return (
            f"MoulinetteSystem("
            f"servers={sc.n_servers}, "
            f"μ={sc.service_rate}/h, "
            f"K={sc.buffer_size}, "
            f"students={n_students})"
        )
