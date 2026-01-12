"""
Module de représentation du système Moulinette complet.

Ce module ne se limite plus à un modèle théorique :
il permet maintenant de simuler des chaînes de files d'attente
avec des arrivées pouvant être:

- Fixes (λ constant)
- Évolutives (λ(t) variable dans le temps)
  → permet de simuler des rushs, variations journalières, effets deadline…

Auteurs: ERO2 Team Markov Moulinette Configurators - EPITA
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
from enum import Enum

from ..models.base_queue import GenericQueue, ChainQueue, QueueMetrics, SimulationResults
from ..personas import Persona, PersonaFactory, StudentType
from ..personas.usage_patterns import (
    UsagePattern, PatternFactory, AcademicCalendar, DeadlineEvent
)
from app.config.server_config import ServerConfig, ServerConfigDefaults, DEFAULT_SERVER_CONFIG


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
        Initialise le système avec une chaîne de queues par défaut (MMC -> MM1).
        
        Args:
            config: Configuration (défaut si None)
        """
        self.config = config or MoulinetteConfig()
        self._state_history: List[SystemState] = []

        # Configuration par défaut : MMC -> MM1
        mmc_queue = GenericQueue(lambda_rate=5, mu_rate=10, c=3, kendall_notation="M/M/c")
        mm1_queue = GenericQueue(lambda_rate=5, mu_rate=10, kendall_notation="M/M/1")
        self._queue_chain = ChainQueue([mmc_queue, mm1_queue])
    
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
    
    def configure_chain(self, queue_chain: List[GenericQueue]) -> 'MoulinetteSystem':
        """
        Configure une chaîne de queues pour la moulinette.

        Args:
            queue_chain: Liste de queues génériques représentant la chaîne.

        Returns:
            self pour chaînage
        """
        self._queue_chain = ChainQueue(queue_chain)
        return self

    def simulate_fixed(self, arrival_rate: float, duration: float) -> SimulationReport:
        """
        Simule la chaîne avec un taux d'arrivée constant λ.

        Args:
            arrival_rate: λ constant
            duration: durée en heures

        Returns:
            SimulationReport
        """
        metrics: SimulationResults = self._queue_chain.simulate(
            arrival_rate=arrival_rate,
            duration=duration
        )

        report = SimulationReport()
        report.simulation_results = [metrics]
        report.avg_waiting_time = np.mean(metrics.waiting_times) if len(metrics.waiting_times) > 0 else 0.0
        report.avg_system_time  = np.mean(metrics.system_times) if len(metrics.system_times) > 0 else 0.0
        report.avg_queue_length = np.mean(metrics.queue_length_trace) if len(metrics.queue_length_trace) > 0 else 0.0
        report.rejection_rate   = metrics.n_rejected / metrics.n_arrivals if metrics.n_arrivals > 0 else 0.0
        report.throughput       = metrics.n_served / duration  # durée en heures
        report.utilization      = np.mean(metrics.system_size_trace / self._queue_chain.total_servers())  # approximation
        return report

    def simulate_evolving(self, arrival_profile, duration: float, step_minutes: float = 5.0):
        """
        Solution finale : simulation continue avec transfert entre queues.
        L'état de chaque queue est préservé entre les pas de temps.
        """
        import heapq
        from app.models.base_queue import SimulationResults
        
        step_h = step_minutes / 60.0
        time = 0.0

        def resolve_lambda(t):
            if callable(arrival_profile):
                return arrival_profile(t)
            else:
                arr = [p for p in arrival_profile if p[0] <= t]
                return arr[-1][1] if arr else arrival_profile[0][1]

        # États persistants pour chaque queue
        queue_states = []
        for i, queue in enumerate(self._queue_chain.queues):
            queue_states.append({
                "event_heap": [],
                "queue_state": [],  # Liste de (customer_id, service_time, arrival_time)
                "busy_servers": 0,
                "service_start_times": {},
                "departure_times": {},
                "accepted_arrivals": [],
                "accepted_service_times": [],
                "waiting_times_list": [],
                "system_times_list": [],
                "n_rejected": 0,
                "customer_counter": 0,
                "pending_service_times": {},  # Pour stocker les temps de service pré-générés
                # Pour le traçage
                "time_trace": [],
                "queue_length_trace": [],
                "system_size_trace": []
            })

        # Boucle principale
        while time < duration:
            lam = resolve_lambda(time)
            step_end = min(time + step_h, duration)
            
            # ====================================================================
            # Traiter chaque queue
            # ====================================================================
            for i, queue in enumerate(self._queue_chain.queues):
                state = queue_states[i]
                c = queue.c
                K = queue.K
                
                # Générer une arrivée externe pour la première queue uniquement
                if i == 0 and lam > 0:
                    next_arrival = time + queue.rng.exponential(1 / lam)
                    if next_arrival < step_end:
                        customer_id = state["customer_counter"]
                        state["customer_counter"] += 1
                        heapq.heappush(state["event_heap"], (next_arrival, "arrival", customer_id))
                
                # Traiter tous les événements dans [time, step_end]
                departures_this_step = []
                
                while state["event_heap"] and state["event_heap"][0][0] < step_end:
                    event_time, event_type, customer_id = heapq.heappop(state["event_heap"])
                    system_size = len(state["queue_state"]) + state["busy_servers"]
                    
                    if event_type == "arrival":
                        # Vérifier capacité
                        if K is not None and system_size >= K:
                            state["n_rejected"] += 1
                            continue
                        
                        # Client accepté
                        state["accepted_arrivals"].append(event_time)
                        
                        # Récupérer ou générer le temps de service
                        if customer_id in state["pending_service_times"]:
                            service_time = state["pending_service_times"].pop(customer_id)
                        else:
                            service_time = queue._generate_service_times_for_model(1)[0]
                        
                        state["accepted_service_times"].append(service_time)
                        
                        # Serveur disponible ?
                        if state["busy_servers"] < c:
                            # Service immédiat
                            state["busy_servers"] += 1
                            state["service_start_times"][customer_id] = event_time
                            depart_time = event_time + service_time
                            state["departure_times"][customer_id] = depart_time
                            
                            state["waiting_times_list"].append(0.0)
                            state["system_times_list"].append(service_time)
                            
                            heapq.heappush(state["event_heap"], (depart_time, "departure", customer_id))
                        else:
                            # Mettre en file d'attente
                            state["queue_state"].append((customer_id, service_time, event_time))
                    
                    else:  # departure
                        state["busy_servers"] -= 1
                        departures_this_step.append((event_time, customer_id))
                        
                        # Servir le prochain client en attente
                        if state["queue_state"]:
                            next_id, next_service_time, next_arrival_time = state["queue_state"].pop(0)
                            state["busy_servers"] += 1
                            
                            state["service_start_times"][next_id] = event_time
                            next_depart = event_time + next_service_time
                            state["departure_times"][next_id] = next_depart
                            
                            wait_time = event_time - next_arrival_time
                            state["waiting_times_list"].append(wait_time)
                            state["system_times_list"].append(next_depart - next_arrival_time)
                            
                            heapq.heappush(state["event_heap"], (next_depart, "departure", next_id))
                
                # Enregistrer l'état actuel pour le traçage
                state["time_trace"].append(step_end)
                state["queue_length_trace"].append(len(state["queue_state"]))
                state["system_size_trace"].append(len(state["queue_state"]) + state["busy_servers"])
                
                # ================================================================
                # TRANSFERT VERS LA QUEUE SUIVANTE
                # ================================================================
                if i + 1 < len(self._queue_chain.queues) and departures_this_step:
                    next_state = queue_states[i + 1]
                    next_queue = self._queue_chain.queues[i + 1]
                    
                    for dep_time, old_id in departures_this_step:
                        # Créer un nouveau client dans la queue suivante
                        new_customer_id = next_state["customer_counter"]
                        next_state["customer_counter"] += 1
                        
                        # Générer son temps de service à l'avance
                        service_time = next_queue._generate_service_times_for_model(1)[0]
                        next_state["pending_service_times"][new_customer_id] = service_time
                        
                        # Planifier son arrivée
                        heapq.heappush(
                            next_state["event_heap"],
                            (dep_time, "arrival", new_customer_id)
                        )
            
            time = step_end

        # ====================================================================
        # Construire les résultats finaux
        # ====================================================================
        sim_results = []
        for i, state in enumerate(queue_states):
            sim_results.append(
                SimulationResults(
                    arrival_times=np.array(state["accepted_arrivals"]),
                    service_start_times=np.array(list(state["service_start_times"].values())),
                    departure_times=np.array(list(state["departure_times"].values())),
                    service_times=np.array(state["accepted_service_times"]),
                    waiting_times=np.array(state["waiting_times_list"]),
                    system_times=np.array(state["system_times_list"]),
                    n_arrivals=len(state["accepted_arrivals"]) + state["n_rejected"],
                    n_served=len(state["accepted_arrivals"]),
                    n_rejected=state["n_rejected"],
                    time_trace=np.array(state["time_trace"]),
                    queue_length_trace=np.array(state["queue_length_trace"]),
                    system_size_trace=np.array(state["system_size_trace"])
                )
            )

        # Créer le rapport
        report = SimulationReport()
        report.simulation_results = sim_results
        
        # Métriques globales
        valid_wait = [np.mean(r.waiting_times) for r in sim_results if len(r.waiting_times) > 0]
        valid_sys = [np.mean(r.system_times) for r in sim_results if len(r.system_times) > 0]
        valid_ql = [np.mean(r.queue_length_trace) for r in sim_results if len(r.queue_length_trace) > 0]
        
        report.avg_waiting_time = np.mean(valid_wait) if valid_wait else 0.0
        report.avg_system_time = np.mean(valid_sys) if valid_sys else 0.0
        report.avg_queue_length = np.mean(valid_ql) if valid_ql else 0.0
        report.rejection_rate = np.mean([r.n_rejected / r.n_arrivals if r.n_arrivals > 0 else 0.0 for r in sim_results])
        report.throughput = sum(r.n_served for r in sim_results) / duration
        
        total_servers = self._queue_chain.total_servers()
        valid_util = [r for r in sim_results if len(r.system_size_trace) > 0]
        if valid_util:
            report.utilization = np.mean([np.mean(r.system_size_trace) / total_servers for r in valid_util])
        else:
            report.utilization = 0.0

        # Traces globales (somme des longueurs de queue)
        if sim_results and all(len(r.time_trace) > 0 for r in sim_results):
            # Utiliser les traces de la dernière queue comme référence temporelle
            report.time_series["time"] = sim_results[-1].time_trace
            
            # Sommer les longueurs de queue de toutes les queues
            min_len = min(len(r.queue_length_trace) for r in sim_results)
            total_queue_length = np.zeros(min_len)
            for r in sim_results:
                total_queue_length += r.queue_length_trace[:min_len]
            
            report.time_series["queue_length"] = total_queue_length

        return report

    def get_queue_model(
        self,
        arrival_rate: Optional[float] = None
    ):
        """
        Cette méthode est désormais obsolète car la chaîne de queues est utilisée.
        """
        raise NotImplementedError("Cette méthode n'est plus utilisée. Configurez une chaîne de queues avec configure_chain().")
    
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
        Calcule les métriques théoriques en utilisant la chaîne de queues configurée.

        Args:
            arrival_rate: Taux d'arrivée moyen (utilisé pour la simulation).

        Returns:
            QueueMetrics agrégées de la chaîne de queues.
        """
        if not hasattr(self, '_queue_chain'):
            raise ValueError("La chaîne de queues n'est pas configurée.")

        # Utiliser la chaîne de queues pour calculer les métriques
        metrics = self._queue_chain.simulate(arrival_rate, duration=1.0)  # Durée fictive pour obtenir les métriques
        return metrics
    
    def analyze_stability(self) -> Dict[str, Any]:
        """
        Analyse la stabilité du système en utilisant ChainQueue.
        """
        scenarios = {
            'normal': self._compute_average_arrival_rate(),
            'peak': self._compute_peak_arrival_rate(),
            'deadline_rush': self._compute_deadline_rush_rate(),
        }

        results = {
            'capacity': self._queue_chain.compute_theoretical_metrics().rho,  # Utilisation globale
            'scenarios': {}
        }

        for name, lambda_rate in scenarios.items():
            try:
                self._queue_chain.update_arrival_rate(lambda_rate)
                metrics = self._queue_chain.compute_theoretical_metrics()
                results['scenarios'][name] = {
                    'arrival_rate': lambda_rate,
                    'utilization': metrics.rho,
                    'stable': metrics.rho < 1,
                    'avg_waiting_time': metrics.Wq,
                    'rejection_rate': metrics.Pk,
                }
            except Exception as e:
                results['scenarios'][name] = {
                    'error': str(e)
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
        target_waiting_time: float = 0.1,
        target_rejection_rate: float = 0.01,
        budget_per_hour: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Génère des recommandations de scaling en utilisant ChainQueue.
        """
        recommendations = {
            'current': {
                'servers': self.config.server_config.n_servers,
                'capacity': self._queue_chain.compute_theoretical_metrics().rho,
            },
            'scenarios': {}
        }

        for scenario, lambda_rate in [
            ('normal', self._compute_average_arrival_rate()),
            ('peak', self._compute_peak_arrival_rate()),
        ]:
            self._queue_chain.update_arrival_rate(lambda_rate)
            metrics = self._queue_chain.compute_theoretical_metrics()
            recommendations['scenarios'][scenario] = {
                'arrival_rate': lambda_rate,
                'utilization': metrics.rho,
                'avg_waiting_time': metrics.Wq,
                'rejection_rate': metrics.Pk,
            }

        return recommendations

    def _find_optimal_servers(
        self,
        lambda_rate: float,
        target_wq: float,
        target_pk: float
    ) -> int:
        """
        Trouve le nombre optimal de serveurs en utilisant ChainQueue.
        """
        for c in range(1, self.config.server_config.max_servers + 1):
            self._queue_chain.update_servers(c)
            self._queue_chain.update_arrival_rate(lambda_rate)
            metrics = self._queue_chain.compute_theoretical_metrics()

            if metrics.Wq <= target_wq and metrics.Pk <= target_pk:
                return c

        return self.config.server_config.max_servers
    
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
        Génère les données pour une heatmap en utilisant ChainQueue.
        """
        if param1_range is None:
            param1_range = list(range(1, 11))
        if param2_range is None:
            param2_range = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

        results = np.zeros((len(param1_range), len(param2_range)))

        for i, p1 in enumerate(param1_range):
            for j, p2 in enumerate(param2_range):
                self._queue_chain.update_servers(p1)
                self._queue_chain.update_arrival_rate(self._compute_average_arrival_rate() * p2)
                metrics = self._queue_chain.compute_theoretical_metrics()

                if metric == 'avg_waiting_time':
                    results[i, j] = metrics.Wq * 60
                elif metric == 'rejection_rate':
                    results[i, j] = metrics.Pk * 100
                elif metric == 'utilization':
                    results[i, j] = metrics.rho * 100

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
