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
    Simulateur de périodes de rush.
    
    Combine les modèles de file d'attente avec les patterns
    d'utilisation pour simuler des scénarios réalistes.
    
    Fonctionnalités:
    - Simulation avec charge variable dans le temps
    - Analyse des pics et goulots d'étranglement
    - Recommandations d'auto-scaling
    - Comparaison de scénarios
    
    Exemple:
        >>> config = SimulationConfig(
        ...     duration_hours=24,
        ...     deadline_at_hour=20.0  # Deadline à 20h
        ... )
        >>> simulator = RushSimulator(config)
        >>> report = simulator.run()
        >>> print(f"Temps d'attente moyen: {report.avg_waiting_time:.2f}h")
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialise le simulateur.
        
        Args:
            config: Configuration de la simulation
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)
    
    def run(self) -> SimulationReport:
        """
        Exécute la simulation complète.
        
        Returns:
            SimulationReport avec tous les résultats
        """
        report = SimulationReport(config=self.config)
        
        # 1. Calculer les taux d'arrivée par période
        arrival_rates = self._compute_arrival_rates()
        
        # 2. Calculer le taux moyen pour les métriques théoriques
        avg_lambda = np.mean(arrival_rates)
        report.theoretical_metrics = self._compute_theoretical_metrics(avg_lambda)
        
        # 3. Exécuter les simulations Monte Carlo
        report.simulation_results = self._run_monte_carlo(arrival_rates)
        
        # 4. Agréger les résultats
        self._aggregate_results(report)
        
        # 5. Analyser les pics
        self._analyze_peaks(report, arrival_rates)
        
        # 6. Générer les recommandations
        self._generate_recommendations(report, arrival_rates)
        
        return report
    
    def _compute_arrival_rates(self) -> np.ndarray:
        """
        Calcule les taux d'arrivée pour chaque pas de temps.
        
        Combine les contributions de tous les personas avec
        le pattern d'utilisation.
        
        Returns:
            Array des taux λ(t) pour chaque intervalle
        """
        config = self.config
        n_steps = int(config.duration_hours * 60 / config.time_step_minutes)
        rates = np.zeros(n_steps)
        
        for i in range(n_steps):
            # Calculer l'heure et le jour
            total_hours = i * config.time_step_minutes / 60
            current_hour = (config.start_hour + int(total_hours)) % 24
            current_day = (config.start_day + int(total_hours / 24)) % 7
            is_weekend = current_day >= 5
            
            # Heures avant deadline
            hours_to_deadline = None
            if config.deadline_at_hour is not None:
                hours_to_deadline = config.deadline_at_hour - total_hours
                if hours_to_deadline < 0:
                    hours_to_deadline = None
            
            # Somme des contributions de tous les personas
            total_rate = 0.0
            for persona in config.personas.values():
                rate = persona.get_arrival_rate(
                    current_hour, 
                    is_weekend,
                    hours_to_deadline
                )
                total_rate += rate
            
            # Appliquer le pattern global
            pattern_factor = config.usage_pattern.hourly_factors[current_hour]
            pattern_factor *= config.usage_pattern.daily_factors[current_day]
            
            rates[i] = total_rate * pattern_factor
        
        return rates
    
    def _compute_theoretical_metrics(
        self,
        avg_lambda: float
    ) -> QueueMetrics:
        """
        Calcule les métriques théoriques pour le taux moyen.
        
        Args:
            avg_lambda: Taux d'arrivée moyen
            
        Returns:
            QueueMetrics théoriques
        """
        server_config = self.config.server_config
        
        # Créer la file M/M/c/K correspondante
        queue = MMcKQueue(
            lambda_rate=avg_lambda,
            mu_rate=server_config.service_rate,
            c=server_config.n_servers,
            K=server_config.buffer_size
        )
        
        return queue.compute_theoretical_metrics()
    
    def _run_monte_carlo(
        self,
        arrival_rates: np.ndarray
    ) -> List[SimulationResults]:
        """
        Exécute plusieurs simulations Monte Carlo.
        
        Pour capturer la variabilité stochastique, on répète
        la simulation plusieurs fois avec des seeds différentes.
        
        Args:
            arrival_rates: Taux d'arrivée par période
            
        Returns:
            Liste des résultats de chaque run
        """
        results = []
        config = self.config
        server_config = config.server_config
        
        for run in range(config.n_simulation_runs):
            seed = config.seed + run if config.seed else None
            
            # Simulation avec taux variable
            result = self._simulate_variable_rate(
                arrival_rates,
                server_config,
                seed
            )
            results.append(result)
        
        return results
    
    def _simulate_variable_rate(
        self,
        arrival_rates: np.ndarray,
        server_config: ServerConfig,
        seed: Optional[int]
    ) -> SimulationResults:
        """
        Simule une file avec taux d'arrivée variable.
        
        Utilise une approche par fenêtre temporelle:
        chaque intervalle est simulé avec son propre λ.
        
        Args:
            arrival_rates: Taux par intervalle
            server_config: Configuration serveur
            seed: Graine aléatoire
            
        Returns:
            SimulationResults agrégés
        """
        rng = np.random.default_rng(seed)
        
        config = self.config
        step_hours = config.time_step_minutes / 60
        
        # Accumulateurs pour toute la simulation
        all_arrivals = []
        all_departures = []
        all_waiting = []
        all_system = []
        
        current_time = 0.0
        last_departure = 0.0  # Pour continuité entre intervalles
        n_rejected = 0
        n_total = 0
        
        # Heap des serveurs (instants de fin)
        import heapq
        servers = []
        
        for i, lambda_rate in enumerate(arrival_rates):
            if lambda_rate <= 0:
                current_time += step_hours
                continue
            
            # Générer arrivées pour cet intervalle
            n_expected = lambda_rate * step_hours
            n_arrivals = rng.poisson(n_expected)
            
            if n_arrivals == 0:
                current_time += step_hours
                continue
            
            # Temps d'arrivée uniformes dans l'intervalle
            arrivals = current_time + np.sort(rng.uniform(0, step_hours, n_arrivals))
            n_total += n_arrivals
            
            for t_arrival in arrivals:
                # Nettoyer les serveurs terminés
                while servers and servers[0] <= t_arrival:
                    heapq.heappop(servers)
                
                n_in_system = len(servers)
                
                if n_in_system < server_config.buffer_size:
                    # Accepter
                    # Temps de service (avec variance)
                    mean_service = 1 / server_config.service_rate
                    variance = mean_service * server_config.service_variance
                    service_time = max(0.01, rng.normal(mean_service, np.sqrt(variance)))
                    
                    if n_in_system < server_config.n_servers:
                        # Serveur disponible
                        t_start = t_arrival
                    else:
                        # Attendre le prochain serveur
                        t_start = heapq.heappop(servers)
                    
                    t_departure = t_start + service_time
                    heapq.heappush(servers, t_departure)
                    
                    all_arrivals.append(t_arrival)
                    all_departures.append(t_departure)
                    all_waiting.append(t_start - t_arrival)
                    all_system.append(t_departure - t_arrival)
                else:
                    # Rejeter
                    n_rejected += 1
            
            current_time += step_hours
        
        # Construire le résultat
        n_served = len(all_arrivals)
        
        if n_served == 0:
            return SimulationResults(n_arrivals=n_total, n_rejected=n_rejected)
        
        return SimulationResults(
            arrival_times=np.array(all_arrivals),
            departure_times=np.array(all_departures),
            waiting_times=np.array(all_waiting),
            system_times=np.array(all_system),
            n_arrivals=n_total,
            n_served=n_served,
            n_rejected=n_rejected
        )
    
    def _aggregate_results(self, report: SimulationReport) -> None:
        """
        Agrège les résultats des simulations Monte Carlo.
        
        Calcule moyennes et écarts-types des métriques.
        """
        results = report.simulation_results
        
        if not results:
            return
        
        # Collecter les métriques de chaque run
        waiting_times = []
        system_times = []
        queue_lengths = []
        rejections = []
        
        for res in results:
            if len(res.waiting_times) > 0:
                waiting_times.extend(res.waiting_times)
                system_times.extend(res.system_times)
            
            if res.n_arrivals > 0:
                rejections.append(res.n_rejected / res.n_arrivals)
        
        if waiting_times:
            report.avg_waiting_time = np.mean(waiting_times)
            report.std_waiting_time = np.std(waiting_times)
            report.avg_system_time = np.mean(system_times)
        
        if rejections:
            report.rejection_rate = np.mean(rejections)
        
        # Throughput et utilisation
        total_served = sum(r.n_served for r in results)
        total_time = self.config.duration_hours * len(results)
        
        if total_time > 0:
            report.throughput = total_served / total_time
            capacity = self.config.server_config.total_capacity
            report.utilization = report.throughput / capacity if capacity > 0 else 0
    
    def _analyze_peaks(
        self,
        report: SimulationReport,
        arrival_rates: np.ndarray
    ) -> None:
        """
        Analyse les périodes de pic de charge.
        
        Identifie les heures avec la charge la plus élevée
        et calcule les métriques de pic.
        """
        config = self.config
        step_hours = config.time_step_minutes / 60
        
        # Regrouper par heure
        hourly_rates = {}
        for i, rate in enumerate(arrival_rates):
            hour = (config.start_hour + int(i * step_hours)) % 24
            if hour not in hourly_rates:
                hourly_rates[hour] = []
            hourly_rates[hour].append(rate)
        
        # Moyenne par heure
        avg_hourly = {h: np.mean(r) for h, r in hourly_rates.items()}
        
        # Top 3 heures de pic
        sorted_hours = sorted(avg_hourly.items(), key=lambda x: -x[1])
        report.peak_hours = [h for h, _ in sorted_hours[:3]]
        report.peak_load = sorted_hours[0][1] if sorted_hours else 0
        
        # Série temporelle pour visualisation
        report.time_series['arrival_rate'] = arrival_rates
        report.time_series['hours'] = np.arange(len(arrival_rates)) * step_hours
    
    def _generate_recommendations(
        self,
        report: SimulationReport,
        arrival_rates: np.ndarray
    ) -> None:
        """
        Génère des recommandations d'optimisation.
        
        Analyse les métriques et propose des ajustements.
        """
        config = self.config
        server_config = config.server_config
        
        recommendations = []
        
        # 1. Vérifier le taux de rejet
        if report.rejection_rate > 0.05:
            recommendations.append(
                f"Taux de rejet élevé ({report.rejection_rate:.1%}). "
                f"Augmenter le buffer K ou le nombre de serveurs."
            )
        
        # 2. Vérifier l'utilisation
        if report.utilization > 0.85:
            recommendations.append(
                f"Utilisation critique ({report.utilization:.1%}). "
                f"Risque de saturation. Prévoir du scaling."
            )
        elif report.utilization < 0.3:
            recommendations.append(
                f"ℹUtilisation faible ({report.utilization:.1%}). "
                f"Possibilité de réduire les serveurs pour économiser."
            )
        
        # 3. Temps d'attente
        if report.avg_waiting_time > 0.5:  # Plus de 30 min
            recommendations.append(
                f"Temps d'attente moyen élevé ({report.avg_waiting_time*60:.0f} min). "
                f"Considérer plus de serveurs ou des serveurs plus rapides."
            )
        
        # 4. Calculer le nombre optimal de serveurs
        peak_lambda = np.max(arrival_rates)
        mu = server_config.service_rate
        
        # Pour ρ_target = 0.7 pendant les pics
        rho_target = 0.7
        optimal_c = int(np.ceil(peak_lambda / (mu * rho_target)))
        optimal_c = max(server_config.min_servers, min(optimal_c, server_config.max_servers))
        
        report.optimal_servers = optimal_c
        
        if optimal_c != server_config.n_servers:
            diff = optimal_c - server_config.n_servers
            if diff > 0:
                recommendations.append(
                    f"Recommandation: Ajouter {diff} serveur(s) pour les pics "
                    f"(passer de {server_config.n_servers} à {optimal_c})."
                )
            else:
                recommendations.append(
                    f"Possibilité de réduire de {-diff} serveur(s) "
                    f"(passer de {server_config.n_servers} à {optimal_c})."
                )
        
        # 5. Estimer les coûts
        hours = config.duration_hours
        current_cost = server_config.n_servers * hours * server_config.cost_per_server_hour
        rejection_cost = report.rejection_rate * sum(p.population_size for p in config.personas.values())
        rejection_cost *= server_config.cost_per_rejection * hours
        
        report.estimated_cost = current_cost + rejection_cost
        
        recommendations.append(
            f"Coût estimé: {report.estimated_cost:.2f}€ "
            f"(serveurs: {current_cost:.2f}€, rejets: {rejection_cost:.2f}€)"
        )
        
        report.recommendations = recommendations
    
    def compare_scenarios(
        self,
        server_counts: List[int]
    ) -> Dict[int, SimulationReport]:
        """
        Compare plusieurs configurations de serveurs.
        
        Utile pour l'optimisation coût/performance.
        
        Args:
            server_counts: Liste des nombres de serveurs à tester
            
        Returns:
            Dict mapping nb_serveurs -> rapport
        """
        results = {}
        
        for n_servers in server_counts:
            # Modifier la config
            config_copy = SimulationConfig(
                duration_hours=self.config.duration_hours,
                time_step_minutes=self.config.time_step_minutes,
                personas=self.config.personas,
                usage_pattern=self.config.usage_pattern,
                server_config=ServerConfig(
                    n_servers=n_servers,
                    service_rate=self.config.server_config.service_rate,
                    buffer_size=self.config.server_config.buffer_size,
                ),
                n_simulation_runs=self.config.n_simulation_runs,
                seed=self.config.seed,
                deadline_at_hour=self.config.deadline_at_hour
            )
            
            simulator = RushSimulator(config_copy)
            results[n_servers] = simulator.run()
        
        return results
    
    def find_optimal_configuration(
        self,
        max_waiting_time: float = 0.25,  # 15 min max
        max_rejection_rate: float = 0.02,  # 2% max
        budget: Optional[float] = None
    ) -> Tuple[int, SimulationReport]:
        """
        Trouve la configuration optimale sous contraintes.
        
        Args:
            max_waiting_time: Temps d'attente max acceptable (heures)
            max_rejection_rate: Taux de rejet max acceptable
            budget: Budget maximum (optionnel)
            
        Returns:
            Tuple (nb_serveurs optimal, rapport correspondant)
        """
        config = self.config.server_config
        
        best_servers = config.min_servers
        best_report = None
        
        for n_servers in range(config.min_servers, config.max_servers + 1):
            # Tester cette configuration
            test_config = SimulationConfig(
                duration_hours=self.config.duration_hours,
                time_step_minutes=self.config.time_step_minutes,
                personas=self.config.personas,
                usage_pattern=self.config.usage_pattern,
                server_config=ServerConfig(
                    n_servers=n_servers,
                    service_rate=config.service_rate,
                    buffer_size=config.buffer_size,
                    cost_per_server_hour=config.cost_per_server_hour,
                ),
                n_simulation_runs=max(3, self.config.n_simulation_runs // 2),
                seed=self.config.seed,
                deadline_at_hour=self.config.deadline_at_hour
            )
            
            simulator = RushSimulator(test_config)
            report = simulator.run()
            
            # Vérifier les contraintes
            if report.avg_waiting_time <= max_waiting_time and \
               report.rejection_rate <= max_rejection_rate:
                
                if budget is None or report.estimated_cost <= budget:
                    best_servers = n_servers
                    best_report = report
                    break  # Premier qui satisfait = minimal
        
        if best_report is None:
            # Prendre le maximum si rien ne satisfait
            best_servers = config.max_servers
            test_config.server_config.n_servers = best_servers
            simulator = RushSimulator(test_config)
            best_report = simulator.run()
        
        return best_servers, best_report
