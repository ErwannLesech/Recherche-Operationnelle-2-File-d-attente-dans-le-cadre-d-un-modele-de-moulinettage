"""
Modèle de file d'attente M/G/c.

M/G/c = Arrivées Markoviennes / Service Général / c serveurs

Ce modèle représente une file d'attente avec:
- Arrivées selon un processus de Poisson (M)
- Temps de service suivant une distribution générale (G)
- c serveurs en parallèle
- Capacité infinie

Auteurs: ERO2 Team Markov Moulinette Configurators - EPITA
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
from enum import Enum

from .base_queue import BaseQueueModel, QueueMetrics, SimulationResults


class ServiceDistribution(Enum):
    """Types de distribution de service supportés."""
    EXPONENTIAL = "exponential"
    DETERMINISTIC = "deterministic"
    UNIFORM = "uniform"
    GAMMA = "gamma"
    LOGNORMAL = "lognormal"


class MGcQueue(BaseQueueModel):
    """
    File d'attente M/G/c (Multi-serveur avec service général).
    
    Utilise l'approximation de Kingman/Allen-Cunneen pour les systèmes
    multi-serveurs avec distribution de service générale.
    
    Paramètres:
        lambda_rate: Taux d'arrivée (λ)
        service_mean: Temps moyen de service (1/μ)
        service_variance: Variance du temps de service
        c: Nombre de serveurs
    """
    
    def __init__(
        self,
        lambda_rate: float,
        service_mean: float,
        service_variance: float = None,
        c: int = 1,
        distribution: ServiceDistribution = ServiceDistribution.EXPONENTIAL,
        distribution_params: dict = None,
        seed: Optional[int] = None,
        allow_unstable: bool = True
    ):
        """
        Initialise le modèle M/G/c.
        
        Args:
            lambda_rate: Taux d'arrivée des clients
            service_mean: Temps moyen de service (E[S] = 1/μ)
            service_variance: Variance du temps de service (Var[S])
            c: Nombre de serveurs
            distribution: Type de distribution pour simulation
            distribution_params: Paramètres additionnels de la distribution
            seed: Graine pour le générateur aléatoire
            allow_unstable: Si True, permet les systèmes instables
        """
        mu_rate = 1.0 / service_mean if service_mean > 0 else float('inf')
        super().__init__(lambda_rate, mu_rate, c=c, K=None, seed=seed)
        
        self.service_mean = service_mean
        self.allow_unstable = allow_unstable
        
        # Variance par défaut = variance exponentielle (CV² = 1)
        if service_variance is None:
            self.service_variance = service_mean ** 2
        else:
            self.service_variance = service_variance
        
        # CV² = Var / E²
        self.cv_squared = self.service_variance / (service_mean ** 2) if service_mean > 0 else 1.0
        
        self.distribution = distribution
        self.distribution_params = distribution_params or {}
        
        # Vérification de stabilité
        rho = lambda_rate * service_mean / c
        if rho >= 1 and not allow_unstable:
            raise ValueError(
                f"Système instable: ρ = {rho:.4f} ≥ 1. "
                f"Augmentez c ou réduisez λ ou le temps de service."
            )
    
    def compute_erlang_c(self) -> float:
        """
        Calcule la probabilité d'attente (Erlang-C) pour M/M/c.
        Utilisé comme base pour l'approximation M/G/c.
        """
        a = self.lambda_rate / self.mu_rate  # Intensité de trafic
        c = self.c
        rho = self.rho
        
        if rho >= 1:
            return 1.0
        
        # Calcul de P0
        sum_terms = sum((a ** n) / np.math.factorial(n) for n in range(c))
        last_term = (a ** c) / (np.math.factorial(c) * (1 - rho))
        
        P0 = 1.0 / (sum_terms + last_term)
        
        # Erlang-C
        C = ((a ** c) / np.math.factorial(c)) * (1 / (1 - rho)) * P0
        
        return min(C, 1.0)
    
    def compute_theoretical_metrics(self) -> QueueMetrics:
        """
        Calcule les métriques théoriques pour M/G/c.
        
        Utilise l'approximation d'Allen-Cunneen:
        Wq(M/G/c) ≈ Wq(M/M/c) * (1 + CV²) / 2
        
        Cette approximation est exacte pour:
        - CV² = 1: M/M/c (exponentiel)
        - CV² = 0: M/D/c (déterministe)
        """
        rho = self.rho
        c = self.c
        mu = self.mu_rate
        lambda_rate = self.lambda_rate
        cv2 = self.cv_squared
        
        # Calcul Erlang-C
        C = self.compute_erlang_c()
        
        # Wq pour M/M/c
        Wq_mmc = C / (c * mu * (1 - rho)) if rho < 1 else float('inf')
        
        # Approximation Allen-Cunneen pour M/G/c
        Wq = Wq_mmc * (1 + cv2) / 2
        
        # Autres métriques
        Ws = self.service_mean
        W = Wq + Ws
        
        Lq = lambda_rate * Wq
        Ls = lambda_rate * Ws
        L = Lq + Ls
        
        # P0
        a = lambda_rate / mu
        sum_terms = sum((a ** n) / np.math.factorial(n) for n in range(c))
        last_term = (a ** c) / (np.math.factorial(c) * (1 - rho)) if rho < 1 else float('inf')
        P0 = 1.0 / (sum_terms + last_term) if rho < 1 else 0.0
        
        return QueueMetrics(
            rho=rho,
            L=L,
            Lq=Lq,
            Ls=Ls,
            W=W,
            Wq=Wq,
            Ws=Ws,
            P0=P0,
            Pk=0.0,
            lambda_eff=lambda_rate,
            throughput=lambda_rate
        )
    
    def _generate_service_time(self) -> float:
        """Génère un temps de service selon la distribution configurée."""
        mean = self.service_mean
        var = self.service_variance
        
        if self.distribution == ServiceDistribution.DETERMINISTIC:
            return mean
        
        elif self.distribution == ServiceDistribution.EXPONENTIAL:
            return self.rng.exponential(mean)
        
        elif self.distribution == ServiceDistribution.UNIFORM:
            # Uniform avec moyenne et variance données
            # Var = (b-a)² / 12, E = (a+b) / 2
            range_half = np.sqrt(3 * var)
            a = mean - range_half
            b = mean + range_half
            return max(0, self.rng.uniform(a, b))
        
        elif self.distribution == ServiceDistribution.GAMMA:
            # Gamma: E = k*θ, Var = k*θ²
            # k = E²/Var, θ = Var/E
            if var > 0:
                k = (mean ** 2) / var
                theta = var / mean
                return self.rng.gamma(k, theta)
            return mean
        
        elif self.distribution == ServiceDistribution.LOGNORMAL:
            # Lognormal: E = exp(μ + σ²/2), Var = (exp(σ²)-1)*exp(2μ+σ²)
            if var > 0 and mean > 0:
                sigma_sq = np.log(1 + var / (mean ** 2))
                mu_ln = np.log(mean) - sigma_sq / 2
                return self.rng.lognormal(mu_ln, np.sqrt(sigma_sq))
            return mean
        
        else:
            return self.rng.exponential(mean)
    
    def simulate(
        self,
        n_customers: int = 1000,
        max_time: Optional[float] = None
    ) -> SimulationResults:
        """
        Simule le système M/G/c.
        
        Args:
            n_customers: Nombre de clients à simuler
            max_time: Temps maximum de simulation
            
        Returns:
            SimulationResults avec les traces temporelles
        """
        # Initialisation
        arrival_times = []
        service_start_times = []
        departure_times = []
        waiting_times = []
        system_times = []
        service_times_list = []
        
        # État des serveurs
        server_end_times = [0.0] * self.c
        
        current_time = 0.0
        n_served = 0
        
        # Traces temporelles
        time_trace = [0.0]
        queue_length_trace = [0]
        system_size_trace = [0]
        
        for i in range(n_customers):
            # Génération du temps inter-arrivée (exponentiel)
            inter_arrival = self.rng.exponential(1.0 / self.lambda_rate)
            current_time += inter_arrival
            
            if max_time and current_time > max_time:
                break
            
            arrival_times.append(current_time)
            
            # Temps de service selon la distribution
            service_time = self._generate_service_time()
            service_times_list.append(service_time)
            
            # Trouver le serveur qui se libère le plus tôt
            earliest_server = min(range(self.c), key=lambda s: server_end_times[s])
            earliest_available = server_end_times[earliest_server]
            
            # Début du service
            service_start = max(current_time, earliest_available)
            service_start_times.append(service_start)
            
            # Temps d'attente
            wait_time = service_start - current_time
            waiting_times.append(wait_time)
            
            # Fin du service
            departure = service_start + service_time
            departure_times.append(departure)
            server_end_times[earliest_server] = departure
            
            # Temps dans le système
            system_time = departure - current_time
            system_times.append(system_time)
            
            n_served += 1
            
            # Mise à jour des traces
            in_queue = sum(1 for st in service_start_times if st > current_time)
            in_service = sum(1 for j, dep in enumerate(departure_times) 
                           if service_start_times[j] <= current_time < dep)
            
            time_trace.append(current_time)
            queue_length_trace.append(in_queue)
            system_size_trace.append(in_queue + in_service)
        
        return SimulationResults(
            arrival_times=np.array(arrival_times),
            service_start_times=np.array(service_start_times),
            departure_times=np.array(departure_times),
            service_times=np.array(service_times_list),
            waiting_times=np.array(waiting_times),
            system_times=np.array(system_times),
            n_arrivals=len(arrival_times),
            n_served=n_served,
            n_rejected=0,
            time_trace=np.array(time_trace),
            queue_length_trace=np.array(queue_length_trace),
            system_size_trace=np.array(system_size_trace)
        )
    
    def _get_kendall_notation(self) -> str:
        """Retourne la notation de Kendall du modèle."""
        return f"M/G/{self.c}"
    
    def _get_model_description(self) -> str:
        """Retourne une description du modèle."""
        return f"""
M/G/{self.c} - File d'attente multi-serveur avec service general

Parametres:
- Taux d'arrivee (lambda): {self.lambda_rate} clients/unite de temps
- Temps moyen de service: {self.service_mean:.4f}
- Variance du service: {self.service_variance:.4f}
- CV2 (coefficient de variation2): {self.cv_squared:.4f}
- Nombre de serveurs (c): {self.c}
- Distribution de service: {self.distribution.value}

Caracteristiques:
- Arrivees: Processus de Poisson
- Service: Distribution generale
- Capacite: Infinie
- Discipline: FIFO

Facteur d'utilisation: rho = lambda*E[S]/c = {self.rho:.4f}
"""
