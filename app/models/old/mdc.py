"""
Modèle de file d'attente M/D/c.

M/D/c = Arrivées Markoviennes / Service Déterministe / c serveurs

Ce modèle représente une file d'attente avec:
- Arrivées selon un processus de Poisson (M)
- Temps de service constants/déterministes (D)
- c serveurs en parallèle
- Capacité infinie

Auteurs: ERO2 Team Markov Moulinette Configurators - EPITA
"""

import numpy as np
import math
from typing import Optional
from dataclasses import dataclass

from ..base_queue import BaseQueueModel, QueueMetrics, SimulationResults


class MDcQueue(BaseQueueModel):
    """
    File d'attente M/D/c (Multi-serveur avec service déterministe).
    
    Approximation basée sur la formule de Pollaczek-Khinchin généralisée
    pour les systèmes multi-serveurs avec variance de service nulle.
    
    Paramètres:
        lambda_rate: Taux d'arrivée (λ)
        mu_rate: Taux de service par serveur (μ)
        c: Nombre de serveurs
    """
    
    def __init__(
        self,
        lambda_rate: float,
        mu_rate: float,
        c: int = 1,
        seed: Optional[int] = None,
        allow_unstable: bool = True
    ):
        """
        Initialise le modèle M/D/c.
        
        Args:
            lambda_rate: Taux d'arrivée des clients
            mu_rate: Taux de service par serveur
            c: Nombre de serveurs
            seed: Graine pour le générateur aléatoire
            allow_unstable: Si True, permet les systèmes instables
        """
        super().__init__(lambda_rate, mu_rate, c=c, K=None, seed=seed)
        self.allow_unstable = allow_unstable
        
        # Vérification de stabilité
        rho = lambda_rate / (c * mu_rate)
        if rho >= 1 and not allow_unstable:
            raise ValueError(
                f"Système instable: ρ = {rho:.4f} ≥ 1. "
                f"Augmentez c ou μ, ou diminuez λ."
            )
    
    @property
    def service_time(self) -> float:
        """Temps de service constant (déterministe)."""
        return 1.0 / self.mu_rate
    
    def compute_erlang_c(self) -> float:
        """
        Calcule la probabilité d'attente (Erlang-C).
        
        C(c, a) = P(attente > 0)
        """
        a = self.lambda_rate / self.mu_rate  # Intensité de trafic
        c = self.c
        
        # Calcul de P0
        sum_terms = sum((a ** n) / math.factorial(n) for n in range(c))
        last_term = (a ** c) / (math.factorial(c) * (1 - self.rho))
        
        P0 = 1.0 / (sum_terms + last_term)
        
        # Erlang-C
        C = ((a ** c) / math.factorial(c)) * (1 / (1 - self.rho)) * P0
        
        return C
    
    def compute_theoretical_metrics(self) -> QueueMetrics:
        """
        Calcule les métriques théoriques pour M/D/c.
        
        Utilise une approximation basée sur M/M/c avec correction
        pour la variance nulle du service (CV² = 0).
        
        Pour M/D/c: Wq ≈ Wq(M/M/c) * (1 + CV²) / 2 avec CV² = 0
        Donc Wq(M/D/c) ≈ Wq(M/M/c) / 2
        """
        rho = self.rho
        c = self.c
        mu = self.mu_rate
        lambda_rate = self.lambda_rate
        
        # Calcul Erlang-C pour M/M/c
        C = self.compute_erlang_c()
        
        # Wq pour M/M/c
        Wq_mmc = C / (c * mu * (1 - rho))
        
        # Correction pour service déterministe (CV² = 0)
        # Wq(M/D/c) ≈ Wq(M/M/c) * (1 + 0) / 2 = Wq(M/M/c) / 2
        Wq = Wq_mmc / 2
        
        # Autres métriques
        Ws = 1.0 / mu
        W = Wq + Ws
        
        Lq = lambda_rate * Wq
        Ls = lambda_rate * Ws
        L = Lq + Ls
        
        # P0 (probabilité système vide)
        a = lambda_rate / mu
        sum_terms = sum((a ** n) / math.factorial(n) for n in range(c))
        last_term = (a ** c) / (math.factorial(c) * (1 - rho))
        P0 = 1.0 / (sum_terms + last_term)
        
        return QueueMetrics(
            rho=rho,
            L=L,
            Lq=Lq,
            Ls=Ls,
            W=W,
            Wq=Wq,
            Ws=Ws,
            P0=P0,
            Pk=0.0,  # Pas de blocage (capacité infinie)
            lambda_eff=lambda_rate,
            throughput=lambda_rate
        )
    
    def simulate(
        self,
        n_customers: int = 1000,
        max_time: Optional[float] = None
    ) -> SimulationResults:
        """
        Simule le système M/D/c.
        
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
        
        # État des serveurs (temps de fin de service pour chaque serveur)
        server_end_times = [0.0] * self.c
        
        current_time = 0.0
        n_served = 0
        
        # Traces temporelles
        time_trace = [0.0]
        queue_length_trace = [0]
        system_size_trace = [0]
        
        queue = []  # File d'attente
        
        for i in range(n_customers):
            # Génération du temps inter-arrivée (exponentiel)
            inter_arrival = self.rng.exponential(1.0 / self.lambda_rate)
            current_time += inter_arrival
            
            if max_time and current_time > max_time:
                break
            
            arrival_times.append(current_time)
            
            # Temps de service déterministe
            service_time = self.service_time
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
            # Calcul de la taille de la file à ce moment
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
        return f"M/D/{self.c}"
    
    def _get_model_description(self) -> str:
        """Retourne une description du modèle."""
        return f"""
M/D/{self.c} - File d'attente multi-serveur avec service deterministe

Parametres:
- Taux d'arrivee (lambda): {self.lambda_rate} clients/unite de temps
- Taux de service (mu): {self.mu_rate} clients/unite de temps/serveur
- Nombre de serveurs (c): {self.c}
- Temps de service: {self.service_time:.4f} (constant)

Caracteristiques:
- Arrivees: Processus de Poisson
- Service: Temps constant (variance = 0)
- Capacite: Infinie
- Discipline: FIFO

Facteur d'utilisation: rho = lambda/(c*mu) = {self.rho:.4f}
"""
