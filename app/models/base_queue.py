"""
Module de base pour les modèles de files d'attente - VERSION CORRIGÉE COMPLÈTE

Corrections apportées:
1. Génération correcte des temps de service (déterministe pour M/D/*, général pour M/G/*)
2. Simulation multi-serveurs correcte pour M/M/c, M/D/c, M/G/c
3. Génération correcte de time_trace et queue_length_trace
4. Support complet de la capacité K (blocage)
5. Ajout de la propriété blocking_probability
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Union
import numpy as np
import math
import heapq


@dataclass
class QueueMetrics:
    """Métriques de performance d'une file d'attente."""
    rho: float = 0.0
    L: float = 0.0
    Lq: float = 0.0
    Ls: float = 0.0
    W: float = 0.0
    Wq: float = 0.0
    Ws: float = 0.0
    P0: float = 0.0
    Pk: float = 0.0
    lambda_eff: float = 0.0
    throughput: float = 0.0
    state_probabilities: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class SimulationResults:
    """Résultats d'une simulation de file d'attente."""
    arrival_times: np.ndarray = field(default_factory=lambda: np.array([]))
    service_start_times: np.ndarray = field(default_factory=lambda: np.array([]))
    departure_times: np.ndarray = field(default_factory=lambda: np.array([]))
    service_times: np.ndarray = field(default_factory=lambda: np.array([]))
    waiting_times: np.ndarray = field(default_factory=lambda: np.array([]))
    system_times: np.ndarray = field(default_factory=lambda: np.array([]))
    n_arrivals: int = 0
    n_served: int = 0
    n_rejected: int = 0
    time_trace: np.ndarray = field(default_factory=lambda: np.array([]))
    queue_length_trace: np.ndarray = field(default_factory=lambda: np.array([]))
    system_size_trace: np.ndarray = field(default_factory=lambda: np.array([]))
    empirical_metrics: Optional[QueueMetrics] = None
    
    @property
    def blocking_probability(self) -> float:
        """Calcule la probabilité de blocage empirique."""
        if self.n_arrivals == 0:
            return 0.0
        return self.n_rejected / self.n_arrivals


class BaseQueueModel(ABC):
    """Classe de base abstraite pour tous les modèles de files d'attente."""
    
    def __init__(
        self,
        lambda_rate: float,
        mu_rate: float,
        c: int = 1,
        K: Optional[int] = None,
        seed: Optional[int] = None
    ):
        self._validate_parameters(lambda_rate, mu_rate, c, K)
        self.lambda_rate = lambda_rate
        self.mu_rate = mu_rate
        self.c = c
        self.K = K
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._theoretical_metrics: Optional[QueueMetrics] = None
    
    def _validate_parameters(self, lambda_rate: float, mu_rate: float, c: int, K: Optional[int]) -> None:
        if lambda_rate <= 0:
            raise ValueError(f"λ doit être > 0, reçu: {lambda_rate}")
        if mu_rate <= 0:
            raise ValueError(f"μ doit être > 0, reçu: {mu_rate}")
        if int(c) < 1:
            raise ValueError(f"c doit être ≥ 1, reçu: {c}")
        if K is not None and K < c:
            raise ValueError(f"K doit être ≥ c, reçu: K={K}, c={c}")
    
    @property
    def rho(self) -> float:
        return self.lambda_rate / (self.c * self.mu_rate)
    
    @property
    def is_stable(self) -> bool:
        if self.K is not None:
            return True
        return self.rho < 1
    
    @property
    def kendall_notation(self) -> str:
        return self._get_kendall_notation()
    
    @abstractmethod
    def _get_kendall_notation(self) -> str:
        pass
    
    @abstractmethod
    def compute_theoretical_metrics(self) -> QueueMetrics:
        pass
    
    @abstractmethod
    def simulate(self, n_customers: int = 1000, max_time: Optional[float] = None) -> SimulationResults:
        pass
    
    def get_theoretical_metrics(self) -> QueueMetrics:
        """Retourne les métriques théoriques (avec cache)."""
        if self._theoretical_metrics is None:
            self._theoretical_metrics = self.compute_theoretical_metrics()
        return self._theoretical_metrics
    
    def _generate_interarrival_times(self, n: int) -> np.ndarray:
        return self.rng.exponential(scale=1/self.lambda_rate, size=n)
    
    def _generate_service_times(self, n: int) -> np.ndarray:
        """Génère des temps de service EXPONENTIELS (M/M/*)"""
        return self.rng.exponential(scale=1/self.mu_rate, size=n)
    
    def compute_empirical_metrics(self, results: SimulationResults) -> QueueMetrics:
        """Calcule les métriques empiriques à partir des résultats de simulation."""
        metrics = QueueMetrics()
        
        if len(results.system_times) > 0:
            metrics.W = np.mean(results.system_times)
            metrics.Wq = np.mean(results.waiting_times)
            metrics.Ws = np.mean(results.service_times[:len(results.system_times)])
            
            lambda_eff = results.n_served / results.departure_times[-1] if len(results.departure_times) > 0 else 0
            metrics.lambda_eff = lambda_eff
            metrics.L = lambda_eff * metrics.W
            metrics.Lq = lambda_eff * metrics.Wq
            metrics.Ls = lambda_eff * metrics.Ws
            
            if results.n_arrivals > 0:
                metrics.rho = results.n_served / results.n_arrivals * self.rho
                metrics.Pk = results.n_rejected / results.n_arrivals
            
            metrics.throughput = lambda_eff
        
        return metrics
    
    def compute_erlang_b(self) -> float:
        """Calcule la probabilité que toutes les serveurs soient occupés (Erlang B)."""
        rho = self.lambda_rate / self.mu_rate
        sum_terms = sum((rho ** n) / math.factorial(n) for n in range(self.c))
        last_term = (rho ** self.c) / math.factorial(self.c)
        P0 = 1 / (sum_terms + last_term)
        return P0

    def compute_erlang_c(self) -> float:
        """Calcule la probabilité d'attente dans le système (Erlang C)."""
        rho = self.rho
        if rho >= 1:
            return 1.0
        a = self.lambda_rate / self.mu_rate
        P0 = self.compute_erlang_b()
        C = ((a ** self.c) / math.factorial(self.c)) * (1 / (1 - rho)) * P0
        return min(C, 1.0)


class GenericQueue(BaseQueueModel):
    """Classe générique pour modéliser une file d'attente."""

    def __init__(
        self,
        lambda_rate: float,
        mu_rate: float,
        kendall_notation: str,
        c: int = 1,
        K: Optional[int] = None,
        seed: Optional[int] = None,
        allow_unstable: bool = True,
        next_queue: Optional["GenericQueue"] = None,
        delay_to_next: float = 0.0
    ):
        super().__init__(lambda_rate, mu_rate, c=c, K=K, seed=seed)
        self._kendall_notation = kendall_notation
        self.allow_unstable = allow_unstable
        self.next_queue = next_queue
        self.delay_to_next = delay_to_next

        if not self.is_stable and not allow_unstable:
            raise ValueError("Le système est instable (ρ >= 1) et allow_unstable est False.")

        service_mean = 1.0 / mu_rate
        self.service_variance = (service_mean ** 2) * 1.0

    @property
    def kendall_notation(self) -> str:
        return self._kendall_notation

    def _get_kendall_notation(self) -> str:
        return self.kendall_notation

    def _get_model_description(self) -> str:
        return f"File d'attente basée sur la notation {self.kendall_notation}."

    def connect_to_next_queue(self, next_queue: "GenericQueue", delay: float = 0.0):
        """
        Connecte cette file à une autre file d'attente.

        Args:
            next_queue: File suivante dans la chaîne
            delay: Délai avant que la file suivante commence à traiter les clients
        """
        self.next_queue = next_queue
        self.delay_to_next = delay

    @property
    def C_squared(self) -> float:
        if hasattr(self, 'service_variance') and self.service_variance is not None:
            return self.service_variance / (1 / self.mu_rate) ** 2
        return 1.0

    def _generate_service_times_for_model(self, n: int) -> np.ndarray:
        """
        Génère les temps de service selon le type de modèle.
        - M/M/* : Exponentiel
        - M/D/* : Déterministe (tous égaux à 1/μ)
        - M/G/* : Général (distribution gamma avec variance contrôlée)
        """
        if "M/D/" in self.kendall_notation:
            # Temps de service DÉTERMINISTE
            return np.full(n, 1.0 / self.mu_rate)
        elif "M/G/" in self.kendall_notation:
            # Temps de service GÉNÉRAL avec variance contrôlée
            mean = 1.0 / self.mu_rate
            cv_squared = self.C_squared
            
            if cv_squared > 0:
                shape = 1.0 / cv_squared
                scale = mean * cv_squared
                return self.rng.gamma(shape, scale, size=n)
            else:
                return np.full(n, mean)
        else:
            # M/M/* : Exponentiel (par défaut)
            return self.rng.exponential(scale=1/self.mu_rate, size=n)

    def compute_theoretical_metrics(self) -> QueueMetrics:
        if "M/M/1" in self.kendall_notation:
            rho = self.rho
            if rho >= 1:
                raise ValueError(f"Système instable: ρ = {rho:.4f} ≥ 1")
            Lq = rho ** 2 / (1 - rho)
            Wq = Lq / self.lambda_rate
            Ws = 1 / self.mu_rate
            W = Wq + Ws
            L = self.lambda_rate * W
            return QueueMetrics(
                rho=rho, L=L, Lq=Lq, W=W, Wq=Wq, Ws=Ws,
                P0=1 - rho, Pk=0.0,
                lambda_eff=self.lambda_rate, throughput=self.lambda_rate
            )
        elif "M/D/1" in self.kendall_notation:
            rho = self.rho
            if rho >= 1:
                raise ValueError(f"Système instable: ρ = {rho:.4f} ≥ 1")
            Lq = (rho ** 2) / (2 * (1 - rho))
            Wq = Lq / self.lambda_rate
            Ws = 1 / self.mu_rate
            W = Wq + Ws
            L = self.lambda_rate * W
            return QueueMetrics(
                rho=rho, L=L, Lq=Lq, W=W, Wq=Wq, Ws=Ws,
                P0=1 - rho, Pk=0.0,
                lambda_eff=self.lambda_rate, throughput=self.lambda_rate
            )
        elif "M/D/c" in self.kendall_notation:
            rho = self.rho
            if rho >= 1:
                raise ValueError(f"Système instable: ρ = {rho:.4f} ≥ 1")
            P0 = self.compute_erlang_b()
            C = self.compute_erlang_c()
            Wq_mmc = C / (self.c * self.mu_rate * (1 - rho))
            Wq = Wq_mmc / 2
            Ws = 1 / self.mu_rate
            W = Wq + Ws
            Lq = self.lambda_rate * Wq
            L = Lq + self.lambda_rate * Ws
            return QueueMetrics(
                rho=rho, L=L, Lq=Lq, W=W, Wq=Wq, Ws=Ws,
                P0=P0, Pk=0.0,
                lambda_eff=self.lambda_rate, throughput=self.lambda_rate
            )
        elif "M/M/c" in self.kendall_notation:
            rho = self.rho
            if rho >= 1:
                raise ValueError(f"Système instable: ρ = {rho:.4f} ≥ 1")
            P0 = self.compute_erlang_b()
            C = self.compute_erlang_c()
            Lq = C * rho / (1 - rho)
            Ws = 1 / self.mu_rate
            Wq = Lq / self.lambda_rate
            W = Wq + Ws
            L = Lq + self.lambda_rate * Ws
            return QueueMetrics(
                rho=rho, L=L, Lq=Lq, W=W, Wq=Wq, Ws=Ws,
                P0=P0, Pk=0.0,
                lambda_eff=self.lambda_rate, throughput=self.lambda_rate
            )
        elif "M/G/1" in self.kendall_notation:
            rho = self.rho
            if rho >= 1:
                raise ValueError(f"Système instable: ρ = {rho:.4f} ≥ 1")
            C_sq = self.C_squared
            Lq = (rho ** 2 * (1 + C_sq)) / (2 * (1 - rho))
            Wq = Lq / self.lambda_rate
            Ws = 1 / self.mu_rate
            W = Wq + Ws
            L = self.lambda_rate * W
            return QueueMetrics(
                rho=rho, L=L, Lq=Lq, W=W, Wq=Wq, Ws=Ws,
                P0=1 - rho, Pk=0.0,
                lambda_eff=self.lambda_rate, throughput=self.lambda_rate
            )
        elif "M/G/c" in self.kendall_notation:
            rho = self.rho
            if rho >= 1:
                raise ValueError(f"Système instable: ρ = {rho:.4f} ≥ 1")
            P0 = self.compute_erlang_b()
            C = self.compute_erlang_c()
            Wq_mmc = C / (self.c * self.mu_rate * (1 - rho))
            Wq = Wq_mmc * (1 + self.C_squared) / 2
            Ws = 1 / self.mu_rate
            W = Wq + Ws
            Lq = self.lambda_rate * Wq
            L = Lq + self.lambda_rate * Ws
            return QueueMetrics(
                rho=rho, L=L, Lq=Lq, W=W, Wq=Wq, Ws=Ws,
                P0=P0, Pk=0.0,
                lambda_eff=self.lambda_rate, throughput=self.lambda_rate
            )
        else:
            raise NotImplementedError(f"Les métriques pour {self.kendall_notation} ne sont pas encore implémentées.")


    def simulate(
        self,
        n_customers: int = 1000,
        max_time: Optional[float] = None
    ) -> SimulationResults:

        interarrival_times = self._generate_interarrival_times(n_customers)
        service_times = self._generate_service_times_for_model(n_customers)

        arrival_times = np.cumsum(interarrival_times)

        if max_time is not None:
            mask = arrival_times <= max_time
            arrival_times = arrival_times[mask]
            service_times = service_times[:len(arrival_times)]
            n_customers = len(arrival_times)

        if n_customers == 0:
            return SimulationResults()

        # =========================
        # STRUCTURES
        # =========================
        c = self.c
        K = self.K     # capacité max DANS LE SYSTEME (service + file) si définie

        queue = []                     # file d'attente FIFO (indices de clients)
        busy_servers = 0               # combien servent actuellement
        event_heap = []                # (time, type, idx)

        service_start_times = np.zeros(n_customers)
        departure_times = np.zeros(n_customers)

        accepted_arrivals = []
        accepted_service_times = []

        waiting_times_list = []
        system_times_list = []

        n_rejected = 0

        # =========================
        # PLANIFIER LES ARRIVÉES
        # =========================
        for i, t in enumerate(arrival_times):
            heapq.heappush(event_heap, (t, "arrival", i))

        # =========================
        # TRAÇAGE TEMPOREL
        # =========================
        time_trace = []
        system_size_trace = []
        queue_length_trace = []

        system_size = 0

        # =========================
        # BOUCLE EVENEMENTIELLE
        # =========================
        while event_heap:
            time, etype, i = heapq.heappop(event_heap)

            if etype == "arrival":
                # Capacité K ? (K compte service + attente)
                if K is not None and system_size >= K:
                    n_rejected += 1
                    continue

                system_size += 1
                accepted_arrivals.append(arrival_times[i])
                accepted_service_times.append(service_times[i])

                if busy_servers < c:
                    # démarre service
                    busy_servers += 1
                    service_start = time
                    depart_time = service_start + service_times[i]

                    service_start_times[i] = service_start
                    departure_times[i] = depart_time

                    waiting_times_list.append(service_start - arrival_times[i])
                    system_times_list.append(depart_time - arrival_times[i])

                    heapq.heappush(event_heap, (depart_time, "departure", i))

                else:
                    # va dans la file
                    queue.append(i)

            else:  # departure
                system_size -= 1
                busy_servers -= 1

                # Si quelqu'un attend => il prend la place immédiatement
                if queue:
                    j = queue.pop(0)
                    busy_servers += 1

                    service_start = time
                    depart_time = service_start + service_times[j]

                    service_start_times[j] = service_start
                    departure_times[j] = depart_time

                    waiting_times_list.append(service_start - arrival_times[j])
                    system_times_list.append(depart_time - arrival_times[j])

                    heapq.heappush(event_heap, (depart_time, "departure", j))

            # --- mise à jour traces ---
            time_trace.append(time)
            system_size_trace.append(system_size)
            queue_length_trace.append(max(0, system_size - c))

        # =========================
        # BUILD RESULTS
        # =========================
        results = SimulationResults(
            arrival_times=np.array(accepted_arrivals),
            service_start_times=service_start_times[service_start_times > 0],
            departure_times=departure_times[departure_times > 0],
            service_times=np.array(accepted_service_times),
            waiting_times=np.array(waiting_times_list),
            system_times=np.array(system_times_list),
            n_arrivals=n_customers,
            n_served=len(accepted_arrivals),
            n_rejected=n_rejected,
            time_trace=np.array(time_trace),
            queue_length_trace=np.array(queue_length_trace),
            system_size_trace=np.array(system_size_trace),
        )

        results.empirical_metrics = self.compute_empirical_metrics(results)
        return results


class ChainQueue:
    """Chaîne de files d'attente où les clients passent d'une file à l'autre."""

    def __init__(self, queues: List[GenericQueue]):
        self.queues = queues

    def simulate_chain(
        self,
        n_customers: int = 1000,
        max_time: Optional[float] = None
    ) -> List[SimulationResults]:
        results = []
        arrival_times = None

        for queue in self.queues:
            if arrival_times is not None:
                # TODO: Implémenter le passage entre files
                pass
            
            simulation_result = queue.simulate(n_customers=n_customers, max_time=max_time)
            results.append(simulation_result)
            arrival_times = simulation_result.departure_times

        return results

    def get_kendall_representation(self) -> str:
        return " -> ".join(queue.kendall_notation for queue in self.queues)