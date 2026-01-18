"""
Module de base pour les modèles de files d'attente - VERSION FINALE COMPLÈTE

Corrections apportées:
1. Génération correcte des temps de service (déterministe pour M/D/*, général pour M/G/*)
2. Simulation multi-serveurs correcte pour M/M/c, M/D/c, M/G/c
3. Génération correcte de time_trace et queue_length_trace
4. Support complet de la capacité K (blocage)
5. Ajout de la propriété blocking_probability
6. Méthode receive_external_arrivals pour la chaîne de queues
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
import math
import heapq


@dataclass
class QueueMetrics:
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
        if self.n_arrivals == 0:
            return 0.0
        return self.n_rejected / self.n_arrivals


class BaseQueueModel(ABC):
    def __init__(self, lambda_rate: float, mu_rate: float, c: int = 1, K: Optional[int] = None, seed: Optional[int] = None):
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

    def get_theoretical_metrics(self) -> QueueMetrics:
        if self._theoretical_metrics is None:
            self._theoretical_metrics = self.compute_theoretical_metrics()
        return self._theoretical_metrics

    def _generate_interarrival_times(self, n: int) -> np.ndarray:
        return self.rng.exponential(scale=1/self.lambda_rate, size=n)

    def _generate_service_times(self, n: int) -> np.ndarray:
        return self.rng.exponential(scale=1/self.mu_rate, size=n)

    def compute_empirical_metrics(self, results: SimulationResults) -> QueueMetrics:
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
        rho = self.lambda_rate / self.mu_rate
        sum_terms = sum((rho ** n) / math.factorial(n) for n in range(self.c))
        last_term = (rho ** self.c) / math.factorial(self.c)
        P0 = 1 / (sum_terms + last_term)
        return P0

    def compute_erlang_c(self) -> float:
        rho = self.rho
        if rho >= 1:
            return 1.0
        a = self.lambda_rate / self.mu_rate
        P0 = self.compute_erlang_b()
        C = ((a ** self.c) / math.factorial(self.c)) * (1 / (1 - rho)) * P0
        return min(C, 1.0)


class GenericQueue(BaseQueueModel):
    def __init__(self, lambda_rate: float, mu_rate: float, kendall_notation: str,
                 c: int = 1, K: Optional[int] = None, seed: Optional[int] = None,
                 allow_unstable: bool = True, next_queue: Optional["GenericQueue"] = None, delay_to_next: float = 0.0):
        super().__init__(lambda_rate, mu_rate, c=c, K=K, seed=seed)
        self._kendall_notation = kendall_notation
        self.allow_unstable = allow_unstable
        self.next_queue = next_queue
        self.delay_to_next = delay_to_next
        if not self.is_stable and not allow_unstable:
            raise ValueError("Le système est instable (ρ >= 1) et allow_unstable est False.")
        self.service_variance = (1.0 / mu_rate) ** 2

    @property
    def kendall_notation(self) -> str:
        return self._kendall_notation

    def _get_model_description(self) -> str:
        return f"File d'attente basée sur la notation {self.kendall_notation}."

    def connect_to_next_queue(self, next_queue: "GenericQueue", delay: float = 0.0):
        self.next_queue = next_queue
        self.delay_to_next = delay

    @property
    def C_squared(self) -> float:
        return self.service_variance / (1 / self.mu_rate) ** 2 if self.service_variance is not None else 1.0

    def _generate_service_times_for_model(self, n: int) -> np.ndarray:
        if "M/D/" in self.kendall_notation:
            return np.full(n, 1.0 / self.mu_rate)
        elif "M/G/" in self.kendall_notation:
            mean = 1.0 / self.mu_rate
            cv_squared = self.C_squared
            if cv_squared > 0:
                shape = 1.0 / cv_squared
                scale = mean * cv_squared
                return self.rng.gamma(shape, scale, size=n)
            else:
                return np.full(n, mean)
        else:
            return self.rng.exponential(scale=1/self.mu_rate, size=n)

    def receive_external_arrivals(self, arrival_times: List[float]):
        if not hasattr(self, "_external_arrivals"):
            self._external_arrivals = []
        self._external_arrivals.extend(arrival_times)

    def _is_model(self, base_notation: str) -> bool:
        """Vérifie si la notation correspond à un modèle (ex: M/M/c -> M/M/1, M/M/2, etc.)"""
        import re
        # Pour M/M/c, M/D/c, M/G/c - le 'c' peut être un nombre
        if base_notation.endswith("/c"):
            prefix = base_notation[:-1]  # "M/M/" ou "M/D/" ou "M/G/"
            return self.kendall_notation.startswith(prefix) and self.c > 1
        elif base_notation.endswith("/1"):
            prefix = base_notation[:-1]  # "M/M/" ou "M/D/" ou "M/G/"
            return self.kendall_notation.startswith(prefix) and self.c == 1
        else:
            return base_notation in self.kendall_notation
    
    def compute_theoretical_metrics(self) -> QueueMetrics:
        # Détection du modèle basée sur la notation et le nombre de serveurs
        is_mm1 = self._is_model("M/M/1") or (self.kendall_notation.startswith("M/M/") and self.c == 1)
        is_md1 = self._is_model("M/D/1") or (self.kendall_notation.startswith("M/D/") and self.c == 1)
        is_mg1 = self._is_model("M/G/1") or (self.kendall_notation.startswith("M/G/") and self.c == 1)
        is_mmc = self.kendall_notation.startswith("M/M/") and self.c > 1
        is_mdc = self.kendall_notation.startswith("M/D/") and self.c > 1
        is_mgc = self.kendall_notation.startswith("M/G/") and self.c > 1
        
        if is_mm1:
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
        elif is_md1:
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
        elif is_mdc:
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
        elif is_mmc:
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
        elif is_mg1:
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
        elif is_mgc:
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
        max_time: Optional[float] = None,
        external_arrival_times: Optional[np.ndarray] = None
    ) -> SimulationResults:

        if external_arrival_times is not None:
            arrival_times = np.array(external_arrival_times, dtype=float)
            n_customers = len(arrival_times)
            service_times = self._generate_service_times_for_model(n_customers)
        else:
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

    def simulate_step(
        self,
        current_time: float,
        max_time: Optional[float] = None,
        lambda_rate_func: Optional[callable] = None,
        event_heap=None,
        queue_state=None,
        busy_servers=0,
        service_start_times=None,
        departure_times=None,
        accepted_arrivals=None,
        accepted_service_times=None,
        waiting_times_list=None,
        system_times_list=None,
        n_rejected=0,
    ):
        """
        Simule un seul événement pour cette queue.
        Compatible avec ChainQueue.simulate_step_chain.
        """
        if event_heap is None:
            event_heap = []
        if queue_state is None:
            queue_state = []
        if service_start_times is None:
            service_start_times = {}
        if departure_times is None:
            departure_times = {}
        if accepted_arrivals is None:
            accepted_arrivals = []
        if accepted_service_times is None:
            accepted_service_times = []
        if waiting_times_list is None:
            waiting_times_list = []
        if system_times_list is None:
            system_times_list = []

        c = self.c
        K = self.K

        if lambda_rate_func is not None and len(event_heap) == 0:
            if callable(lambda_rate_func):
                lam = lambda_rate_func(current_time)
            else:
                lam = float(lambda_rate_func)
            next_arrival = current_time + self.rng.exponential(1 / lam)
            heapq.heappush(event_heap, (next_arrival, "arrival", len(accepted_arrivals)))

        if not event_heap:
            return None  # plus d’événements

        # Traiter le prochain événement
        time, etype, idx = heapq.heappop(event_heap)

        if max_time is not None and time > max_time:
            return None

        system_size = len(queue_state) + busy_servers

        if etype == "arrival":
            if K is not None and system_size >= K:
                n_rejected += 1
            else:
                accepted_arrivals.append(time)
                service_time = self._generate_service_times_for_model(1)[0]
                accepted_service_times.append(service_time)

                if busy_servers < c:
                    busy_servers += 1
                    service_start_times[idx] = time
                    depart_time = time + service_time
                    departure_times[idx] = depart_time
                    waiting_times_list.append(0.0)
                    system_times_list.append(service_time)
                    heapq.heappush(event_heap, (depart_time, "departure", idx))
                else:
                    queue_state.append(idx)

        else:  # départ
            busy_servers -= 1
            if queue_state:
                j = queue_state.pop(0)
                busy_servers += 1
                service_start = time
                service_time = accepted_service_times[j]
                service_start_times[j] = service_start
                depart_time = service_start + service_time
                departure_times[j] = depart_time
                waiting_times_list.append(service_start - accepted_arrivals[j])
                system_times_list.append(depart_time - accepted_arrivals[j])
                heapq.heappush(event_heap, (depart_time, "departure", j))

        return {
            "event_heap": event_heap,
            "queue_state": queue_state,
            "busy_servers": busy_servers,
            "service_start_times": service_start_times,
            "departure_times": departure_times,
            "accepted_arrivals": accepted_arrivals,
            "accepted_service_times": accepted_service_times,
            "waiting_times_list": waiting_times_list,
            "system_times_list": system_times_list,
            "n_rejected": n_rejected,
            "current_time": time
        }


class ChainQueue:
    """Chaîne de files d'attente où les clients passent d'une file à l'autre."""

    def __init__(self, queues: List['GenericQueue']):
        self.queues = queues

    def simulate_step(
        self,
        current_time: float,
        max_time: Optional[float] = None,
        lambda_rate_func: Optional[callable] = None,
        event_heap=None,
        queue_state=None,
        busy_servers=0,
        service_start_times=None,
        departure_times=None,
        accepted_arrivals=None,
        accepted_service_times=None,
        waiting_times_list=None,
        system_times_list=None,
        n_rejected=0,
        c=None,
        K=None
    ):
        """
        Traite le prochain événement dans une seule queue.
        Permet d'utiliser un lambda_rate évolutif.
        """
        # Si c/K sont passés explicitement, on les utilise, sinon on prend ceux de la queue
        if c is None:
            c = getattr(self, 'c', None)
        if K is None:
            K = getattr(self, 'K', None)

        # Initialisation des structures
        if event_heap is None:
            event_heap = []
        if queue_state is None:
            queue_state = []
        if service_start_times is None:
            service_start_times = {}
        if departure_times is None:
            departure_times = {}
        if accepted_arrivals is None:
            accepted_arrivals = []
        if accepted_service_times is None:
            accepted_service_times = []
        if waiting_times_list is None:
            waiting_times_list = []
        if system_times_list is None:
            system_times_list = []

        # Génération d’une arrivée si aucune future et lambda fourni
        if lambda_rate_func is not None and len(event_heap) == 0:
            lam = lambda_rate_func(current_time)
            next_arrival = current_time + np.random.exponential(1 / lam)
            heapq.heappush(event_heap, (next_arrival, "arrival", len(accepted_arrivals)))

        if not event_heap:
            return None  # plus d’événements

        # Traite le prochain événement
        time, etype, idx = heapq.heappop(event_heap)

        if max_time is not None and time > max_time:
            return None

        system_size = len(queue_state) + busy_servers

        if etype == "arrival":
            if K is not None and system_size >= K:
                n_rejected += 1
            else:
                accepted_arrivals.append(time)
                service_time = self._generate_service_times_for_model(1)[0]
                accepted_service_times.append(service_time)

                if busy_servers < c:
                    busy_servers += 1
                    service_start_times[idx] = time
                    depart_time = time + service_time
                    departure_times[idx] = depart_time
                    waiting_times_list.append(0.0)
                    system_times_list.append(service_time)
                    heapq.heappush(event_heap, (depart_time, "departure", idx))
                else:
                    queue_state.append(idx)

        else:  # départ
            busy_servers -= 1
            if queue_state:
                j = queue_state.pop(0)
                busy_servers += 1
                service_start = time
                service_time = accepted_service_times[j]
                service_start_times[j] = service_start
                depart_time = service_start + service_time
                departure_times[j] = depart_time
                waiting_times_list.append(service_start - accepted_arrivals[j])
                system_times_list.append(depart_time - accepted_arrivals[j])
                heapq.heappush(event_heap, (depart_time, "departure", j))

        return {
            "event_heap": event_heap,
            "queue_state": queue_state,
            "busy_servers": busy_servers,
            "service_start_times": service_start_times,
            "departure_times": departure_times,
            "accepted_arrivals": accepted_arrivals,
            "accepted_service_times": accepted_service_times,
            "waiting_times_list": waiting_times_list,
            "system_times_list": system_times_list,
            "n_rejected": n_rejected,
            "current_time": time
        }

    def simulate_step_chain(
        self,
        current_time: float,
        lambda_rate_func: Optional[callable] = None,
        max_time: Optional[float] = None
    ):
        """
        Simule un pas pour toute la chaîne:
        - La première queue reçoit les arrivées selon lambda_rate_func
        - Les clients qui terminent une queue passent à la suivante
        """
        all_results = []

        for i, queue in enumerate(self.queues):
            # Appliquer lambda_rate uniquement sur la première queue
            lam = lambda_rate_func if i == 0 else None

            res_dict = queue.simulate_step(
                current_time=current_time,
                max_time=max_time,
                lambda_rate_func=lam
            )

            if res_dict is None:
                continue

            all_results.append(res_dict)

            # Clients qui quittent cette queue → arrivées pour la suivante
            incoming_arrivals = list(res_dict["departure_times"].values())
            if i + 1 < len(self.queues) and incoming_arrivals:
                next_queue = self.queues[i + 1]
                if hasattr(next_queue, "receive_external_arrivals"):
                    next_queue.receive_external_arrivals(incoming_arrivals)

        return all_results

    def simulate_chain(
        self,
        n_customers: int = 1000,
        max_time: Optional[float] = None,
        lambda_rate: Optional[float] = None
    ) -> List['SimulationResults']:
        results = []
        current_arrivals = None

        for i, queue in enumerate(self.queues):
            if current_arrivals is not None:
                if getattr(queue, "delay_to_next", 0) != 0:
                    current_arrivals = current_arrivals + queue.delay_to_next
                sim_result = queue.simulate(
                    external_arrival_times=current_arrivals,
                    max_time=max_time
                )
            else:
                if lambda_rate is not None:
                    sim_result = queue.simulate(n_customers=n_customers, lambda_rate=lambda_rate, max_time=max_time)
                else:
                    sim_result = queue.simulate(n_customers=n_customers, max_time=max_time)

            results.append(sim_result)
            current_arrivals = sim_result.departure_times

            if len(current_arrivals) == 0:
                break

        return results

    def simulate(self, n_customers=1000, max_time=None, lambda_rate=None):
        results = self.simulate_chain(n_customers=n_customers, max_time=max_time, lambda_rate=lambda_rate)

        if len(results) == 0:
            raise RuntimeError("Aucune file dans la chaîne")

        first = results[0]
        last = results[-1]

        global_result = {
            "n_arrivals": first.n_arrivals,
            "n_served": last.n_served,
            "n_rejected": sum(r.n_rejected for r in results)
        }

        if len(first.arrival_times) > 0 and len(last.departure_times) > 0:
            m = min(len(first.arrival_times), len(last.departure_times))
            global_result["system_times"] = last.departure_times[:m] - first.arrival_times[:m]
            global_result["waiting_times"] = np.concatenate(
                [r.waiting_times for r in results if len(r.waiting_times) > 0]
            ) if any(len(r.waiting_times) > 0 for r in results) else np.array([])
        else:
            global_result["system_times"] = np.array([])
            global_result["waiting_times"] = np.array([])

        global_result["time_trace"] = np.concatenate(
            [r.time_trace for r in results if len(r.time_trace) > 0]
        ) if any(len(r.time_trace) > 0 for r in results) else np.array([])

        global_result["queue_length_trace"] = np.concatenate(
            [r.queue_length_trace for r in results if len(r.queue_length_trace) > 0]
        ) if any(len(r.queue_length_trace) > 0 for r in results) else np.array([])

        return global_result, results

    def update_arrival_rate(self, lambda_rate: float):
        for queue in self.queues:
            if hasattr(queue, 'lambda_rate'):
                queue.lambda_rate = lambda_rate

    def update_servers(self, n_servers: int):
        for queue in self.queues:
            if hasattr(queue, 'c'):
                queue.c = n_servers

    def compute_theoretical_metrics(self) -> 'QueueMetrics':
        metrics_list = [q.compute_theoretical_metrics() for q in self.queues]
        return metrics_list[-1]

    def total_servers(self) -> int:
        return sum(getattr(q, 'c', 1) for q in self.queues)

    def get_kendall_representation(self) -> str:
        return " -> ".join(getattr(queue, "kendall_notation", "?") for queue in self.queues)