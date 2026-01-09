"""
Modèle de file d'attente M/M/c.

File d'attente avec:
- Arrivées selon un processus de Poisson (M)
- Temps de service exponentiels (M)
- c serveurs en parallèle
- Capacité infinie
- Discipline FIFO

Ce modèle est pertinent pour la moulinette car elle dispose
de plusieurs runners de correction fonctionnant en parallèle.

Formules théoriques (sous condition de stabilité ρ = λ/(cμ) < 1):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│ Métrique              │ Formule                                   │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│ Utilisation ρ         │ λ/(cμ)                                    │
│ a = λ/μ               │ Intensité de trafic                       │
│ P₀ (système vide)     │ [Σₙ₌₀^{c-1} aⁿ/n! + aᶜ/(c!(1-ρ))]⁻¹       │
│ C(c,a) Erlang C       │ (aᶜ/c!)·P₀ / (1-ρ)                        │
│ Lq (nb moyen file)    │ C(c,a)·ρ/(1-ρ)                            │
│ L (nb moyen système)  │ Lq + a                                    │
│ Wq (temps attente)    │ Lq/λ                                      │
│ W (temps système)     │ Wq + 1/μ                                  │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Application à la moulinette:
- λ = taux de soumission global (tags/heure)
- μ = capacité de traitement par runner (corrections/heure)
- c = nombre de runners disponibles

Auteurs: ERO2 Team Markov Moulinette Configurators - EPITA
"""

import numpy as np
import heapq
from math import factorial
from typing import Optional
from .base_queue import BaseQueueModel, QueueMetrics, SimulationResults


class MMcQueue(BaseQueueModel):
    """
    Implémentation du modèle M/M/c (Erlang C).
    
    Cas d'usage pour la moulinette:
    - Modélise le pool de runners de correction
    - Permet de déterminer le nombre optimal de serveurs
    - Base pour l'analyse d'auto-scaling
    
    Exemple:
        >>> queue = MMcQueue(lambda_rate=50, mu_rate=12, c=5)
        >>> metrics = queue.get_theoretical_metrics()
        >>> print(f"Probabilité d'attente: {metrics.C_c:.2%}")
    """
    
    def __init__(
        self,
        lambda_rate: float,
        mu_rate: float,
        c: int = 1,
        seed: Optional[int] = None
    ):
        """
        Initialise une file M/M/c.
        
        Args:
            lambda_rate: Taux d'arrivée λ (soumissions/heure)
            mu_rate: Taux de service μ par serveur (corrections/heure)
            c: Nombre de serveurs (runners)
            seed: Graine aléatoire
            
        Raises:
            ValueError: Si ρ = λ/(cμ) ≥ 1 (système instable)
        """
        super().__init__(lambda_rate, mu_rate, c=c, K=None, seed=seed)
        
        if not self.is_stable:
            raise ValueError(
                f"Système instable: ρ = {self.rho:.4f} ≥ 1. "
                f"Besoin de plus de serveurs ou d'augmenter μ. "
                f"c_min = ⌈λ/μ⌉ = {int(np.ceil(lambda_rate/mu_rate))}"
            )
        
        # Intensité de trafic a = λ/μ
        self.a = lambda_rate / mu_rate
    
    def _get_kendall_notation(self) -> str:
        return f"M/M/{self.c}"
    
    def _get_model_description(self) -> str:
        return f"""
        ╔══════════════════════════════════════════════════════════════════╗
        ║                        MODÈLE M/M/{self.c}                       ║
        ╠══════════════════════════════════════════════════════════════════╣
        ║ Description:                                                     ║
        ║   File d'attente avec {self.c} serveurs identiques en parallèle. ║
        ║                                                                  ║
        ║ Hypothèses:                                                      ║
        ║   • Arrivées: Processus de Poisson de taux λ                     ║
        ║   • Service: Temps exponentiels de taux μ (par serveur)          ║
        ║   • Serveurs: {self.c} serveurs identiques                       ║
        ║   • Capacité: Infinie                                            ║
        ║   • Discipline: FIFO                                             ║
        ║                                                                  ║
        ║ Condition de stabilité: ρ = λ/(cμ) < 1                           ║
        ║                                                                  ║
        ║ Application moulinette:                                          ║
        ║   Modélise le pool de runners de correction.                     ║
        ║   Les c serveurs traitent les soumissions en parallèle.          ║
        ║   Utile pour dimensionner l'infrastructure.                      ║
        ╚══════════════════════════════════════════════════════════════════╝
        """
    
    def compute_theoretical_metrics(self) -> QueueMetrics:
        """
        Calcule les métriques théoriques pour M/M/c.
        
        Utilise la formule d'Erlang C pour le calcul des probabilités
        d'attente et des temps moyens.
        
        Returns:
            QueueMetrics avec toutes les métriques calculées
        """
        lambda_rate = self.lambda_rate
        mu_rate = self.mu_rate
        c = self.c
        rho = self.rho
        a = self.a  # λ/μ
        
        # Calcul de P₀ (probabilité système vide)
        # P₀ = [Σₙ₌₀^{c-1} aⁿ/n! + aᶜ/(c!(1-ρ))]⁻¹
        sum_term = sum([(a ** n) / factorial(n) for n in range(c)])
        last_term = (a ** c) / (factorial(c) * (1 - rho))
        P0 = 1 / (sum_term + last_term)
        
        # Formule d'Erlang C: C(c, a) = P(attente > 0)
        # C(c, a) = (aᶜ/c!) · P₀ / (1 - ρ)
        erlang_c = ((a ** c) / factorial(c)) * P0 / (1 - rho)
        
        # Nombre moyen en file d'attente
        # Lq = C(c, a) · ρ / (1 - ρ)
        Lq = erlang_c * rho / (1 - rho)
        
        # Nombre moyen en service = a = λ/μ (par propriété PASTA)
        Ls = a
        
        # Nombre moyen total dans le système
        L = Lq + Ls
        
        # Temps moyens (via loi de Little)
        Wq = Lq / lambda_rate
        Ws = 1 / mu_rate
        W = Wq + Ws
        
        # Distribution stationnaire P(N = n)
        state_probs = self._compute_state_probabilities(P0, a, c, rho)
        
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
            throughput=lambda_rate,
            state_probabilities=state_probs
        )
    
    def _compute_state_probabilities(
        self,
        P0: float,
        a: float,
        c: int,
        rho: float,
        n_max: int = 100
    ) -> np.ndarray:
        """
        Calcule la distribution stationnaire P(N = n).
        
        Pour M/M/c:
        - P(N = n) = (aⁿ/n!) · P₀,  pour n < c
        - P(N = n) = (aⁿ/(c! · cⁿ⁻ᶜ)) · P₀,  pour n ≥ c
        
        Args:
            P0: Probabilité système vide
            a: Intensité de trafic λ/μ
            c: Nombre de serveurs
            rho: Facteur d'utilisation
            n_max: Nombre max d'états à calculer
            
        Returns:
            Array des probabilités P(N = n)
        """
        probs = np.zeros(n_max + 1)
        
        for n in range(n_max + 1):
            if n < c:
                probs[n] = ((a ** n) / factorial(n)) * P0
            else:
                probs[n] = ((a ** c) / factorial(c)) * (rho ** (n - c)) * P0
        
        return probs
    
    def compute_erlang_c(self) -> float:
        """
        Calcule la formule d'Erlang C: probabilité qu'un client doive attendre.
        
        C(c, a) = P(tous les serveurs occupés ET file non vide | arrivée)
        
        Returns:
            Probabilité d'attente (Erlang C)
        """
        metrics = self.get_theoretical_metrics()
        a = self.a
        c = self.c
        rho = self.rho
        P0 = metrics.P0
        
        return ((a ** c) / factorial(c)) * P0 / (1 - rho)
    
    def simulate(
        self,
        n_customers: int = 1000,
        max_time: Optional[float] = None
    ) -> SimulationResults:
        """
        Simule une file M/M/c avec n_customers clients.
        
        Utilise un tas (heap) pour gérer efficacement les c serveurs.
        Le tas stocke les instants de fin de service, permettant de
        trouver en O(log c) le prochain serveur disponible.
        
        Algorithme:
        1. Générer arrivées T_n et temps de service S_n
        2. Pour chaque arrivée:
           - Libérer les serveurs dont le service est terminé
           - Affecter le client au serveur disponible le plus tôt
           - Calculer son instant de départ
        
        Args:
            n_customers: Nombre de clients à simuler
            max_time: Temps maximum de simulation
            
        Returns:
            SimulationResults avec traces complètes
        """
        # Générer temps
        interarrival_times = self._generate_interarrival_times(n_customers)
        service_times = self._generate_service_times(n_customers)
        arrival_times = np.cumsum(interarrival_times)
        
        # Filtrer par temps max
        if max_time is not None:
            mask = arrival_times <= max_time
            arrival_times = arrival_times[mask]
            service_times = service_times[:len(arrival_times)]
            n_customers = len(arrival_times)
        
        if n_customers == 0:
            return SimulationResults()
        
        # Tas des serveurs (contient les instants de fin de service)
        servers = []  # min-heap: premier élément = prochain serveur libre
        
        # Résultats
        service_start_times = np.zeros(n_customers)
        departure_times = np.zeros(n_customers)
        waiting_times = np.zeros(n_customers)
        system_times = np.zeros(n_customers)
        
        # Traces temporelles
        events = []
        
        for i in range(n_customers):
            t_arrival = arrival_times[i]
            
            # Libérer les serveurs qui ont fini avant cette arrivée
            while servers and servers[0] <= t_arrival:
                t_depart = heapq.heappop(servers)
                events.append((t_depart, -1, 'departure'))
            
            # Déterminer l'instant de début de service
            if len(servers) < self.c:
                # Serveur disponible: service immédiat
                t_service_start = t_arrival
            else:
                # Tous les serveurs occupés: attendre le premier disponible
                t_service_start = heapq.heappop(servers)
            
            # Calculer instant de départ
            t_departure = t_service_start + service_times[i]
            
            # Ajouter au tas des serveurs
            heapq.heappush(servers, t_departure)
            
            # Enregistrer résultats
            service_start_times[i] = t_service_start
            departure_times[i] = t_departure
            waiting_times[i] = t_service_start - t_arrival
            system_times[i] = t_departure - t_arrival
            
            # Événements pour trace
            events.append((t_arrival, 1, 'arrival'))
        
        # Ajouter les départs restants
        while servers:
            events.append((heapq.heappop(servers), -1, 'departure'))
        
        # Construire traces temporelles
        time_trace, queue_trace, system_trace = self._build_traces(events, self.c)
        
        results = SimulationResults(
            arrival_times=arrival_times,
            service_start_times=service_start_times,
            departure_times=departure_times,
            service_times=service_times[:n_customers],
            waiting_times=waiting_times,
            system_times=system_times,
            n_arrivals=n_customers,
            n_served=n_customers,
            n_rejected=0,
            time_trace=time_trace,
            queue_length_trace=queue_trace,
            system_size_trace=system_trace
        )
        
        results.empirical_metrics = self.compute_empirical_metrics(results)
        
        return results
    
    def _build_traces(
        self,
        events: list,
        c: int
    ) -> tuple:
        """Construit les traces N(t) et Q(t) à partir des événements."""
        events.sort(key=lambda x: (x[0], -x[1]))
        
        time_trace = [0.0]
        system_trace = [0]
        queue_trace = [0]
        
        current_system = 0
        
        for time, delta, _ in events:
            current_system += delta
            current_queue = max(0, current_system - c)
            
            time_trace.append(time)
            system_trace.append(current_system)
            queue_trace.append(current_queue)
        
        return np.array(time_trace), np.array(queue_trace), np.array(system_trace)
    
    def find_optimal_servers(
        self,
        target_wq: float = None,
        target_prob_wait: float = None
    ) -> int:
        """
        Trouve le nombre optimal de serveurs pour atteindre un objectif.
        
        Args:
            target_wq: Temps d'attente cible (en heures)
            target_prob_wait: Probabilité d'attente cible (Erlang C)
            
        Returns:
            Nombre minimum de serveurs nécessaires
        """
        if target_wq is None and target_prob_wait is None:
            raise ValueError("Spécifier target_wq ou target_prob_wait")
        
        # Nombre minimum pour stabilité
        c_min = int(np.ceil(self.a))
        
        for c in range(c_min, c_min + 100):
            test_queue = MMcQueue(self.lambda_rate, self.mu_rate, c)
            metrics = test_queue.get_theoretical_metrics()
            
            if target_wq is not None and metrics.Wq <= target_wq:
                return c
            
            if target_prob_wait is not None:
                erlang_c = test_queue.compute_erlang_c()
                if erlang_c <= target_prob_wait:
                    return c
        
        return c_min + 100  # Fallback
