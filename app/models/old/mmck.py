"""
Modèle de file d'attente M/M/c/K.

File d'attente avec:
- Arrivées selon un processus de Poisson (M)
- Temps de service exponentiels (M)
- c serveurs en parallèle
- Capacité FINIE K (incluant clients en service)
- Discipline FIFO

Ce modèle est CRUCIAL pour la moulinette car:
1. Les ressources mémoire/CPU sont limitées
2. On ne peut pas accepter un nombre infini de soumissions
3. Les rejets (blocages) sont une métrique importante

Formules théoriques:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│ Paramètre             │ Formule                                   │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│ a = λ/μ              │ Intensité de trafic offert                 │
│ ρ = λ/(cμ)           │ Facteur d'utilisation                      │
│                       │                                           │
│ Taux de transition:   │                                           │
│   λₙ = λ si n < K    │ Taux d'arrivée (0 si n = K)               │
│   μₙ = nμ si n ≤ c   │ Taux de service (cμ si n > c)             │
│                       │                                           │
│ P₀                    │ Constante de normalisation                │
│ Pₙ (0 ≤ n < c)       │ (aⁿ/n!) · P₀                              │
│ Pₙ (c ≤ n ≤ K)       │ (aᶜ/c!) · ρⁿ⁻ᶜ · P₀                       │
│                       │                                           │
│ P_K (blocage)         │ Probabilité d'être rejeté = πₖ            │
│ λ_eff                 │ λ(1 - P_K) = taux effectif                │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMPORTANT: Le système est TOUJOURS stable (pas de condition ρ < 1)
car les clients en excès sont simplement rejetés.

Application à la moulinette:
- K = capacité du buffer de soumissions
- P_K = taux de rejet (soumissions perdues)
- Objectif: minimiser P_K tout en limitant les coûts

Auteurs: ERO2 Team Markov Moulinette Configurators - EPITA
"""

import numpy as np
import heapq
from math import factorial
from typing import Optional
from ..base_queue import BaseQueueModel, QueueMetrics, SimulationResults


class MMcKQueue(BaseQueueModel):
    """
    Implémentation du modèle M/M/c/K (Erlang B généralisé).
    
    Cas d'usage pour la moulinette:
    - Modèle réaliste avec capacité limitée
    - Permet de calculer le taux de rejet des soumissions
    - Base pour l'optimisation coût/qualité de service
    
    Exemple:
        >>> queue = MMcKQueue(lambda_rate=100, mu_rate=10, c=5, K=20)
        >>> metrics = queue.get_theoretical_metrics()
        >>> print(f"Taux de rejet: {metrics.Pk:.2%}")
    """
    
    def __init__(
        self,
        lambda_rate: float,
        mu_rate: float,
        c: int = 1,
        K: int = 10,
        seed: Optional[int] = None
    ):
        """
        Initialise une file M/M/c/K.
        
        Args:
            lambda_rate: Taux d'arrivée λ
            mu_rate: Taux de service μ par serveur
            c: Nombre de serveurs
            K: Capacité maximale du système (K ≥ c)
            seed: Graine aléatoire
        """
        if K < c:
            raise ValueError(f"K doit être ≥ c. Reçu K={K}, c={c}")
        
        super().__init__(lambda_rate, mu_rate, c=c, K=K, seed=seed)
        
        # Intensité de trafic offert
        self.a = lambda_rate / mu_rate
    
    def _get_kendall_notation(self) -> str:
        return f"M/M/{self.c}/{self.K}"
    
    def _get_model_description(self) -> str:
        return f"""
        ╔══════════════════════════════════════════════════════════════════╗
        ║                      MODÈLE M/M/{self.c}/{self.K}                          ║
        ╠══════════════════════════════════════════════════════════════════╣
        ║ Description:                                                     ║
        ║   File d'attente à capacité finie avec {self.c} serveurs.              ║
        ║   Maximum {self.K} clients dans le système (file + service).          ║
        ║                                                                  ║
        ║ Hypothèses:                                                      ║
        ║   • Arrivées: Poisson(λ) - rejetées si système plein           ║
        ║   • Service: Exp(μ) par serveur                                 ║
        ║   • Serveurs: {self.c} serveurs identiques                           ║
        ║   • Capacité: {self.K} clients maximum (K ≥ c)                        ║
        ║   • Discipline: FIFO                                            ║
        ║                                                                  ║
        ║ Stabilité: TOUJOURS stable (clients rejetés si plein)           ║
        ║                                                                  ║
        ║ Application moulinette:                                          ║
        ║   Modèle réaliste avec buffer limité.                           ║
        ║   Métrique clé: P_K = probabilité de rejet.                     ║
        ║   Permet d'optimiser K et c pour minimiser les pertes.          ║
        ╚══════════════════════════════════════════════════════════════════╝
        """
    
    def compute_theoretical_metrics(self) -> QueueMetrics:
        """
        Calcule les métriques théoriques pour M/M/c/K.
        
        Utilise les équations de balance détaillée pour
        calculer la distribution stationnaire.
        
        Returns:
            QueueMetrics avec métriques et probabilités
        """
        lambda_rate = self.lambda_rate
        mu_rate = self.mu_rate
        c = self.c
        K = self.K
        a = self.a
        rho = self.rho
        
        # Calcul de la distribution stationnaire π_n
        pi = self._compute_stationary_distribution()
        
        # Probabilités clés
        P0 = pi[0]
        Pk = pi[K]  # Probabilité de blocage
        
        # Taux d'arrivée effectif
        lambda_eff = lambda_rate * (1 - Pk)
        
        # Nombre moyen dans le système
        L = sum(n * pi[n] for n in range(K + 1))
        
        # Nombre moyen en file (attente)
        Lq = sum((n - c) * pi[n] for n in range(c, K + 1))
        
        # Nombre moyen en service
        Ls = sum(min(n, c) * pi[n] for n in range(K + 1))
        
        # Temps moyens (via loi de Little avec λ_eff)
        W = L / lambda_eff if lambda_eff > 0 else 0
        Wq = Lq / lambda_eff if lambda_eff > 0 else 0
        Ws = 1 / mu_rate
        
        return QueueMetrics(
            rho=rho,
            L=L,
            Lq=Lq,
            Ls=Ls,
            W=W,
            Wq=Wq,
            Ws=Ws,
            P0=P0,
            Pk=Pk,
            lambda_eff=lambda_eff,
            throughput=lambda_eff,
            state_probabilities=pi
        )
    
    def _compute_stationary_distribution(self) -> np.ndarray:
        """
        Calcule la distribution stationnaire π_n pour n = 0, 1, ..., K.
        
        Équations de balance détaillée:
        - λ·πₙ = μₙ₊₁·πₙ₊₁  pour n = 0, 1, ..., K-1
        
        Avec:
        - μₙ = n·μ   pour n ≤ c
        - μₙ = c·μ   pour n > c
        
        Solution:
        - πₙ = (aⁿ/n!) · π₀          pour n ≤ c
        - πₙ = (aᶜ/c!) · ρⁿ⁻ᶜ · π₀   pour n > c
        
        Normalisation: Σπₙ = 1
        """
        a = self.a
        c = self.c
        K = self.K
        rho = self.rho
        
        # Calcul des termes non normalisés
        pi_unnorm = np.zeros(K + 1)
        
        for n in range(K + 1):
            if n <= c:
                pi_unnorm[n] = (a ** n) / factorial(n)
            else:
                pi_unnorm[n] = ((a ** c) / factorial(c)) * (rho ** (n - c))
        
        # Normalisation
        norm = np.sum(pi_unnorm)
        pi = pi_unnorm / norm
        
        return pi
    
    def simulate(
        self,
        n_customers: int = 1000,
        max_time: Optional[float] = None
    ) -> SimulationResults:
        """
        Simule une file M/M/c/K avec rejet des clients quand plein.
        
        Algorithme événementiel:
        1. Générer les arrivées potentielles
        2. Pour chaque arrivée:
           - Vérifier si le système a de la place (N < K)
           - Si oui: accepter et traiter
           - Si non: rejeter (comptabiliser)
        
        Args:
            n_customers: Nombre de clients potentiels à simuler
            max_time: Temps maximum
            
        Returns:
            SimulationResults avec traces et compteurs de rejet
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
        
        # Tas des serveurs occupés
        servers = []
        
        # Listes pour clients acceptés
        accepted_arrivals = []
        accepted_service_times = []
        service_start_times_list = []
        departure_times_list = []
        waiting_times_list = []
        system_times_list = []
        
        # Compteurs
        n_rejected = 0
        
        # Traces temporelles
        events = []
        
        for i in range(n_customers):
            t_arrival = arrival_times[i]
            
            # Libérer les serveurs terminés
            while servers and servers[0] <= t_arrival:
                t_depart = heapq.heappop(servers)
                events.append((t_depart, -1, 'departure'))
            
            # Compter clients actuels dans le système
            n_in_system = len(servers)
            
            if n_in_system < self.K:
                # Accepter le client
                accepted_arrivals.append(t_arrival)
                accepted_service_times.append(service_times[i])
                
                if n_in_system < self.c:
                    # Serveur disponible
                    t_service_start = t_arrival
                else:
                    # Attendre le prochain serveur libre
                    t_service_start = heapq.heappop(servers)
                
                t_departure = t_service_start + service_times[i]
                
                heapq.heappush(servers, t_departure)
                
                service_start_times_list.append(t_service_start)
                departure_times_list.append(t_departure)
                waiting_times_list.append(t_service_start - t_arrival)
                system_times_list.append(t_departure - t_arrival)
                
                events.append((t_arrival, 1, 'arrival'))
            else:
                # Rejeter le client (système plein)
                n_rejected += 1
        
        # Départs restants
        while servers:
            events.append((heapq.heappop(servers), -1, 'departure'))
        
        n_served = len(accepted_arrivals)
        
        # Construire traces
        time_trace, queue_trace, system_trace = self._build_traces(events)
        
        results = SimulationResults(
            arrival_times=np.array(accepted_arrivals),
            service_start_times=np.array(service_start_times_list),
            departure_times=np.array(departure_times_list),
            service_times=np.array(accepted_service_times),
            waiting_times=np.array(waiting_times_list),
            system_times=np.array(system_times_list),
            n_arrivals=n_customers,
            n_served=n_served,
            n_rejected=n_rejected,
            time_trace=time_trace,
            queue_length_trace=queue_trace,
            system_size_trace=system_trace
        )
        
        results.empirical_metrics = self.compute_empirical_metrics(results)
        
        return results
    
    def _build_traces(self, events: list) -> tuple:
        """Construit les traces temporelles."""
        events.sort(key=lambda x: (x[0], -x[1]))
        
        time_trace = [0.0]
        system_trace = [0]
        queue_trace = [0]
        
        current_system = 0
        
        for time, delta, _ in events:
            current_system += delta
            current_queue = max(0, current_system - self.c)
            
            time_trace.append(time)
            system_trace.append(current_system)
            queue_trace.append(current_queue)
        
        return np.array(time_trace), np.array(queue_trace), np.array(system_trace)
    
    def find_optimal_capacity(
        self,
        target_blocking: float = 0.01
    ) -> int:
        """
        Trouve la capacité K minimale pour atteindre un taux de blocage cible.
        
        Args:
            target_blocking: Probabilité de blocage maximale acceptable
            
        Returns:
            Capacité K minimale
        """
        for K in range(self.c, self.c + 200):
            test_queue = MMcKQueue(
                self.lambda_rate, self.mu_rate, self.c, K
            )
            metrics = test_queue.get_theoretical_metrics()
            
            if metrics.Pk <= target_blocking:
                return K
        
        return self.c + 200
    
    def find_optimal_servers_for_blocking(
        self,
        target_blocking: float = 0.01
    ) -> int:
        """
        Trouve le nombre de serveurs c pour un taux de blocage cible.
        
        Args:
            target_blocking: Probabilité de blocage cible
            
        Returns:
            Nombre optimal de serveurs
        """
        for c in range(1, 100):
            # Garder K proportionnel à c
            K = max(self.K, c * 2)
            test_queue = MMcKQueue(
                self.lambda_rate, self.mu_rate, c, K
            )
            metrics = test_queue.get_theoretical_metrics()
            
            if metrics.Pk <= target_blocking:
                return c
        
        return 100
