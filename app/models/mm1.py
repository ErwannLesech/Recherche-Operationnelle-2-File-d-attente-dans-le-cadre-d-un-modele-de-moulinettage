"""
Modèle de file d'attente M/M/1.

File d'attente avec:
- Arrivées selon un processus de Poisson (M = Markovien)
- Temps de service exponentiels (M)
- Un seul serveur (1)
- Capacité infinie
- Discipline FIFO

C'est le modèle le plus simple et fondamental en théorie des files d'attente.
Il sert de base pour comprendre les systèmes plus complexes.

Formules théoriques (sous condition de stabilité ρ = λ/μ < 1):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│ Métrique              │ Formule                          │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│ Utilisation ρ         │ λ/μ                              │
│ P₀ (système vide)     │ 1 - ρ                            │
│ Pₙ (n clients)        │ (1 - ρ)ρⁿ                        │
│ L (nb moyen système)  │ ρ/(1 - ρ)                        │
│ Lq (nb moyen file)    │ ρ²/(1 - ρ)                       │
│ W (temps système)     │ 1/(μ - λ)                        │
│ Wq (temps attente)    │ ρ/(μ - λ) = λ/(μ(μ - λ))         │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Application à la moulinette:
- λ = taux de soumission des étudiants (tags/heure)
- μ = capacité de traitement du serveur (corrections/heure)
- Un seul serveur de correction

Auteurs: ERO2 Team Markov Moulinette Configurators - EPITA
"""

import numpy as np
from typing import Optional
from .base_queue import BaseQueueModel, QueueMetrics, SimulationResults


class MM1Queue(BaseQueueModel):
    """
    Implémentation du modèle M/M/1.
    
    Cas d'usage pour la moulinette:
    - Modèle de base quand on a un seul runner de correction
    - Permet de calculer le temps d'attente moyen des étudiants
    - Base pour dimensionner le nombre de serveurs nécessaires
    
    Exemple:
        >>> queue = MM1Queue(lambda_rate=10, mu_rate=12)  # 10 tags/h, 12 corrections/h
        >>> metrics = queue.get_theoretical_metrics()
        >>> print(f"Temps d'attente moyen: {metrics.Wq:.2f} heures")
    """
    
    def __init__(
        self,
        lambda_rate: float,
        mu_rate: float,
        seed: Optional[int] = None
    ):
        """
        Initialise une file M/M/1.
        
        Args:
            lambda_rate: Taux d'arrivée λ (soumissions/heure)
            mu_rate: Taux de service μ (corrections/heure)
            seed: Graine aléatoire pour reproductibilité
            
        Raises:
            ValueError: Si ρ = λ/μ ≥ 1 (système instable)
        """
        super().__init__(lambda_rate, mu_rate, c=1, K=None, seed=seed)
        
        if not self.is_stable:
            raise ValueError(
                f"Système instable: ρ = {self.rho:.4f} ≥ 1. "
                f"Le taux d'arrivée λ={lambda_rate} doit être < μ={mu_rate}"
            )
    
    def _get_kendall_notation(self) -> str:
        return "M/M/1"
    
    def _get_model_description(self) -> str:
        return """
        ╔══════════════════════════════════════════════════════════════════╗
        ║                        MODÈLE M/M/1                              ║
        ╠══════════════════════════════════════════════════════════════════╣
        ║ Description:                                                     ║
        ║   File d'attente simple avec un seul serveur.                    ║
        ║                                                                  ║
        ║ Hypothèses:                                                      ║
        ║   • Arrivées: Processus de Poisson de taux λ                     ║
        ║   • Service: Temps exponentiels de taux μ                        ║
        ║   • Serveur: 1 seul                                              ║
        ║   • Capacité: Infinie                                            ║
        ║   • Discipline: FIFO (Premier arrivé, premier servi)             ║
        ║                                                                  ║
        ║ Condition de stabilité: ρ = λ/μ < 1                              ║
        ║                                                                  ║
        ║ Application moulinette:                                          ║
        ║   Modélise un runner unique traitant les soumissions             ║
        ║   séquentiellement. Simple mais limité pour les rushes.          ║
        ╚══════════════════════════════════════════════════════════════════╝
        """
    
    def compute_theoretical_metrics(self) -> QueueMetrics:
        """
        Calcule les métriques théoriques pour M/M/1.
        
        Formules utilisées:
        - ρ = λ/μ (facteur d'utilisation)
        - P₀ = 1 - ρ (probabilité système vide)
        - L = ρ/(1-ρ) (nombre moyen dans le système)
        - Lq = ρ²/(1-ρ) (nombre moyen en file)
        - W = 1/(μ-λ) (temps moyen dans le système)
        - Wq = ρ/(μ-λ) (temps moyen d'attente)
        
        Returns:
            QueueMetrics avec toutes les métriques calculées
        """
        rho = self.rho
        lambda_rate = self.lambda_rate
        mu_rate = self.mu_rate
        
        # Probabilité système vide
        P0 = 1 - rho
        
        # Nombre moyen de clients
        L = rho / (1 - rho)
        Lq = (rho ** 2) / (1 - rho)
        Ls = rho  # Nombre moyen en service = ρ pour M/M/1
        
        # Temps moyens
        W = 1 / (mu_rate - lambda_rate)
        Wq = rho / (mu_rate - lambda_rate)
        Ws = 1 / mu_rate
        
        # Distribution stationnaire P(N = n) = (1-ρ)ρⁿ
        # On calcule pour n = 0, 1, ..., n_max où P(N > n_max) < ε
        epsilon = 1e-6
        n_max = int(np.ceil(np.log(epsilon) / np.log(rho))) if rho > 0 else 1
        n_max = min(n_max, 1000)  # Limite pour éviter tableaux trop grands
        
        n_values = np.arange(n_max + 1)
        state_probs = (1 - rho) * (rho ** n_values)
        
        return QueueMetrics(
            rho=rho,
            L=L,
            Lq=Lq,
            Ls=Ls,
            W=W,
            Wq=Wq,
            Ws=Ws,
            P0=P0,
            Pk=0.0,  # Pas de blocage dans M/M/1
            lambda_eff=lambda_rate,
            throughput=lambda_rate,
            state_probabilities=state_probs
        )
    
    def simulate(
        self,
        n_customers: int = 1000,
        max_time: Optional[float] = None
    ) -> SimulationResults:
        """
        Simule une file M/M/1 avec n_customers clients.
        
        Algorithme:
        1. Générer les temps inter-arrivées Q_k ~ Exp(λ)
        2. Calculer les instants d'arrivée T_n = Σ Q_k
        3. Générer les temps de service S_k ~ Exp(μ)
        4. Calculer les départs:
           - D₁ = T₁ + S₁
           - Dₙ = max(Tₙ, Dₙ₋₁) + Sₙ
        
        Args:
            n_customers: Nombre de clients à simuler
            max_time: Temps maximum de simulation (optionnel)
            
        Returns:
            SimulationResults avec toutes les traces
        """
        # Générer temps inter-arrivées et de service
        interarrival_times = self._generate_interarrival_times(n_customers)
        service_times = self._generate_service_times(n_customers)
        
        # Calculer instants d'arrivée: T_n = Σ_{k=1}^n Q_k
        arrival_times = np.cumsum(interarrival_times)
        
        # Filtrer par temps max si spécifié
        if max_time is not None:
            mask = arrival_times <= max_time
            arrival_times = arrival_times[mask]
            service_times = service_times[:len(arrival_times)]
            n_customers = len(arrival_times)
        
        if n_customers == 0:
            return SimulationResults()
        
        # Initialiser tableaux de résultats
        service_start_times = np.zeros(n_customers)
        departure_times = np.zeros(n_customers)
        waiting_times = np.zeros(n_customers)
        system_times = np.zeros(n_customers)
        
        # Simulation FIFO
        for k in range(n_customers):
            if k == 0:
                # Premier client: service immédiat
                service_start_times[k] = arrival_times[k]
            else:
                # Client suivant: attend si serveur occupé
                # B_n = max(T_n, D_{n-1})
                service_start_times[k] = max(arrival_times[k], departure_times[k-1])
            
            # Instant de départ: D_n = B_n + S_n
            departure_times[k] = service_start_times[k] + service_times[k]
            
            # Temps d'attente: W_q = B_n - T_n
            waiting_times[k] = service_start_times[k] - arrival_times[k]
            
            # Temps dans le système: W = D_n - T_n
            system_times[k] = departure_times[k] - arrival_times[k]
        
        # Construire trace temporelle N(t)
        time_trace, queue_trace, system_trace = self._build_time_traces(
            arrival_times, departure_times, service_start_times
        )
        
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
        
        # Calculer métriques empiriques
        results.empirical_metrics = self.compute_empirical_metrics(results)
        
        return results
    
    def _build_time_traces(
        self,
        arrival_times: np.ndarray,
        departure_times: np.ndarray,
        service_start_times: np.ndarray
    ) -> tuple:
        """
        Construit les traces temporelles N(t) et Q(t).
        
        N(t) = nombre de clients dans le système à l'instant t
        Q(t) = nombre de clients en file d'attente à l'instant t
        
        Returns:
            Tuple (time_trace, queue_trace, system_trace)
        """
        # Créer événements triés
        events = []
        for t in arrival_times:
            events.append((t, 1, 'arrival'))  # +1 pour arrivée
        for t in departure_times:
            events.append((t, -1, 'departure'))  # -1 pour départ
        
        events.sort(key=lambda x: (x[0], -x[1]))  # Trier par temps, départs avant arrivées si même temps
        
        time_trace = [0.0]
        system_trace = [0]
        queue_trace = [0]
        
        current_system = 0
        current_queue = 0
        
        for time, delta, event_type in events:
            current_system += delta
            
            if event_type == 'arrival':
                if current_system > 1:  # Si serveur était occupé
                    current_queue += 1
            else:  # departure
                if current_queue > 0:
                    current_queue -= 1
            
            time_trace.append(time)
            system_trace.append(current_system)
            queue_trace.append(current_queue)
        
        return np.array(time_trace), np.array(queue_trace), np.array(system_trace)
    
    def compute_probability_n_customers(self, n: int) -> float:
        """
        Calcule P(N = n), la probabilité d'avoir exactement n clients.
        
        Pour M/M/1: P(N = n) = (1 - ρ)ρⁿ
        
        Args:
            n: Nombre de clients
            
        Returns:
            Probabilité P(N = n)
        """
        if n < 0:
            return 0.0
        return (1 - self.rho) * (self.rho ** n)
    
    def compute_percentile_waiting_time(self, p: float) -> float:
        """
        Calcule le percentile p du temps d'attente.
        
        Pour M/M/1, le temps d'attente suit une distribution mixte:
        - P(Wq = 0) = 1 - ρ (service immédiat si système vide)
        - Pour t > 0: F(t) = 1 - ρ·exp(-(μ-λ)t)
        
        Args:
            p: Percentile souhaité (entre 0 et 1)
            
        Returns:
            Temps d'attente correspondant au percentile p
        """
        if p <= 1 - self.rho:
            return 0.0
        
        # Résoudre 1 - ρ·exp(-(μ-λ)t) = p
        # => t = -ln((1-p)/ρ) / (μ-λ)
        rate = self.mu_rate - self.lambda_rate
        return -np.log((1 - p) / self.rho) / rate
