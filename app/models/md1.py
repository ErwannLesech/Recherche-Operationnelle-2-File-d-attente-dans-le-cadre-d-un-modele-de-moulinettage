"""
Modèle de file d'attente M/D/1.

File d'attente avec:
- Arrivées selon un processus de Poisson (M)
- Temps de service DÉTERMINISTES (D) - constants
- Un seul serveur (1)
- Capacité infinie
- Discipline FIFO

Ce modèle est intéressant pour la moulinette car:
- Certains traitements ont des durées très prévisibles
- Les tests automatisés ont souvent des timeouts fixes
- Comparaison avec M/M/1 pour voir l'impact de la variance

Formules théoriques (Pollaczek-Khinchin pour temps constant):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│ Métrique              │ Formule                                   │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│ Utilisation ρ         │ λ/μ = λ·D (D = temps de service)         │
│ Service D             │ Temps de service constant = 1/μ           │
│                       │                                           │
│ Variance service Var  │ 0 (déterministe)                          │
│ Coefficient C²s       │ Var/(1/μ)² = 0                            │
│                       │                                           │
│ Lq (Pollaczek)        │ ρ²/(2(1-ρ)) = moitié de M/M/1!           │
│ L                     │ Lq + ρ                                    │
│ Wq                    │ ρ/(2μ(1-ρ))                               │
│ W                     │ Wq + D                                    │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RÉSULTAT CLÉ: Lq(M/D/1) = Lq(M/M/1) / 2
Le temps d'attente est DIVISÉ PAR 2 par rapport à M/M/1!

Cela montre l'intérêt d'avoir des temps de service prévisibles
(timeouts, quotas de temps, parallélisation déterministe).

Auteurs: ERO2 Team - EPITA
"""

import numpy as np
from typing import Optional
from .base_queue import BaseQueueModel, QueueMetrics, SimulationResults


class MD1Queue(BaseQueueModel):
    """
    Implémentation du modèle M/D/1.
    
    Cas d'usage pour la moulinette:
    - Modélise des corrections avec temps fixe (timeout)
    - Compare avec M/M/1 pour quantifier l'impact de la variance
    - Utile pour la partie finale de la pipeline (déterministe)
    
    Exemple:
        >>> # 10 tags/h, service fixe de 5 min = 12 corrections/h
        >>> queue = MD1Queue(lambda_rate=10, mu_rate=12)
        >>> metrics = queue.get_theoretical_metrics()
        >>> # Comparer avec M/M/1: Lq est 2x plus petit!
    """
    
    def __init__(
        self,
        lambda_rate: float,
        mu_rate: float,
        seed: Optional[int] = None
    ):
        """
        Initialise une file M/D/1.
        
        Args:
            lambda_rate: Taux d'arrivée λ
            mu_rate: Taux de service μ (temps de service = 1/μ)
            seed: Graine aléatoire (pour les arrivées)
            
        Raises:
            ValueError: Si ρ = λ/μ ≥ 1
        """
        super().__init__(lambda_rate, mu_rate, c=1, K=None, seed=seed)
        
        if not self.is_stable:
            raise ValueError(
                f"Système instable: ρ = {self.rho:.4f} ≥ 1"
            )
        
        # Temps de service déterministe
        self.service_time = 1 / mu_rate
    
    def _get_kendall_notation(self) -> str:
        return "M/D/1"
    
    def _get_model_description(self) -> str:
        return f"""
        ╔══════════════════════════════════════════════════════════════════╗
        ║                        MODÈLE M/D/1                              ║
        ╠══════════════════════════════════════════════════════════════════╣
        ║ Description:                                                     ║
        ║   File d'attente avec temps de service CONSTANT.                ║
        ║                                                                  ║
        ║ Hypothèses:                                                      ║
        ║   • Arrivées: Processus de Poisson de taux λ                    ║
        ║   • Service: Temps CONSTANT D = 1/μ = {self.service_time:.4f}           ║
        ║   • Serveur: 1 seul                                             ║
        ║   • Capacité: Infinie                                           ║
        ║   • Discipline: FIFO                                            ║
        ║                                                                  ║
        ║ Condition de stabilité: ρ = λ/μ < 1                             ║
        ║                                                                  ║
        ║ AVANTAGE CLÉ:                                                    ║
        ║   Le temps d'attente moyen est DIVISÉ PAR 2 par rapport         ║
        ║   à M/M/1! Cela montre l'intérêt des timeouts et de la         ║
        ║   prévisibilité des temps de traitement.                        ║
        ║                                                                  ║
        ║ Application moulinette:                                          ║
        ║   Modélise des runners avec timeout fixe ou des tests          ║
        ║   dont la durée est très prévisible.                            ║
        ╚══════════════════════════════════════════════════════════════════╝
        """
    
    def compute_theoretical_metrics(self) -> QueueMetrics:
        """
        Calcule les métriques théoriques pour M/D/1.
        
        Utilise la formule de Pollaczek-Khinchin simplifiée
        pour le cas déterministe (variance = 0).
        
        Formule P-K générale: Lq = λ²·Var[S] + ρ² / (2(1-ρ))
        Pour service constant: Var[S] = 0
        Donc: Lq = ρ² / (2(1-ρ)) = Lq(M/M/1) / 2
        
        Returns:
            QueueMetrics avec métriques calculées
        """
        lambda_rate = self.lambda_rate
        mu_rate = self.mu_rate
        rho = self.rho
        D = self.service_time
        
        # Nombre moyen en file (Pollaczek-Khinchin, cas D)
        # Lq = ρ² / (2(1-ρ))
        Lq = (rho ** 2) / (2 * (1 - rho))
        
        # Nombre moyen en service = ρ
        Ls = rho
        
        # Nombre moyen total
        L = Lq + Ls
        
        # Temps moyens
        Wq = Lq / lambda_rate  # = ρ / (2μ(1-ρ))
        Ws = D
        W = Wq + Ws
        
        # P₀ pour M/D/1 n'a pas de forme simple fermée
        # On utilise une approximation
        P0 = 1 - rho  # Approximation (exacte pour les moments)
        
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
            throughput=lambda_rate,
            state_probabilities=np.array([P0, rho])  # Approximation
        )
    
    def _generate_service_times(self, n: int) -> np.ndarray:
        """
        Génère n temps de service CONSTANTS.
        
        Surcharge la méthode parente pour retourner des valeurs
        déterministes au lieu d'exponentielles.
        
        Args:
            n: Nombre de temps à générer
            
        Returns:
            Array de n valeurs toutes égales à D = 1/μ
        """
        return np.full(n, self.service_time)
    
    def simulate(
        self,
        n_customers: int = 1000,
        max_time: Optional[float] = None
    ) -> SimulationResults:
        """
        Simule une file M/D/1.
        
        Identique à M/M/1 sauf que les temps de service
        sont constants au lieu d'exponentiels.
        
        Args:
            n_customers: Nombre de clients à simuler
            max_time: Temps maximum
            
        Returns:
            SimulationResults avec traces
        """
        # Générer temps (arrivées aléatoires, services constants)
        interarrival_times = self._generate_interarrival_times(n_customers)
        service_times = self._generate_service_times(n_customers)  # Constants!
        arrival_times = np.cumsum(interarrival_times)
        
        # Filtrer par temps max
        if max_time is not None:
            mask = arrival_times <= max_time
            arrival_times = arrival_times[mask]
            service_times = service_times[:len(arrival_times)]
            n_customers = len(arrival_times)
        
        if n_customers == 0:
            return SimulationResults()
        
        # Simulation FIFO
        service_start_times = np.zeros(n_customers)
        departure_times = np.zeros(n_customers)
        waiting_times = np.zeros(n_customers)
        system_times = np.zeros(n_customers)
        
        for k in range(n_customers):
            if k == 0:
                service_start_times[k] = arrival_times[k]
            else:
                service_start_times[k] = max(arrival_times[k], departure_times[k-1])
            
            departure_times[k] = service_start_times[k] + service_times[k]
            waiting_times[k] = service_start_times[k] - arrival_times[k]
            system_times[k] = departure_times[k] - arrival_times[k]
        
        # Traces temporelles
        time_trace, queue_trace, system_trace = self._build_traces(
            arrival_times, departure_times
        )
        
        results = SimulationResults(
            arrival_times=arrival_times,
            service_start_times=service_start_times,
            departure_times=departure_times,
            service_times=service_times,
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
        arrival_times: np.ndarray,
        departure_times: np.ndarray
    ) -> tuple:
        """Construit les traces N(t) et Q(t)."""
        events = []
        for t in arrival_times:
            events.append((t, 1, 'arrival'))
        for t in departure_times:
            events.append((t, -1, 'departure'))
        
        events.sort(key=lambda x: (x[0], -x[1]))
        
        time_trace = [0.0]
        system_trace = [0]
        queue_trace = [0]
        
        current_system = 0
        
        for time, delta, event_type in events:
            current_system += delta
            current_queue = max(0, current_system - 1)
            
            time_trace.append(time)
            system_trace.append(current_system)
            queue_trace.append(current_queue)
        
        return np.array(time_trace), np.array(queue_trace), np.array(system_trace)
    
    def compare_with_mm1(self) -> dict:
        """
        Compare les métriques M/D/1 avec M/M/1 équivalent.
        
        Démontre que le temps d'attente est divisé par 2
        quand les temps de service sont déterministes.
        
        Returns:
            Dict avec comparaison des métriques
        """
        from .mm1 import MM1Queue
        
        md1_metrics = self.get_theoretical_metrics()
        mm1_queue = MM1Queue(self.lambda_rate, self.mu_rate)
        mm1_metrics = mm1_queue.get_theoretical_metrics()
        
        return {
            'md1': {
                'Lq': md1_metrics.Lq,
                'Wq': md1_metrics.Wq,
                'L': md1_metrics.L,
                'W': md1_metrics.W
            },
            'mm1': {
                'Lq': mm1_metrics.Lq,
                'Wq': mm1_metrics.Wq,
                'L': mm1_metrics.L,
                'W': mm1_metrics.W
            },
            'ratio_Lq': md1_metrics.Lq / mm1_metrics.Lq if mm1_metrics.Lq > 0 else 0,
            'ratio_Wq': md1_metrics.Wq / mm1_metrics.Wq if mm1_metrics.Wq > 0 else 0,
            'improvement_percent': (1 - md1_metrics.Wq / mm1_metrics.Wq) * 100 if mm1_metrics.Wq > 0 else 0
        }
