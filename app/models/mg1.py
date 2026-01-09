"""
Modèle de file d'attente M/G/1.

File d'attente avec:
- Arrivées selon un processus de Poisson (M)
- Temps de service suivant une distribution GÉNÉRALE (G)
- Un seul serveur (1)
- Capacité infinie
- Discipline FIFO

Ce modèle est le plus flexible car il accepte n'importe quelle
distribution de service. On utilise la formule de Pollaczek-Khinchin.

Formules théoriques (Pollaczek-Khinchin):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│ Paramètre             │ Description                               │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│ E[S] = 1/μ            │ Espérance du temps de service             │
│ Var[S] = σ²           │ Variance du temps de service              │
│ C²s = σ²μ²            │ Coefficient de variation au carré         │
│ ρ = λ/μ               │ Facteur d'utilisation                     │
│                       │                                           │
│ FORMULE P-K:          │                                           │
│ Lq = (λ²σ² + ρ²)      │                                           │
│      ─────────────    │                                           │
│        2(1-ρ)         │                                           │
│                       │                                           │
│ Équivalent:           │                                           │
│ Lq = ρ²(1 + C²s)      │                                           │
│      ───────────      │                                           │
│        2(1-ρ)         │                                           │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cas particuliers:
- C²s = 1 (exponentiel) → M/M/1
- C²s = 0 (constant) → M/D/1 (Lq divisé par 2)
- C²s > 1 (haute variance) → pire que M/M/1

Application à la moulinette:
- Permet de modéliser des temps de service réalistes
- On peut ajuster la variance pour correspondre aux données réelles
- Distributions supportées: Exponentielle, Uniforme, Normale tronquée, etc.

Auteurs: ERO2 Team Markov Moulinette Configurators - EPITA
"""

import numpy as np
from typing import Optional, Callable, Tuple
from enum import Enum
from .base_queue import BaseQueueModel, QueueMetrics, SimulationResults


class ServiceDistribution(Enum):
    """Distributions de service supportées."""
    EXPONENTIAL = "exponential"      # M/M/1: C²s = 1
    DETERMINISTIC = "deterministic"  # M/D/1: C²s = 0
    UNIFORM = "uniform"              # C²s = 1/3
    ERLANG = "erlang"                # C²s = 1/k
    HYPEREXPONENTIAL = "hyperexponential"  # C²s > 1
    LOGNORMAL = "lognormal"          # Modèle réaliste
    CUSTOM = "custom"                # Distribution personnalisée


class MG1Queue(BaseQueueModel):
    """
    Implémentation du modèle M/G/1 (General Service).
    
    Cas d'usage pour la moulinette:
    - Modélisation réaliste avec données empiriques
    - Permet d'ajuster la variance du temps de service
    - Compare l'impact de différentes distributions
    
    Exemple:
        >>> # Service avec distribution lognormale (réaliste)
        >>> queue = MG1Queue(
        ...     lambda_rate=10, 
        ...     service_mean=0.1,
        ...     service_variance=0.02,
        ...     distribution=ServiceDistribution.LOGNORMAL
        ... )
        >>> metrics = queue.get_theoretical_metrics()
    """
    
    def __init__(
        self,
        lambda_rate: float,
        service_mean: float,
        service_variance: float = None,
        distribution: ServiceDistribution = ServiceDistribution.EXPONENTIAL,
        distribution_params: dict = None,
        seed: Optional[int] = None
    ):
        """
        Initialise une file M/G/1.
        
        Args:
            lambda_rate: Taux d'arrivée λ
            service_mean: Temps de service moyen E[S] = 1/μ
            service_variance: Variance du temps de service Var[S]
            distribution: Type de distribution de service
            distribution_params: Paramètres additionnels pour la distribution
            seed: Graine aléatoire
        """
        # Calculer μ depuis la moyenne
        mu_rate = 1 / service_mean
        
        super().__init__(lambda_rate, mu_rate, c=1, K=None, seed=seed)
        
        self.service_mean = service_mean
        self.distribution = distribution
        self.distribution_params = distribution_params or {}
        
        # Calculer ou définir la variance
        if service_variance is None:
            self.service_variance = self._default_variance(distribution, service_mean)
        else:
            self.service_variance = service_variance
        
        # Coefficient de variation au carré
        self.C_squared = self.service_variance / (service_mean ** 2)
        
        if not self.is_stable:
            raise ValueError(
                f"Système instable: ρ = {self.rho:.4f} ≥ 1"
            )
        
        # Configurer le générateur de service
        self._service_generator = self._create_service_generator()
    
    def _default_variance(
        self,
        dist: ServiceDistribution,
        mean: float
    ) -> float:
        """Calcule la variance par défaut selon la distribution."""
        if dist == ServiceDistribution.EXPONENTIAL:
            return mean ** 2  # Var = 1/μ² pour exp(μ)
        elif dist == ServiceDistribution.DETERMINISTIC:
            return 0.0
        elif dist == ServiceDistribution.UNIFORM:
            # Pour U[a,b] avec moyenne mean: Var = (b-a)²/12
            # Si a=0, b=2*mean: Var = mean²/3
            return (mean ** 2) / 3
        elif dist == ServiceDistribution.ERLANG:
            k = self.distribution_params.get('k', 2)
            return (mean ** 2) / k
        else:
            return mean ** 2  # Par défaut comme exponentielle
    
    def _create_service_generator(self) -> Callable[[int], np.ndarray]:
        """Crée le générateur de temps de service selon la distribution."""
        mean = self.service_mean
        var = self.service_variance
        dist = self.distribution
        
        if dist == ServiceDistribution.EXPONENTIAL:
            return lambda n: self.rng.exponential(mean, n)
        
        elif dist == ServiceDistribution.DETERMINISTIC:
            return lambda n: np.full(n, mean)
        
        elif dist == ServiceDistribution.UNIFORM:
            # U[a,b] avec E=mean, Var=var
            # (b-a)²/12 = var → b-a = sqrt(12*var)
            width = np.sqrt(12 * var)
            a = mean - width / 2
            b = mean + width / 2
            a = max(0, a)  # Temps positifs
            return lambda n: self.rng.uniform(a, b, n)
        
        elif dist == ServiceDistribution.ERLANG:
            k = self.distribution_params.get('k', 2)
            # Erlang(k, θ): E=kθ, Var=kθ²
            theta = mean / k
            return lambda n: self.rng.gamma(k, theta, n)
        
        elif dist == ServiceDistribution.HYPEREXPONENTIAL:
            # Mélange de deux exponentielles
            p = self.distribution_params.get('p', 0.5)
            mu1 = self.distribution_params.get('mu1', 1/mean * 0.5)
            mu2 = self.distribution_params.get('mu2', 1/mean * 1.5)
            
            def gen(n):
                choices = self.rng.random(n) < p
                result = np.zeros(n)
                result[choices] = self.rng.exponential(1/mu1, choices.sum())
                result[~choices] = self.rng.exponential(1/mu2, (~choices).sum())
                return result
            
            return gen
        
        elif dist == ServiceDistribution.LOGNORMAL:
            # Lognormal: si X ~ LogN(μ, σ²), alors E[X] = exp(μ + σ²/2)
            # et Var[X] = (exp(σ²) - 1) * exp(2μ + σ²)
            # On résout pour μ et σ à partir de mean et var
            sigma_sq = np.log(1 + var / (mean ** 2))
            mu = np.log(mean) - sigma_sq / 2
            sigma = np.sqrt(sigma_sq)
            return lambda n: self.rng.lognormal(mu, sigma, n)
        
        else:  # CUSTOM ou autre
            return lambda n: self.rng.exponential(mean, n)
    
    def _get_kendall_notation(self) -> str:
        dist_letter = {
            ServiceDistribution.EXPONENTIAL: "M",
            ServiceDistribution.DETERMINISTIC: "D",
            ServiceDistribution.UNIFORM: "U",
            ServiceDistribution.ERLANG: f"E_{self.distribution_params.get('k', 2)}",
            ServiceDistribution.HYPEREXPONENTIAL: "H",
            ServiceDistribution.LOGNORMAL: "LN",
            ServiceDistribution.CUSTOM: "G"
        }
        s = dist_letter.get(self.distribution, "G")
        return f"M/{s}/1"
    
    def _get_model_description(self) -> str:
        return f"""
        ╔══════════════════════════════════════════════════════════════════╗
        ║                        MODÈLE M/G/1                              ║
        ╠══════════════════════════════════════════════════════════════════╣
        ║ Description:                                                     ║
        ║   File d'attente avec distribution de service GÉNÉRALE.          ║
        ║                                                                  ║
        ║ Configuration actuelle:                                          ║
        ║   • Distribution: {self.distribution.value:<30}                  ║
        ║   • E[S] = {self.service_mean:.4f} (temps moyen de service)      ║
        ║   • Var[S] = {self.service_variance:.4f} (variance)              ║
        ║   • C²s = {self.C_squared:.4f} (coefficient de variation²)       ║
        ║                                                                  ║
        ║ Formule de Pollaczek-Khinchin:                                   ║
        ║   Lq = ρ²(1 + C²s) / (2(1-ρ))                                    ║
        ║                                                                  ║
        ║ Impact de C²s:                                                   ║
        ║   • C²s = 0 (M/D/1): Lq minimal                                  ║
        ║   • C²s = 1 (M/M/1): Référence                                   ║
        ║   • C²s > 1: Haute variance, Lq augmenté                         ║
        ║                                                                  ║
        ║ Application moulinette:                                          ║
        ║   Permet de calibrer le modèle avec des données réelles.         ║
        ╚══════════════════════════════════════════════════════════════════╝
        """
    
    def compute_theoretical_metrics(self) -> QueueMetrics:
        """
        Calcule les métriques via la formule de Pollaczek-Khinchin.
        
        P-K: Lq = (λ²σ² + ρ²) / (2(1-ρ))
           = ρ²(1 + C²s) / (2(1-ρ))
        
        Returns:
            QueueMetrics avec métriques P-K
        """
        lambda_rate = self.lambda_rate
        rho = self.rho
        C_sq = self.C_squared
        mean = self.service_mean
        
        # Formule de Pollaczek-Khinchin
        Lq = (rho ** 2) * (1 + C_sq) / (2 * (1 - rho))
        
        # Nombre moyen en service = ρ
        Ls = rho
        
        # Nombre moyen total
        L = Lq + Ls
        
        # Temps moyens
        Wq = Lq / lambda_rate
        Ws = mean
        W = Wq + Ws
        
        # Probabilité système vide
        P0 = 1 - rho
        
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
            state_probabilities=np.array([P0, rho])
        )
    
    def _generate_service_times(self, n: int) -> np.ndarray:
        """Génère n temps de service selon la distribution configurée."""
        return self._service_generator(n)
    
    def simulate(
        self,
        n_customers: int = 1000,
        max_time: Optional[float] = None
    ) -> SimulationResults:
        """
        Simule une file M/G/1.
        
        Args:
            n_customers: Nombre de clients
            max_time: Temps maximum
            
        Returns:
            SimulationResults avec traces
        """
        # Générer temps
        interarrival_times = self._generate_interarrival_times(n_customers)
        service_times = self._generate_service_times(n_customers)
        arrival_times = np.cumsum(interarrival_times)
        
        # Filtrer
        if max_time is not None:
            mask = arrival_times <= max_time
            arrival_times = arrival_times[mask]
            service_times = service_times[:len(arrival_times)]
            n_customers = len(arrival_times)
        
        if n_customers == 0:
            return SimulationResults()
        
        # Simulation
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
        
        # Traces
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
        """Construit les traces temporelles."""
        events = []
        for t in arrival_times:
            events.append((t, 1))
        for t in departure_times:
            events.append((t, -1))
        
        events.sort(key=lambda x: (x[0], -x[1]))
        
        time_trace = [0.0]
        system_trace = [0]
        queue_trace = [0]
        
        current = 0
        
        for time, delta in events:
            current += delta
            time_trace.append(time)
            system_trace.append(current)
            queue_trace.append(max(0, current - 1))
        
        return np.array(time_trace), np.array(queue_trace), np.array(system_trace)
    
    def compare_distributions(
        self,
        n_simulations: int = 30,
        n_customers: int = 1000
    ) -> dict:
        """
        Compare les performances de différentes distributions de service.
        
        Args:
            n_simulations: Nombre de simulations par distribution
            n_customers: Clients par simulation
            
        Returns:
            Dict avec comparaison des métriques
        """
        results = {}
        
        distributions = [
            (ServiceDistribution.DETERMINISTIC, {}),
            (ServiceDistribution.EXPONENTIAL, {}),
            (ServiceDistribution.ERLANG, {'k': 2}),
            (ServiceDistribution.ERLANG, {'k': 5}),
            (ServiceDistribution.UNIFORM, {}),
        ]
        
        for dist, params in distributions:
            queue = MG1Queue(
                self.lambda_rate,
                self.service_mean,
                distribution=dist,
                distribution_params=params,
                seed=self.seed
            )
            
            theo = queue.get_theoretical_metrics()
            
            # Simulations
            wq_values = []
            for i in range(n_simulations):
                queue.rng = np.random.default_rng(self.seed + i if self.seed else None)
                sim = queue.simulate(n_customers)
                if sim.empirical_metrics:
                    wq_values.append(sim.empirical_metrics.Wq)
            
            key = f"{dist.value}" + (f"_k{params.get('k', '')}" if 'k' in params else "")
            results[key] = {
                'C_squared': queue.C_squared,
                'Wq_theoretical': theo.Wq,
                'Wq_simulated_mean': np.mean(wq_values) if wq_values else 0,
                'Wq_simulated_std': np.std(wq_values) if wq_values else 0,
                'Lq_theoretical': theo.Lq
            }
        
        return results
