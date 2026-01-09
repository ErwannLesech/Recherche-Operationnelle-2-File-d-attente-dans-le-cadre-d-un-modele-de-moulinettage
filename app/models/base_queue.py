"""
Module de base pour les modèles de files d'attente.

Ce module définit la classe abstraite BaseQueueModel qui sert de fondation
pour tous les modèles de files d'attente implémentés (M/M/1, M/M/c, M/M/c/K, M/D/1, M/G/1).

Notation de Kendall: A/S/c/K/N/D
- A : Distribution des inter-arrivées (M=Markovien/Poisson, D=Déterministe, G=Général)
- S : Distribution des temps de service (M, D, G)
- c : Nombre de serveurs
- K : Capacité du système (∞ si non spécifié)
- N : Taille de la population source (∞ si non spécifié)
- D : Discipline de service (FIFO par défaut)

Auteurs: ERO2 Team Markov Moulinette Configurators - EPITA
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
import numpy as np


@dataclass
class QueueMetrics:
    """
    Métriques de performance d'une file d'attente.
    
    Attributs théoriques basés sur la théorie des files d'attente:
    - rho (ρ): Facteur d'utilisation = λ/(cμ)
    - L: Nombre moyen de clients dans le système (E[N])
    - Lq: Nombre moyen de clients en file d'attente
    - Ls: Nombre moyen de clients en service
    - W: Temps moyen de séjour dans le système (E[T])
    - Wq: Temps moyen d'attente en file
    - Ws: Temps moyen de service
    - P0: Probabilité que le système soit vide
    - Pk: Probabilité de blocage (pour files à capacité finie)
    
    Relations fondamentales (Loi de Little):
    - L = λ * W
    - Lq = λ * Wq
    """
    rho: float = 0.0           # Facteur d'utilisation ρ = λ/(cμ)
    L: float = 0.0             # Nombre moyen dans le système
    Lq: float = 0.0            # Nombre moyen en file
    Ls: float = 0.0            # Nombre moyen en service
    W: float = 0.0             # Temps moyen dans le système
    Wq: float = 0.0            # Temps moyen d'attente
    Ws: float = 0.0            # Temps moyen de service (= 1/μ)
    P0: float = 0.0            # Probabilité système vide
    Pk: float = 0.0            # Probabilité de blocage
    lambda_eff: float = 0.0    # Taux d'arrivée effectif
    throughput: float = 0.0    # Débit effectif du système
    
    # Distributions de probabilité
    state_probabilities: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class SimulationResults:
    """
    Résultats d'une simulation de file d'attente.
    
    Contient toutes les traces temporelles et statistiques
    issues de la simulation Monte Carlo.
    """
    # Temps
    arrival_times: np.ndarray = field(default_factory=lambda: np.array([]))     # T_n
    service_start_times: np.ndarray = field(default_factory=lambda: np.array([]))  # B_n
    departure_times: np.ndarray = field(default_factory=lambda: np.array([]))    # D_n
    service_times: np.ndarray = field(default_factory=lambda: np.array([]))      # S_k
    
    # Métriques par client
    waiting_times: np.ndarray = field(default_factory=lambda: np.array([]))      # W_q = B_n - T_n
    system_times: np.ndarray = field(default_factory=lambda: np.array([]))       # W = D_n - T_n
    
    # Compteurs
    n_arrivals: int = 0
    n_served: int = 0
    n_rejected: int = 0
    
    # Traces temporelles pour visualisation
    time_trace: np.ndarray = field(default_factory=lambda: np.array([]))
    queue_length_trace: np.ndarray = field(default_factory=lambda: np.array([]))
    system_size_trace: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Métriques empiriques
    empirical_metrics: Optional[QueueMetrics] = None


class BaseQueueModel(ABC):
    """
    Classe de base abstraite pour tous les modèles de files d'attente.
    
    Cette classe définit l'interface commune et les méthodes partagées
    par tous les modèles de files d'attente suivant la notation de Kendall.
    
    Paramètres fondamentaux:
    - lambda_rate (λ): Taux d'arrivée des clients (clients/unité de temps)
    - mu_rate (μ): Taux de service par serveur (clients servis/unité de temps)
    - c: Nombre de serveurs
    - K: Capacité maximale du système (None = infinie)
    
    Condition de stabilité (pour files infinies):
    ρ = λ/(cμ) < 1
    """
    
    def __init__(
        self,
        lambda_rate: float,
        mu_rate: float,
        c: int = 1,
        K: Optional[int] = None,
        seed: Optional[int] = None
    ):
        """
        Initialise le modèle de file d'attente.
        
        Args:
            lambda_rate: Taux d'arrivée λ (> 0)
            mu_rate: Taux de service μ par serveur (> 0)
            c: Nombre de serveurs (≥ 1)
            K: Capacité du système (None = infinie)
            seed: Graine pour le générateur aléatoire
        """
        self._validate_parameters(lambda_rate, mu_rate, c, K)
        
        self.lambda_rate = lambda_rate
        self.mu_rate = mu_rate
        self.c = c
        self.K = K
        self.seed = seed
        
        # Générateur aléatoire reproductible
        self.rng = np.random.default_rng(seed)
        
        # Cache pour les métriques théoriques
        self._theoretical_metrics: Optional[QueueMetrics] = None
    
    def _validate_parameters(
        self,
        lambda_rate: float,
        mu_rate: float,
        c: int,
        K: Optional[int]
    ) -> None:
        """Valide les paramètres d'entrée."""
        if lambda_rate <= 0:
            raise ValueError(f"λ doit être > 0, reçu: {lambda_rate}")
        if mu_rate <= 0:
            raise ValueError(f"μ doit être > 0, reçu: {mu_rate}")
        if c < 1:
            raise ValueError(f"c doit être ≥ 1, reçu: {c}")
        if K is not None and K < c:
            raise ValueError(f"K doit être ≥ c, reçu: K={K}, c={c}")
    
    @property
    def rho(self) -> float:
        """
        Facteur d'utilisation ρ = λ/(cμ).
        
        Représente la fraction du temps où les serveurs sont occupés.
        Pour un système stable avec capacité infinie: ρ < 1
        """
        return self.lambda_rate / (self.c * self.mu_rate)
    
    @property
    def is_stable(self) -> bool:
        """
        Vérifie si le système est stable (ρ < 1).
        
        Pour les files à capacité finie (K < ∞), le système est toujours stable.
        Pour les files à capacité infinie, ρ < 1 est nécessaire.
        """
        if self.K is not None:
            return True  # Files à capacité finie sont toujours stables
        return self.rho < 1
    
    @property
    def kendall_notation(self) -> str:
        """Retourne la notation de Kendall du modèle."""
        return self._get_kendall_notation()
    
    @abstractmethod
    def _get_kendall_notation(self) -> str:
        """Retourne la notation de Kendall spécifique au modèle."""
        pass
    
    @abstractmethod
    def compute_theoretical_metrics(self) -> QueueMetrics:
        """
        Calcule les métriques théoriques du modèle.
        
        Returns:
            QueueMetrics contenant toutes les métriques analytiques
        """
        pass
    
    @abstractmethod
    def simulate(
        self,
        n_customers: int = 1000,
        max_time: Optional[float] = None
    ) -> SimulationResults:
        """
        Exécute une simulation Monte Carlo de la file d'attente.
        
        Args:
            n_customers: Nombre de clients à simuler
            max_time: Temps maximum de simulation (optionnel)
            
        Returns:
            SimulationResults contenant toutes les traces de simulation
        """
        pass
    
    def get_theoretical_metrics(self) -> QueueMetrics:
        """
        Retourne les métriques théoriques (avec cache).
        """
        if self._theoretical_metrics is None:
            self._theoretical_metrics = self.compute_theoretical_metrics()
        return self._theoretical_metrics
    
    def _generate_interarrival_times(self, n: int) -> np.ndarray:
        """
        Génère n temps inter-arrivées selon la loi exponentielle.
        
        Pour un processus de Poisson de taux λ:
        Q_k ~ Exp(λ), donc E[Q_k] = 1/λ
        """
        return self.rng.exponential(scale=1/self.lambda_rate, size=n)
    
    def _generate_service_times(self, n: int) -> np.ndarray:
        """
        Génère n temps de service selon la loi exponentielle.
        
        Pour un service exponentiel de taux μ:
        S_k ~ Exp(μ), donc E[S_k] = 1/μ
        """
        return self.rng.exponential(scale=1/self.mu_rate, size=n)
    
    def compute_empirical_metrics(
        self,
        results: SimulationResults
    ) -> QueueMetrics:
        """
        Calcule les métriques empiriques à partir des résultats de simulation.
        
        Utilise les estimateurs suivants:
        - Ŵ = mean(W_i) : Temps moyen de séjour
        - Ŵq = mean(Wq_i) : Temps moyen d'attente
        - L̂ = (1/T) ∫₀ᵀ N(t) dt : Nombre moyen dans le système
        """
        metrics = QueueMetrics()
        
        if len(results.system_times) > 0:
            # Temps moyens
            metrics.W = np.mean(results.system_times)
            metrics.Wq = np.mean(results.waiting_times)
            metrics.Ws = np.mean(results.service_times[:len(results.system_times)])
            
            # Loi de Little pour estimer L et Lq
            lambda_eff = results.n_served / results.departure_times[-1] if len(results.departure_times) > 0 else 0
            metrics.lambda_eff = lambda_eff
            metrics.L = lambda_eff * metrics.W
            metrics.Lq = lambda_eff * metrics.Wq
            metrics.Ls = lambda_eff * metrics.Ws
            
            # Utilisation
            if results.n_arrivals > 0:
                metrics.rho = results.n_served / results.n_arrivals * self.rho
            
            # Taux de rejet
            if results.n_arrivals > 0:
                metrics.Pk = results.n_rejected / results.n_arrivals
            
            metrics.throughput = lambda_eff
        
        return metrics
    
    def run_multiple_simulations(
        self,
        n_runs: int = 30,
        n_customers: int = 1000
    ) -> Tuple[List[SimulationResults], Dict[str, Tuple[float, float]]]:
        """
        Exécute plusieurs simulations pour estimer variance et intervalles de confiance.
        
        Args:
            n_runs: Nombre de répétitions
            n_customers: Nombre de clients par simulation
            
        Returns:
            Tuple (liste des résultats, dict des (moyenne, écart-type) par métrique)
        """
        results_list = []
        metrics_arrays = {
            'W': [], 'Wq': [], 'L': [], 'Lq': [], 'rho': [], 'Pk': []
        }
        
        for i in range(n_runs):
            # Changer la seed pour chaque run
            self.rng = np.random.default_rng(self.seed + i if self.seed else None)
            
            result = self.simulate(n_customers)
            results_list.append(result)
            
            if result.empirical_metrics:
                for key in metrics_arrays:
                    metrics_arrays[key].append(getattr(result.empirical_metrics, key))
        
        # Calculer moyenne et écart-type
        stats = {}
        for key, values in metrics_arrays.items():
            if values:
                stats[key] = (np.mean(values), np.std(values))
            else:
                stats[key] = (0.0, 0.0)
        
        return results_list, stats
    
    def get_model_description(self) -> str:
        """Retourne une description détaillée du modèle."""
        return self._get_model_description()
    
    @abstractmethod
    def _get_model_description(self) -> str:
        """Retourne la description spécifique au modèle."""
        pass
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"λ={self.lambda_rate}, μ={self.mu_rate}, "
            f"c={self.c}, K={self.K}, ρ={self.rho:.4f})"
        )
