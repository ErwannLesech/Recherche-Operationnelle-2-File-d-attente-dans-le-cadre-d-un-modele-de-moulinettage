"""
Module d'optimisation des coûts.

Ce module implémente l'optimisation coût/qualité de service
pour le dimensionnement de la moulinette.

Fonction objectif:
══════════════════════════════════════════════════════════════
    min [ α × E[T] + β × Coût(K, μ, c) ]    avec α + β = 1
══════════════════════════════════════════════════════════════

Où:
- E[T] = Temps moyen de réponse (séjour dans le système)
- Coût = Coût des serveurs + Coût des rejets + Coût d'insatisfaction
- α = Poids accordé à la performance
- β = Poids accordé au coût

Ce trade-off permet de trouver le point optimal entre:
- Trop de serveurs → Coût élevé mais bon service
- Peu de serveurs → Coût faible mais mauvais service

L'optimisation explore l'espace (K, μ, c) pour minimiser
la fonction objectif sous contraintes.

Auteurs: ERO2 Team Markov Moulinette Configurators - EPITA
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from enum import Enum

from ..models import MMcKQueue, MMcQueue
from ..models.base_queue import QueueMetrics


class CostType(Enum):
    """Types de coûts à considérer."""
    SERVER = "server"           # Coût des serveurs
    REJECTION = "rejection"     # Coût des rejets
    WAITING = "waiting"         # Coût d'attente (insatisfaction)
    INFRASTRUCTURE = "infra"    # Coût d'infrastructure fixe


@dataclass
class CostModel:
    """
    Modèle de coûts pour l'optimisation.
    
    Définit tous les paramètres de coût pour
    l'analyse économique du système.
    """
    # Coûts des serveurs
    cost_per_server_hour: float = 0.50          # €/heure/serveur
    server_startup_cost: float = 0.10           # €/démarrage serveur
    server_min_uptime_hours: float = 0.5        # Durée min avant arrêt
    
    # Coûts des rejets
    cost_per_rejection: float = 0.05            # €/rejet direct
    reputation_cost_factor: float = 0.01        # €/rejet (impact réputation)
    
    # Coûts d'attente (insatisfaction étudiant)
    cost_per_waiting_minute: float = 0.001      # €/minute d'attente
    waiting_threshold_minutes: float = 10.0      # Seuil acceptable
    excess_waiting_penalty: float = 0.01        # €/minute au-delà du seuil
    
    # Coûts d'infrastructure
    fixed_infrastructure_cost: float = 10.0     # €/jour fixe
    buffer_cost_per_slot: float = 0.001         # €/slot de buffer
    
    # Contraintes
    max_budget_per_hour: float = 50.0           # Budget max horaire
    max_total_budget: float = 1000.0            # Budget total max
    
    def compute_server_cost(
        self,
        n_servers: int,
        hours: float,
        n_startups: int = 1
    ) -> float:
        """
        Calcule le coût des serveurs.
        
        Coût = n × t × c_hourly + n_start × c_startup
        
        Args:
            n_servers: Nombre de serveurs
            hours: Durée de fonctionnement
            n_startups: Nombre de démarrages
            
        Returns:
            Coût total des serveurs
        """
        running_cost = n_servers * hours * self.cost_per_server_hour
        startup_cost = n_startups * self.server_startup_cost
        return running_cost + startup_cost
    
    def compute_rejection_cost(
        self,
        n_rejections: int,
        total_requests: int
    ) -> float:
        """
        Calcule le coût des rejets.
        
        Inclut le coût direct et l'impact réputation.
        
        Args:
            n_rejections: Nombre de rejets
            total_requests: Nombre total de requêtes
            
        Returns:
            Coût total des rejets
        """
        direct_cost = n_rejections * self.cost_per_rejection
        
        # Coût de réputation (non-linéaire)
        rejection_rate = n_rejections / total_requests if total_requests > 0 else 0
        reputation_cost = (rejection_rate ** 2) * total_requests * self.reputation_cost_factor
        
        return direct_cost + reputation_cost
    
    def compute_waiting_cost(
        self,
        avg_waiting_minutes: float,
        n_customers: int
    ) -> float:
        """
        Calcule le coût d'attente (insatisfaction).
        
        Modèle avec seuil: coût linéaire puis pénalité au-delà.
        
        Args:
            avg_waiting_minutes: Temps d'attente moyen
            n_customers: Nombre de clients
            
        Returns:
            Coût total d'insatisfaction
        """
        if avg_waiting_minutes <= self.waiting_threshold_minutes:
            # En dessous du seuil: coût linéaire faible
            return avg_waiting_minutes * n_customers * self.cost_per_waiting_minute
        else:
            # Au-delà du seuil: pénalité
            base_cost = self.waiting_threshold_minutes * n_customers * self.cost_per_waiting_minute
            excess = avg_waiting_minutes - self.waiting_threshold_minutes
            penalty = excess * n_customers * self.excess_waiting_penalty
            return base_cost + penalty
    
    def compute_total_cost(
        self,
        n_servers: int,
        hours: float,
        buffer_size: int,
        n_rejections: int,
        total_requests: int,
        avg_waiting_minutes: float
    ) -> Dict[str, float]:
        """
        Calcule le coût total avec détail par type.
        
        Returns:
            Dict avec coût par catégorie et total
        """
        server_cost = self.compute_server_cost(n_servers, hours)
        rejection_cost = self.compute_rejection_cost(n_rejections, total_requests)
        waiting_cost = self.compute_waiting_cost(avg_waiting_minutes, total_requests - n_rejections)
        infra_cost = self.fixed_infrastructure_cost * (hours / 24) + buffer_size * self.buffer_cost_per_slot
        
        return {
            'server': server_cost,
            'rejection': rejection_cost,
            'waiting': waiting_cost,
            'infrastructure': infra_cost,
            'total': server_cost + rejection_cost + waiting_cost + infra_cost
        }


@dataclass
class OptimizationResult:
    """
    Résultat d'une optimisation.
    
    Contient la configuration optimale et les métriques associées.
    """
    # Configuration optimale
    optimal_servers: int = 0
    optimal_buffer: int = 0
    optimal_service_rate: float = 0.0
    
    # Métriques à l'optimum
    utilization: float = 0.0
    avg_waiting_time: float = 0.0
    rejection_rate: float = 0.0
    
    # Coûts
    costs: Dict[str, float] = field(default_factory=dict)
    
    # Objectif
    objective_value: float = 0.0
    alpha: float = 0.5  # Poids performance
    beta: float = 0.5   # Poids coût
    
    # Détails d'optimisation
    n_iterations: int = 0
    converged: bool = False
    pareto_front: List[Tuple[float, float]] = field(default_factory=list)
    
    # Sensibilité
    sensitivity: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire."""
        return {
            'optimal_servers': self.optimal_servers,
            'optimal_buffer': self.optimal_buffer,
            'optimal_service_rate': self.optimal_service_rate,
            'utilization': self.utilization,
            'avg_waiting_time': self.avg_waiting_time,
            'rejection_rate': self.rejection_rate,
            'costs': self.costs,
            'objective_value': self.objective_value,
            'alpha': self.alpha,
            'beta': self.beta,
            'converged': self.converged
        }


class CostOptimizer:
    """
    Optimiseur de coûts pour le système de moulinette.
    
    Implémente l'optimisation multi-objectif coût/performance:
    
    min [ α × E[T] + β × Coût ]    s.t. contraintes
    
    Méthodes disponibles:
    - Grid search exhaustif
    - Gradient descent (approx)
    - Pareto front analysis
    
    Exemple:
        >>> optimizer = CostOptimizer(
        ...     lambda_rate=100,
        ...     mu_rate=10,
        ...     cost_model=CostModel()
        ... )
        >>> result = optimizer.optimize(alpha=0.6)  # Priorité performance
        >>> print(f"Serveurs optimaux: {result.optimal_servers}")
    """
    
    def __init__(
        self,
        lambda_rate: float,
        mu_rate: float,
        cost_model: Optional[CostModel] = None,
        hours: float = 24.0
    ):
        """
        Initialise l'optimiseur.
        
        Args:
            lambda_rate: Taux d'arrivée moyen
            mu_rate: Taux de service par serveur
            cost_model: Modèle de coûts
            hours: Durée de la période d'analyse
        """
        self.lambda_rate = lambda_rate
        self.mu_rate = mu_rate
        self.cost_model = cost_model or CostModel()
        self.hours = hours
    
    def objective_function(
        self,
        c: int,
        K: int,
        alpha: float = 0.5,
        normalize: bool = True
    ) -> float:
        """
        Fonction objectif à minimiser.
        
        f(c, K) = α × E[T]_norm + β × Coût_norm
        
        Args:
            c: Nombre de serveurs
            K: Taille du buffer
            alpha: Poids de la performance (β = 1 - α)
            normalize: Normaliser les valeurs
            
        Returns:
            Valeur de l'objectif
        """
        beta = 1 - alpha
        
        try:
            # Créer le modèle
            queue = MMcKQueue(self.lambda_rate, self.mu_rate, c, K)
            metrics = queue.compute_theoretical_metrics()
            
            # Performance: temps moyen de séjour (en minutes)
            W_minutes = metrics.W * 60
            
            # Coût
            n_requests = self.lambda_rate * self.hours
            n_rejections = n_requests * metrics.Pk
            n_served = n_requests - n_rejections
            
            costs = self.cost_model.compute_total_cost(
                n_servers=c,
                hours=self.hours,
                buffer_size=K,
                n_rejections=int(n_rejections),
                total_requests=int(n_requests),
                avg_waiting_minutes=metrics.Wq * 60
            )
            
            cost = costs['total']
            
            if normalize:
                # Normalisation pour que les deux termes soient comparables
                # Temps: normaliser par 60 min (1h acceptable)
                W_norm = W_minutes / 60
                # Coût: normaliser par le budget horaire max
                cost_norm = cost / (self.cost_model.max_budget_per_hour * self.hours)
            else:
                W_norm = W_minutes
                cost_norm = cost
            
            return alpha * W_norm + beta * cost_norm
            
        except Exception:
            return float('inf')
    
    def optimize(
        self,
        alpha: float = 0.5,
        c_range: Tuple[int, int] = (1, 20),
        K_range: Tuple[int, int] = (10, 200),
        method: str = 'grid'
    ) -> OptimizationResult:
        """
        Trouve la configuration optimale.
        
        Args:
            alpha: Poids de la performance (0 = que coût, 1 = que perf)
            c_range: Intervalle de recherche pour c
            K_range: Intervalle de recherche pour K
            method: Méthode d'optimisation ('grid', 'gradient')
            
        Returns:
            OptimizationResult avec configuration optimale
        """
        if method == 'grid':
            return self._optimize_grid(alpha, c_range, K_range)
        else:
            return self._optimize_grid(alpha, c_range, K_range)
    
    def _optimize_grid(
        self,
        alpha: float,
        c_range: Tuple[int, int],
        K_range: Tuple[int, int]
    ) -> OptimizationResult:
        """
        Optimisation par recherche exhaustive sur grille.
        """
        best_obj = float('inf')
        best_c, best_K = c_range[0], K_range[0]
        
        n_iterations = 0
        
        # Grille de recherche
        c_values = range(c_range[0], c_range[1] + 1)
        K_values = range(K_range[0], K_range[1] + 1, 10)  # Pas de 10 pour K
        
        for c in c_values:
            for K in K_values:
                n_iterations += 1
                obj = self.objective_function(c, K, alpha)
                
                if obj < best_obj:
                    best_obj = obj
                    best_c, best_K = c, K
        
        # Affiner K autour du meilleur
        for K in range(max(K_range[0], best_K - 10), min(K_range[1], best_K + 10) + 1):
            n_iterations += 1
            obj = self.objective_function(best_c, K, alpha)
            if obj < best_obj:
                best_obj = obj
                best_K = K
        
        # Calculer les métriques à l'optimum
        result = self._compute_result(best_c, best_K, alpha, best_obj, n_iterations)
        
        return result
    
    def _compute_result(
        self,
        c: int,
        K: int,
        alpha: float,
        obj_value: float,
        n_iterations: int
    ) -> OptimizationResult:
        """
        Calcule le résultat complet pour une configuration.
        """
        queue = MMcKQueue(self.lambda_rate, self.mu_rate, c, K)
        metrics = queue.compute_theoretical_metrics()
        
        n_requests = self.lambda_rate * self.hours
        n_rejections = n_requests * metrics.Pk
        
        costs = self.cost_model.compute_total_cost(
            n_servers=c,
            hours=self.hours,
            buffer_size=K,
            n_rejections=int(n_rejections),
            total_requests=int(n_requests),
            avg_waiting_minutes=metrics.Wq * 60
        )
        
        return OptimizationResult(
            optimal_servers=c,
            optimal_buffer=K,
            optimal_service_rate=self.mu_rate,
            utilization=metrics.rho,
            avg_waiting_time=metrics.W * 60,  # minutes
            rejection_rate=metrics.Pk,
            costs=costs,
            objective_value=obj_value,
            alpha=alpha,
            beta=1 - alpha,
            n_iterations=n_iterations,
            converged=True
        )
    
    def compute_pareto_front(
        self,
        c_range: Tuple[int, int] = (1, 20),
        K_range: Tuple[int, int] = (10, 200),
        n_points: int = 20
    ) -> List[OptimizationResult]:
        """
        Calcule le front de Pareto coût/performance.
        
        Le front de Pareto contient les solutions non-dominées:
        aucune autre solution n'est meilleure sur les deux critères.
        
        Args:
            c_range: Intervalle pour c
            K_range: Intervalle pour K
            n_points: Nombre de points à calculer
            
        Returns:
            Liste des solutions du front de Pareto
        """
        # Varier alpha de 0 à 1
        alphas = np.linspace(0.05, 0.95, n_points)
        
        results = []
        for alpha in alphas:
            result = self.optimize(alpha, c_range, K_range)
            results.append(result)
        
        # Filtrer pour ne garder que le front de Pareto
        pareto_results = []
        for r in results:
            is_dominated = False
            for other in results:
                if other is r:
                    continue
                # other domine r si meilleur sur les deux critères
                if (other.avg_waiting_time <= r.avg_waiting_time and
                    other.costs['total'] <= r.costs['total'] and
                    (other.avg_waiting_time < r.avg_waiting_time or
                     other.costs['total'] < r.costs['total'])):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_results.append(r)
        
        return pareto_results
    
    def sensitivity_analysis(
        self,
        base_c: int,
        base_K: int,
        alpha: float = 0.5,
        perturbation: float = 0.1
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyse de sensibilité des paramètres.
        
        Mesure l'impact d'une variation de chaque paramètre
        sur la fonction objectif.
        
        Args:
            base_c: Configuration de base (serveurs)
            base_K: Configuration de base (buffer)
            alpha: Poids
            perturbation: Amplitude de la perturbation (%)
            
        Returns:
            Dict avec sensibilité par paramètre
        """
        base_obj = self.objective_function(base_c, base_K, alpha)
        
        sensitivities = {}
        
        # Sensibilité au nombre de serveurs
        delta_c = max(1, int(base_c * perturbation))
        obj_plus = self.objective_function(base_c + delta_c, base_K, alpha)
        obj_minus = self.objective_function(max(1, base_c - delta_c), base_K, alpha)
        
        sensitivities['n_servers'] = {
            'base': base_c,
            'delta': delta_c,
            'obj_increase': obj_plus - base_obj,
            'obj_decrease': obj_minus - base_obj,
            'sensitivity': (obj_plus - obj_minus) / (2 * delta_c) if delta_c > 0 else 0
        }
        
        # Sensibilité à la taille du buffer
        delta_K = max(1, int(base_K * perturbation))
        obj_plus = self.objective_function(base_c, base_K + delta_K, alpha)
        obj_minus = self.objective_function(base_c, max(base_c, base_K - delta_K), alpha)
        
        sensitivities['buffer_size'] = {
            'base': base_K,
            'delta': delta_K,
            'obj_increase': obj_plus - base_obj,
            'obj_decrease': obj_minus - base_obj,
            'sensitivity': (obj_plus - obj_minus) / (2 * delta_K) if delta_K > 0 else 0
        }
        
        # Sensibilité au taux d'arrivée
        delta_lambda = self.lambda_rate * perturbation
        
        optimizer_plus = CostOptimizer(
            self.lambda_rate + delta_lambda, self.mu_rate, self.cost_model, self.hours
        )
        optimizer_minus = CostOptimizer(
            max(0.1, self.lambda_rate - delta_lambda), self.mu_rate, self.cost_model, self.hours
        )
        
        obj_plus = optimizer_plus.objective_function(base_c, base_K, alpha)
        obj_minus = optimizer_minus.objective_function(base_c, base_K, alpha)
        
        sensitivities['arrival_rate'] = {
            'base': self.lambda_rate,
            'delta': delta_lambda,
            'obj_increase': obj_plus - base_obj,
            'obj_decrease': obj_minus - base_obj,
            'sensitivity': (obj_plus - obj_minus) / (2 * delta_lambda) if delta_lambda > 0 else 0
        }
        
        return sensitivities
    
    def find_break_even_point(
        self,
        current_c: int,
        current_K: int,
        target_c: int,
        alpha: float = 0.5
    ) -> Dict[str, float]:
        """
        Trouve le point d'équilibre pour justifier un changement de config.
        
        Calcule à partir de quel taux d'arrivée il devient rentable
        de passer de la config actuelle à la config cible.
        
        Args:
            current_c: Serveurs actuels
            current_K: Buffer actuel
            target_c: Serveurs cibles
            alpha: Poids
            
        Returns:
            Dict avec analyse du break-even
        """
        # Fonction pour trouver où les deux configs sont équivalentes
        def diff(lambda_rate):
            opt1 = CostOptimizer(lambda_rate, self.mu_rate, self.cost_model, self.hours)
            opt2 = CostOptimizer(lambda_rate, self.mu_rate, self.cost_model, self.hours)
            
            obj1 = opt1.objective_function(current_c, current_K, alpha)
            obj2 = opt2.objective_function(target_c, current_K, alpha)
            
            return obj1 - obj2
        
        # Recherche dichotomique
        lambda_min = 1
        lambda_max = self.lambda_rate * 3
        
        for _ in range(50):
            lambda_mid = (lambda_min + lambda_max) / 2
            d = diff(lambda_mid)
            
            if abs(d) < 0.001:
                break
            elif d > 0:
                lambda_max = lambda_mid
            else:
                lambda_min = lambda_mid
        
        break_even_lambda = lambda_mid
        
        # Calculer les coûts à ce point
        opt = CostOptimizer(break_even_lambda, self.mu_rate, self.cost_model, self.hours)
        
        return {
            'break_even_arrival_rate': break_even_lambda,
            'current_arrival_rate': self.lambda_rate,
            'should_upgrade': self.lambda_rate > break_even_lambda,
            'margin': (self.lambda_rate - break_even_lambda) / break_even_lambda * 100,
            'current_config_obj': self.objective_function(current_c, current_K, alpha),
            'target_config_obj': self.objective_function(target_c, current_K, alpha)
        }
