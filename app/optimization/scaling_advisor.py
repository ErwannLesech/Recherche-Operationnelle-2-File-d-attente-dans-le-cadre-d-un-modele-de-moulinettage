"""
Module de recommandations d'auto-scaling.

Ce module analyse la charge et génère des recommandations
pour l'ajustement dynamique du nombre de serveurs.

Stratégies de scaling:
══════════════════════════════════════════════════════════════
1. RÉACTIF: Ajuste basé sur la charge actuelle
   - Simple mais avec latence de réaction
   - Risque de sur/sous-provisionnement temporaire

2. PROGRAMMÉ: Ajuste selon un calendrier
   - Prévisible et stable
   - Ne gère pas les pics inattendus

3. PRÉDICTIF: Anticipe la charge future
   - Optimal en théorie
   - Nécessite des données historiques

4. HYBRIDE: Combine programmé + réactif
   - Base programmée avec ajustements réactifs
   - Bon compromis en pratique
══════════════════════════════════════════════════════════════

Auteurs: ERO2 Team Markov Moulinette Configurators - EPITA
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from enum import Enum
from datetime import datetime, timedelta

from ..models import MMcKQueue
from ..personas import Persona, StudentType


class ScalingAction(Enum):
    """Actions de scaling possibles."""
    NONE = "none"              # Pas de changement
    SCALE_UP = "scale_up"      # Ajouter des serveurs
    SCALE_DOWN = "scale_down"  # Retirer des serveurs
    EMERGENCY = "emergency"    # Scale up d'urgence


class UrgencyLevel(Enum):
    """Niveau d'urgence de la recommandation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ScalingRecommendation:
    """
    Recommandation de scaling.
    
    Contient tous les détails pour une décision de scaling.
    """
    action: ScalingAction
    urgency: UrgencyLevel
    
    # Serveurs
    current_servers: int
    recommended_servers: int
    delta_servers: int
    
    # Contexte
    current_load: float
    predicted_load: float
    peak_expected_in_hours: Optional[float] = None
    
    # Justification
    reason: str = ""
    confidence: float = 0.0
    
    # Impact estimé
    estimated_cost_impact: float = 0.0
    estimated_waiting_time_before: float = 0.0
    estimated_waiting_time_after: float = 0.0
    
    # Méta
    timestamp: datetime = field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None
    
    def is_action_required(self) -> bool:
        """Retourne True si une action est nécessaire."""
        return self.action != ScalingAction.NONE
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire."""
        return {
            'action': self.action.value,
            'urgency': self.urgency.value,
            'current_servers': self.current_servers,
            'recommended_servers': self.recommended_servers,
            'delta_servers': self.delta_servers,
            'current_load': self.current_load,
            'predicted_load': self.predicted_load,
            'reason': self.reason,
            'confidence': self.confidence,
            'estimated_cost_impact': self.estimated_cost_impact
        }


@dataclass
class ScalingPolicy:
    """
    Politique de scaling configurable.
    """
    # Seuils réactifs
    scale_up_threshold: float = 0.75      # ρ > 0.75 → scale up
    scale_down_threshold: float = 0.35    # ρ < 0.35 → scale down
    emergency_threshold: float = 0.95     # ρ > 0.95 → urgence
    
    # Incréments
    scale_up_increment: int = 2
    scale_down_increment: int = 1
    emergency_increment: int = 5
    
    # Limites
    min_servers: int = 2
    max_servers: int = 50
    
    # Cooldown (éviter oscillations)
    cooldown_minutes: float = 10.0
    
    # Paramètres prédictifs
    prediction_horizon_hours: float = 1.0
    anticipation_margin: float = 0.2  # 20% de marge
    
    # Contraintes
    max_budget_per_hour: Optional[float] = None


class ScalingAdvisor:
    """
    Conseiller de scaling intelligent.
    
    Analyse la situation actuelle et prédit les besoins futurs
    pour recommander des ajustements de capacité.
    
    Exemple:
        >>> advisor = ScalingAdvisor(
        ...     mu_rate=10.0,
        ...     buffer_size=100,
        ...     policy=ScalingPolicy()
        ... )
        >>> 
        >>> reco = advisor.get_recommendation(
        ...     current_servers=4,
        ...     current_lambda=50,
        ...     personas=PersonaFactory.create_all_personas(),
        ...     hour=22,
        ...     hours_to_deadline=2.0
        ... )
        >>> 
        >>> if reco.is_action_required():
        ...     print(f"Action: {reco.action.value}, Serveurs: {reco.recommended_servers}")
    """
    
    def __init__(
        self,
        mu_rate: float = 10.0,
        buffer_size: int = 100,
        policy: Optional[ScalingPolicy] = None,
        cost_per_server_hour: float = 0.5
    ):
        """
        Initialise le conseiller.
        
        Args:
            mu_rate: Taux de service par serveur
            buffer_size: Taille du buffer
            policy: Politique de scaling
            cost_per_server_hour: Coût horaire par serveur
        """
        self.mu_rate = mu_rate
        self.buffer_size = buffer_size
        self.policy = policy or ScalingPolicy()
        self.cost_per_server_hour = cost_per_server_hour
        
        # Historique pour prédiction
        self._load_history: List[Tuple[datetime, float]] = []
        self._recommendation_history: List[ScalingRecommendation] = []
    
    def get_recommendation(
        self,
        current_servers: int,
        current_lambda: float,
        personas: Optional[Dict[StudentType, Persona]] = None,
        hour: int = 12,
        is_weekend: bool = False,
        hours_to_deadline: Optional[float] = None
    ) -> ScalingRecommendation:
        """
        Génère une recommandation de scaling.
        
        Args:
            current_servers: Nombre actuel de serveurs
            current_lambda: Taux d'arrivée actuel
            personas: Personas pour prédiction
            hour: Heure actuelle
            is_weekend: Si c'est le weekend
            hours_to_deadline: Heures avant deadline
            
        Returns:
            ScalingRecommendation avec action recommandée
        """
        # Calculer la charge actuelle
        capacity = current_servers * self.mu_rate
        current_rho = current_lambda / capacity if capacity > 0 else 1.0
        
        # Prédire la charge future
        predicted_lambda = self._predict_future_load(
            current_lambda, personas, hour, is_weekend, hours_to_deadline
        )
        predicted_rho = predicted_lambda / capacity if capacity > 0 else 1.0
        
        # Calculer les serveurs optimaux pour la charge prédite
        optimal_servers = self._compute_optimal_servers(predicted_lambda)
        
        # Déterminer l'action
        action, urgency, reason = self._determine_action(
            current_servers, optimal_servers,
            current_rho, predicted_rho
        )
        
        # Calculer les serveurs recommandés
        if action == ScalingAction.SCALE_UP:
            recommended = min(
                current_servers + self.policy.scale_up_increment,
                self.policy.max_servers
            )
        elif action == ScalingAction.SCALE_DOWN:
            recommended = max(
                current_servers - self.policy.scale_down_increment,
                self.policy.min_servers
            )
        elif action == ScalingAction.EMERGENCY:
            recommended = min(
                current_servers + self.policy.emergency_increment,
                self.policy.max_servers
            )
        else:
            recommended = current_servers
        
        # Estimer l'impact
        cost_impact = (recommended - current_servers) * self.cost_per_server_hour
        
        wt_before = self._estimate_waiting_time(current_servers, current_lambda)
        wt_after = self._estimate_waiting_time(recommended, predicted_lambda)
        
        # Confiance de la prédiction
        confidence = self._compute_confidence(personas, hours_to_deadline)
        
        # Créer la recommandation
        reco = ScalingRecommendation(
            action=action,
            urgency=urgency,
            current_servers=current_servers,
            recommended_servers=recommended,
            delta_servers=recommended - current_servers,
            current_load=current_rho,
            predicted_load=predicted_rho,
            peak_expected_in_hours=self._estimate_peak_hours(hour, hours_to_deadline),
            reason=reason,
            confidence=confidence,
            estimated_cost_impact=cost_impact,
            estimated_waiting_time_before=wt_before,
            estimated_waiting_time_after=wt_after,
            valid_until=datetime.now() + timedelta(minutes=self.policy.cooldown_minutes)
        )
        
        # Sauvegarder dans l'historique
        self._recommendation_history.append(reco)
        
        return reco
    
    def _predict_future_load(
        self,
        current_lambda: float,
        personas: Optional[Dict[StudentType, Persona]],
        hour: int,
        is_weekend: bool,
        hours_to_deadline: Optional[float]
    ) -> float:
        """
        Prédit la charge dans l'horizon de prédiction.
        """
        if personas is None:
            # Sans personas, utiliser un modèle simple
            # Assumer que la charge peut augmenter de 50% en pic
            return current_lambda * 1.3
        
        # Calculer la charge dans 1 heure
        future_hour = (hour + 1) % 24
        future_hours_to_deadline = None
        if hours_to_deadline is not None:
            future_hours_to_deadline = max(0, hours_to_deadline - 1)
        
        # Somme des taux des personas
        predicted_lambda = 0.0
        for persona in personas.values():
            predicted_lambda += persona.get_arrival_rate(
                future_hour, is_weekend, future_hours_to_deadline
            )
        
        # Appliquer marge d'anticipation
        predicted_lambda *= (1 + self.policy.anticipation_margin)
        
        return max(current_lambda, predicted_lambda)
    
    def _compute_optimal_servers(self, lambda_rate: float) -> int:
        """
        Calcule le nombre optimal de serveurs pour un taux donné.
        
        Objectif: ρ ≈ 0.7 (bon compromis performance/coût)
        """
        target_rho = 0.7
        optimal = int(np.ceil(lambda_rate / (self.mu_rate * target_rho)))
        
        return max(self.policy.min_servers, min(optimal, self.policy.max_servers))
    
    def _determine_action(
        self,
        current_servers: int,
        optimal_servers: int,
        current_rho: float,
        predicted_rho: float
    ) -> Tuple[ScalingAction, UrgencyLevel, str]:
        """
        Détermine l'action à prendre.
        """
        policy = self.policy
        
        # Urgence si charge critique
        if current_rho >= policy.emergency_threshold:
            return (
                ScalingAction.EMERGENCY,
                UrgencyLevel.CRITICAL,
                f"Charge critique ({current_rho:.0%}). Scale up d'urgence requis."
            )
        
        # Prédiction de charge élevée
        if predicted_rho >= policy.emergency_threshold:
            return (
                ScalingAction.SCALE_UP,
                UrgencyLevel.HIGH,
                f"Charge prévue critique ({predicted_rho:.0%}). Anticipation recommandée."
            )
        
        # Scale up si charge élevée
        if current_rho >= policy.scale_up_threshold or predicted_rho >= policy.scale_up_threshold:
            urgency = UrgencyLevel.MEDIUM if current_rho < predicted_rho else UrgencyLevel.HIGH
            return (
                ScalingAction.SCALE_UP,
                urgency,
                f"Charge élevée (actuelle: {current_rho:.0%}, prévue: {predicted_rho:.0%}). "
                f"Ajout de serveurs recommandé."
            )
        
        # Scale down si charge faible
        if current_rho < policy.scale_down_threshold and predicted_rho < policy.scale_down_threshold:
            if current_servers > self.policy.min_servers:
                return (
                    ScalingAction.SCALE_DOWN,
                    UrgencyLevel.LOW,
                    f"Charge faible ({current_rho:.0%}). Réduction possible pour économies."
                )
        
        # Pas d'action
        return (
            ScalingAction.NONE,
            UrgencyLevel.LOW,
            f"Charge stable ({current_rho:.0%}). Configuration actuelle adéquate."
        )
    
    def _estimate_waiting_time(
        self,
        n_servers: int,
        lambda_rate: float
    ) -> float:
        """
        Estime le temps d'attente moyen en minutes.
        """
        try:
            queue = MMcKQueue(
                lambda_rate, self.mu_rate, n_servers, self.buffer_size
            )
            metrics = queue.compute_theoretical_metrics()
            return metrics.Wq * 60  # minutes
        except:
            return float('inf')
    
    def _compute_confidence(
        self,
        personas: Optional[Dict],
        hours_to_deadline: Optional[float]
    ) -> float:
        """
        Calcule le niveau de confiance de la prédiction.
        
        Plus élevé si:
        - Personas disponibles
        - Deadline connue
        - Historique suffisant
        """
        confidence = 0.5  # Base
        
        if personas is not None:
            confidence += 0.2
        
        if hours_to_deadline is not None:
            confidence += 0.2
        
        if len(self._load_history) >= 24:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _estimate_peak_hours(
        self,
        current_hour: int,
        hours_to_deadline: Optional[float]
    ) -> Optional[float]:
        """
        Estime dans combien de temps le pic aura lieu.
        """
        if hours_to_deadline is not None and hours_to_deadline < 24:
            # Pic juste avant la deadline
            return max(0, hours_to_deadline - 2)
        
        # Pics typiques: 14h-16h, 21h-23h
        peak_hours = [14, 15, 21, 22, 23]
        
        for peak in peak_hours:
            hours_until = (peak - current_hour) % 24
            if hours_until <= 4:
                return hours_until
        
        return None
    
    def get_scaling_schedule(
        self,
        personas: Dict[StudentType, Persona],
        hours_ahead: int = 24
    ) -> Dict[int, int]:
        """
        Génère un calendrier de scaling sur les prochaines heures.
        
        Args:
            personas: Personas pour la prédiction
            hours_ahead: Nombre d'heures à planifier
            
        Returns:
            Dict {heure: nb_serveurs_recommandé}
        """
        schedule = {}
        
        for h in range(hours_ahead):
            hour = (datetime.now().hour + h) % 24
            
            # Calculer le taux d'arrivée prévu
            lambda_rate = sum(
                p.get_arrival_rate(hour)
                for p in personas.values()
            )
            
            # Serveurs optimaux
            optimal = self._compute_optimal_servers(lambda_rate)
            schedule[hour] = optimal
        
        return schedule
    
    def analyze_scaling_opportunities(
        self,
        current_servers: int,
        personas: Dict[StudentType, Persona],
        hours: int = 24
    ) -> Dict[str, any]:
        """
        Analyse les opportunités de scaling sur une période.
        
        Returns:
            Dict avec analyse des opportunités
        """
        schedule = self.get_scaling_schedule(personas, hours)
        
        # Statistiques
        server_counts = list(schedule.values())
        avg_servers = np.mean(server_counts)
        max_servers = max(server_counts)
        min_servers = min(server_counts)
        
        # Heures où on peut réduire
        reduction_hours = [h for h, s in schedule.items() if s < current_servers]
        
        # Heures où on doit augmenter
        increase_hours = [h for h, s in schedule.items() if s > current_servers]
        
        # Économies potentielles
        current_cost = current_servers * hours * self.cost_per_server_hour
        optimal_cost = sum(schedule.values()) * self.cost_per_server_hour
        savings = current_cost - optimal_cost
        
        return {
            'schedule': schedule,
            'avg_servers_needed': avg_servers,
            'max_servers_needed': max_servers,
            'min_servers_needed': min_servers,
            'current_servers': current_servers,
            'reduction_hours': reduction_hours,
            'increase_hours': increase_hours,
            'potential_savings_per_day': savings,
            'recommendation': self._summarize_analysis(
                current_servers, avg_servers, max_servers, min_servers, savings
            )
        }
    
    def _summarize_analysis(
        self,
        current: int,
        avg: float,
        max_s: int,
        min_s: int,
        savings: float
    ) -> str:
        """Résume l'analyse en texte."""
        if current < avg:
            return (
                f"⚠️ Sous-provisionnement: Vous avez {current} serveurs mais "
                f"la moyenne nécessaire est {avg:.1f}. Risque de dégradation."
            )
        elif current > max_s + 2:
            return (
                f"💰 Sur-provisionnement: Vous avez {current} serveurs mais "
                f"le maximum nécessaire est {max_s}. "
                f"Économie potentielle: {savings:.2f}€/jour."
            )
        else:
            return (
                f"✅ Configuration adaptée: {current} serveurs pour un besoin "
                f"variant entre {min_s} et {max_s}."
            )
