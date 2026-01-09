"""
Tests pour les modules d'optimisation et de scaling.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.optimization import CostOptimizer, ScalingAdvisor, CostModel, ScalingPolicy, ScalingAction
from app.personas import PersonaFactory


class TestCostOptimizer:
    """Tests pour l'optimiseur de coûts."""
    
    def test_finds_optimal_servers(self):
        """L'optimiseur trouve un nombre optimal de serveurs."""
        optimizer = CostOptimizer(
            lambda_rate=30,
            mu_rate=10
        )
        
        result = optimizer.optimize(c_range=(1, 20))
        
        assert result.optimal_servers >= 1
        assert result.optimal_servers <= 20
        assert result.objective_value > 0
    
    def test_more_load_more_servers(self):
        """Plus de charge nécessite plus de serveurs."""
        low_load = CostOptimizer(lambda_rate=20, mu_rate=10)
        high_load = CostOptimizer(lambda_rate=80, mu_rate=10)
        
        low_result = low_load.optimize()
        high_result = high_load.optimize()
        
        assert high_result.optimal_servers > low_result.optimal_servers
    
    def test_alpha_beta_trade_off(self):
        """Differents modeles de cout produisent des resultats differents."""
        # Modele favorisant le cout bas (serveurs moins chers)
        cheap_model = CostModel(cost_per_server_hour=0.1, cost_per_waiting_minute=0.01)
        cheap_focused = CostOptimizer(lambda_rate=30, mu_rate=10, cost_model=cheap_model)
        cheap_result = cheap_focused.optimize()
        
        # Modele favorisant la QoS (attente couteuse)
        qos_model = CostModel(cost_per_server_hour=0.1, cost_per_waiting_minute=1.0)
        qos_focused = CostOptimizer(lambda_rate=30, mu_rate=10, cost_model=qos_model)
        qos_result = qos_focused.optimize()
        
        # QoS devrait avoir plus de serveurs (minimiser attente)
        assert qos_result.optimal_servers >= cheap_result.optimal_servers
    
    def test_cost_model_impact(self):
        """Le modèle de coût affecte l'optimum."""
        cheap_servers = CostModel(cost_per_server_hour=0.1)
        expensive_servers = CostModel(cost_per_server_hour=5.0)
        
        cheap = CostOptimizer(lambda_rate=30, mu_rate=10, cost_model=cheap_servers)
        expensive = CostOptimizer(lambda_rate=30, mu_rate=10, cost_model=expensive_servers)
        
        cheap_result = cheap.optimize()
        expensive_result = expensive.optimize()
        
        # Avec serveurs chers, on en met moins
        assert expensive_result.optimal_servers <= cheap_result.optimal_servers


class TestScalingAdvisor:
    """Tests pour le conseiller de scaling."""
    
    def test_high_load_recommends_scale_up(self):
        """Charge élevée recommande scale up."""
        advisor = ScalingAdvisor(
            mu_rate=10,
            buffer_size=100,
            policy=ScalingPolicy(scale_up_threshold=0.7)
        )
        
        # Charge élevée: λ=45, 4 serveurs, μ=10 → ρ = 1.125 > 1!
        reco = advisor.get_recommendation(
            current_servers=4,
            current_lambda=45,
            hour=14
        )
        
        assert reco.action in [ScalingAction.SCALE_UP, ScalingAction.EMERGENCY]
    
    def test_low_load_recommends_scale_down(self):
        """Charge faible recommande scale down."""
        advisor = ScalingAdvisor(
            mu_rate=10,
            buffer_size=100,
            policy=ScalingPolicy(scale_down_threshold=0.35, min_servers=2)
        )
        
        # Charge faible: λ=10, 10 serveurs, μ=10 → ρ = 0.1
        reco = advisor.get_recommendation(
            current_servers=10,
            current_lambda=10,
            hour=14
        )
        
        assert reco.action == ScalingAction.SCALE_DOWN
    
    def test_stable_load_no_action(self):
        """Charge stable ne recommande aucune action."""
        advisor = ScalingAdvisor(
            mu_rate=10,
            buffer_size=100,
            policy=ScalingPolicy(
                scale_up_threshold=0.8,
                scale_down_threshold=0.3
            )
        )
        
        # Charge moyenne: λ=30, 6 serveurs, μ=10 → ρ = 0.5
        reco = advisor.get_recommendation(
            current_servers=6,
            current_lambda=30,
            hour=14
        )
        
        assert reco.action == ScalingAction.NONE
    
    def test_deadline_anticipation(self):
        """Anticipe la charge avant deadline."""
        advisor = ScalingAdvisor(mu_rate=10, buffer_size=100)
        personas = PersonaFactory.create_all_personas()
        
        # Charge actuelle ok mais deadline dans 2h
        reco = advisor.get_recommendation(
            current_servers=4,
            current_lambda=20,
            personas=personas,
            hour=14,
            hours_to_deadline=2.0
        )
        
        # Devrait anticiper l'augmentation
        assert reco.predicted_load > reco.current_load
    
    def test_respects_min_max_servers(self):
        """Respecte les limites min/max de serveurs."""
        policy = ScalingPolicy(min_servers=3, max_servers=10)
        advisor = ScalingAdvisor(mu_rate=10, buffer_size=100, policy=policy)
        
        # Essayer de descendre sous le min
        reco_down = advisor.get_recommendation(
            current_servers=3,
            current_lambda=5,
            hour=3
        )
        
        assert reco_down.recommended_servers >= 3
        
        # Essayer de monter au-dessus du max
        reco_up = advisor.get_recommendation(
            current_servers=10,
            current_lambda=200,
            hour=22
        )
        
        assert reco_up.recommended_servers <= 10
    
    def test_scheduling_over_24h(self):
        """Génère un planning sur 24h."""
        advisor = ScalingAdvisor(mu_rate=10, buffer_size=100)
        personas = PersonaFactory.create_all_personas()
        
        schedule = advisor.get_scaling_schedule(personas, hours_ahead=24)
        
        assert len(schedule) == 24
        assert all(isinstance(n, int) and n >= 1 for n in schedule.values())
    
    def test_analyze_opportunities(self):
        """Analyse les opportunités de scaling."""
        advisor = ScalingAdvisor(mu_rate=10, buffer_size=100)
        personas = PersonaFactory.create_all_personas()
        
        analysis = advisor.analyze_scaling_opportunities(
            current_servers=5,
            personas=personas,
            hours=24
        )
        
        assert 'schedule' in analysis
        assert 'avg_servers_needed' in analysis
        assert 'recommendation' in analysis


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
