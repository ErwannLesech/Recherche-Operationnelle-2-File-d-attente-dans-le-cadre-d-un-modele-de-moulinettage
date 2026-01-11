"""
Tests unitaires pour les modèles de files d'attente.

Vérifie les formules théoriques et la cohérence des simulations.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Ajouter le chemin pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models import MM1Queue, MMcQueue, MMcKQueue, MD1Queue, MG1Queue


class TestMM1Queue:
    """Tests pour le modèle M/M/1."""
    
    def test_stability_condition(self):
        """Vérifie que ρ < 1 est requis pour stabilité avec allow_unstable=False."""
        # Système stable
        queue = MM1Queue(lambda_rate=5, mu_rate=10)
        assert queue.rho < 1
        
        # Système instable devrait lever une erreur avec allow_unstable=False
        with pytest.raises(ValueError):
            MM1Queue(lambda_rate=15, mu_rate=10, allow_unstable=False)
        
        # Mais devrait fonctionner avec allow_unstable=True (par défaut)
        queue_unstable = MM1Queue(lambda_rate=15, mu_rate=10)
        assert queue_unstable.rho > 1
    
    def test_theoretical_metrics(self):
        """Vérifie les formules théoriques M/M/1."""
        queue = MM1Queue(lambda_rate=4, mu_rate=5)
        metrics = queue.compute_theoretical_metrics()
        
        # ρ = λ/μ = 4/5 = 0.8
        assert abs(metrics.rho - 0.8) < 1e-10
        
        # L = ρ/(1-ρ) = 0.8/0.2 = 4
        assert abs(metrics.L - 4.0) < 1e-10
        
        # W = 1/(μ-λ) = 1/(5-4) = 1
        assert abs(metrics.W - 1.0) < 1e-10
        
        # Wq = ρ/(μ(1-ρ)) = 0.8/(5*0.2) = 0.8
        assert abs(metrics.Wq - 0.8) < 1e-10
        
        # Lq = L - ρ = 4 - 0.8 = 3.2
        assert abs(metrics.Lq - 3.2) < 1e-10
    
    def test_little_law(self):
        """Vérifie la loi de Little: L = λW."""
        queue = MM1Queue(lambda_rate=3, mu_rate=10)
        metrics = queue.compute_theoretical_metrics()
        
        # L = λ * W
        assert abs(metrics.L - queue.lambda_rate * metrics.W) < 1e-10
        
        # Lq = λ * Wq
        assert abs(metrics.Lq - queue.lambda_rate * metrics.Wq) < 1e-10
    
    def test_simulation_convergence(self):
        """Vérifie que la simulation converge vers la théorie."""
        queue = MM1Queue(lambda_rate=5, mu_rate=10)
        
        # Simulation longue pour convergence
        result = queue.simulate(n_customers=5000)
        metrics = queue.compute_theoretical_metrics()
        
        # Calcul du temps moyen système depuis les données brutes
        avg_system_time = float(np.mean(result.system_times)) if len(result.system_times) > 0 else 0.0
        
        # Tolérance de 30% (simulation stochastique peut varier)
        assert abs(avg_system_time - metrics.W) / metrics.W < 0.3


class TestMMcQueue:
    """Tests pour le modèle M/M/c."""
    
    def test_reduces_to_mm1(self):
        """M/M/1 est un cas particulier de M/M/c avec c=1."""
        lambda_rate, mu_rate = 3, 10
        
        mm1 = MM1Queue(lambda_rate, mu_rate)
        mmc = MMcQueue(lambda_rate, mu_rate, c=1)
        
        mm1_metrics = mm1.compute_theoretical_metrics()
        mmc_metrics = mmc.compute_theoretical_metrics()
        
        assert abs(mm1_metrics.L - mmc_metrics.L) < 1e-10
        assert abs(mm1_metrics.W - mmc_metrics.W) < 1e-10
    
    def test_more_servers_better_waiting(self):
        """Plus de serveurs = moins d'attente."""
        lambda_rate, mu_rate = 20, 10
        
        mmc2 = MMcQueue(lambda_rate, mu_rate, c=3)
        mmc4 = MMcQueue(lambda_rate, mu_rate, c=5)
        
        metrics2 = mmc2.compute_theoretical_metrics()
        metrics4 = mmc4.compute_theoretical_metrics()
        
        assert metrics4.Wq < metrics2.Wq
        assert metrics4.Lq < metrics2.Lq
    
    def test_erlang_c_probability(self):
        """Vérifie le calcul de la probabilité d'attente (Erlang-C)."""
        queue = MMcQueue(lambda_rate=10, mu_rate=5, c=3)
        metrics = queue.compute_theoretical_metrics()
        
        # La probabilité d'attente doit être entre 0 et 1
        C = queue.compute_erlang_c()
        assert 0 <= C <= 1


class TestMMcKQueue:
    """Tests pour le modèle M/M/c/K."""
    
    def test_blocking_probability(self):
        """Vérifie la probabilité de blocage."""
        queue = MMcKQueue(
            lambda_rate=50, mu_rate=10,
            c=4, K=20
        )
        metrics = queue.compute_theoretical_metrics()
        
        # Probabilité de blocage entre 0 et 1 (Pk)
        assert 0 <= metrics.Pk <= 1
    
    def test_large_buffer_approaches_mmc(self):
        """Avec K très grand, M/M/c/K ≈ M/M/c."""
        lambda_rate, mu_rate, c = 20, 10, 3
        
        mmc = MMcQueue(lambda_rate, mu_rate, c)
        mmck = MMcKQueue(lambda_rate, mu_rate, c, K=1000)
        
        mmc_metrics = mmc.compute_theoretical_metrics()
        mmck_metrics = mmck.compute_theoretical_metrics()
        
        # Avec grand buffer, blocage négligeable (Pk)
        assert mmck_metrics.Pk < 0.01
        
        # Métriques similaires (tolérance 5%)
        assert abs(mmc_metrics.W - mmck_metrics.W) / mmc_metrics.W < 0.05
    
    def test_effective_throughput(self):
        """Vérifie le débit effectif avec blocage."""
        queue = MMcKQueue(
            lambda_rate=100, mu_rate=10,
            c=4, K=10
        )
        metrics = queue.compute_theoretical_metrics()
        
        # Débit effectif = λ * (1 - P_blocage)
        effective_lambda = queue.lambda_rate * (1 - metrics.Pk)
        
        # Le débit ne peut pas dépasser la capacité
        assert effective_lambda <= queue.c * queue.mu_rate


class TestMD1Queue:
    """Tests pour le modèle M/D/1."""
    
    def test_less_waiting_than_mm1(self):
        """M/D/1 a moins de variance donc moins d'attente que M/M/1."""
        lambda_rate, mu_rate = 4, 10
        
        mm1 = MM1Queue(lambda_rate, mu_rate)
        md1 = MD1Queue(lambda_rate, mu_rate)
        
        mm1_metrics = mm1.compute_theoretical_metrics()
        md1_metrics = md1.compute_theoretical_metrics()
        
        # Wq(M/D/1) = Wq(M/M/1) / 2
        assert md1_metrics.Wq < mm1_metrics.Wq
        assert abs(md1_metrics.Wq - mm1_metrics.Wq / 2) < 1e-10
    
    def test_pollaczek_khinchin(self):
        """Vérifie la formule de Pollaczek-Khinchin avec CV²=0."""
        queue = MD1Queue(lambda_rate=3, mu_rate=5)
        metrics = queue.compute_theoretical_metrics()
        
        # Pour M/D/1, CV² = 0
        # Wq = ρ / (2μ(1-ρ))
        rho = 3 / 5
        expected_Wq = rho / (2 * 5 * (1 - rho))
        
        assert abs(metrics.Wq - expected_Wq) < 1e-10


class TestMG1Queue:
    """Tests pour le modèle M/G/1."""
    
    def test_cv_squared_effect(self):
        """CV² plus grand = plus d'attente."""
        lambda_rate = 4
        service_mean = 0.1  # mu = 10
        
        # CV² = variance / mean²
        # low_var: CV² = 0.5 -> variance = 0.5 * 0.1² = 0.005
        # high_var: CV² = 2.0 -> variance = 2.0 * 0.1² = 0.02
        low_var = MG1Queue(lambda_rate, service_mean, service_variance=0.005)
        high_var = MG1Queue(lambda_rate, service_mean, service_variance=0.02)
        
        low_metrics = low_var.compute_theoretical_metrics()
        high_metrics = high_var.compute_theoretical_metrics()
        
        assert high_metrics.Wq > low_metrics.Wq
    
    def test_cv0_equals_md1(self):
        """M/G/1 avec CV²=0 équivaut à M/D/1."""
        lambda_rate = 3
        mu_rate = 10
        service_mean = 1.0 / mu_rate  # 0.1
        
        # CV² = 0 -> variance = 0
        mg1 = MG1Queue(lambda_rate, service_mean, service_variance=0)
        md1 = MD1Queue(lambda_rate, mu_rate)
        
        mg1_metrics = mg1.compute_theoretical_metrics()
        md1_metrics = md1.compute_theoretical_metrics()
        
        assert abs(mg1_metrics.Wq - md1_metrics.Wq) < 1e-10
    
    def test_cv1_equals_mm1(self):
        """M/G/1 avec CV²=1 équivaut à M/M/1."""
        lambda_rate = 3
        mu_rate = 10
        service_mean = 1.0 / mu_rate  # 0.1
        
        # CV² = 1 -> variance = mean² = 0.01
        mg1 = MG1Queue(lambda_rate, service_mean, service_variance=service_mean**2)
        mm1 = MM1Queue(lambda_rate, mu_rate)
        
        mg1_metrics = mg1.compute_theoretical_metrics()
        mm1_metrics = mm1.compute_theoretical_metrics()
        
        assert abs(mg1_metrics.Wq - mm1_metrics.Wq) < 1e-10


class TestCrossModelConsistency:
    """Tests de cohérence entre modèles."""
    
    def test_same_utilization(self):
        """Tous les modèles avec mêmes paramètres ont même ρ."""
        lambda_rate = 5
        mu_rate = 10
        service_mean = 1.0 / mu_rate
        
        mm1 = MM1Queue(lambda_rate, mu_rate)
        md1 = MD1Queue(lambda_rate, mu_rate)
        mg1 = MG1Queue(lambda_rate, service_mean, service_variance=service_mean**2)  # CV²=1
        
        assert mm1.rho == md1.rho == 0.5
        assert abs(mg1.rho - 0.5) < 1e-10
    
    def test_waiting_order(self):
        """Wq(M/D/1) < Wq(M/M/1) < Wq(M/G/1, CV²>1)."""
        lambda_rate = 4
        mu_rate = 10
        service_mean = 1.0 / mu_rate
        
        md1 = MD1Queue(lambda_rate, mu_rate)
        mm1 = MM1Queue(lambda_rate, mu_rate)
        # CV² = 2 -> variance = 2 * mean²
        mg1 = MG1Queue(lambda_rate, service_mean, service_variance=2 * service_mean**2)
        
        md1_wq = md1.compute_theoretical_metrics().Wq
        mm1_wq = mm1.compute_theoretical_metrics().Wq
        mg1_wq = mg1.compute_theoretical_metrics().Wq
        
        assert md1_wq < mm1_wq < mg1_wq


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
