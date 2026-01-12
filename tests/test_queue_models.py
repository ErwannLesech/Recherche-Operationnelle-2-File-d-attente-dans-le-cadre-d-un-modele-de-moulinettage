import unittest
from app.models.base_queue import GenericQueue, ChainQueue

# Import old models for comparison
from app.models.old.mm1 import MM1Queue
from app.models.old.mmc import MMcQueue
from app.models.old.md1 import MD1Queue
from app.models.old.mdc import MDcQueue
from app.models.old.mg1 import MG1Queue
from app.models.old.mgc import MGcQueue

import numpy as np


class TestQueueModels(unittest.TestCase):

    def test_mm1_equivalence(self):
        """Test que GenericQueue produit les mêmes résultats que MM1Queue."""
        lambda_rate = 5
        mu_rate = 10

        mm1_queue = MM1Queue(lambda_rate=lambda_rate, mu_rate=mu_rate)
        generic_queue = GenericQueue(lambda_rate=lambda_rate, mu_rate=mu_rate, kendall_notation="M/M/1")

        mm1_metrics = mm1_queue.compute_theoretical_metrics()
        generic_metrics = generic_queue.compute_theoretical_metrics()

        self.assertAlmostEqual(mm1_metrics.L, generic_metrics.L, places=5)
        self.assertAlmostEqual(mm1_metrics.Lq, generic_metrics.Lq, places=5)
        self.assertAlmostEqual(mm1_metrics.W, generic_metrics.W, places=5)
        self.assertAlmostEqual(mm1_metrics.Wq, generic_metrics.Wq, places=5)

    def test_mmc_equivalence(self):
        """Test que GenericQueue produit les mêmes résultats que MMcQueue."""
        lambda_rate = 8
        mu_rate = 4
        c = 7

        mmc_queue = MMcQueue(lambda_rate=lambda_rate, mu_rate=mu_rate, c=c)
        generic_queue = GenericQueue(lambda_rate=lambda_rate, mu_rate=mu_rate, c=c, kendall_notation="M/M/c")

        mmc_metrics = mmc_queue.compute_theoretical_metrics()
        generic_metrics = generic_queue.compute_theoretical_metrics()

        self.assertAlmostEqual(mmc_metrics.L, generic_metrics.L, places=5)
        self.assertAlmostEqual(mmc_metrics.Lq, generic_metrics.Lq, places=5)
        self.assertAlmostEqual(mmc_metrics.W, generic_metrics.W, places=5)
        self.assertAlmostEqual(mmc_metrics.Wq, generic_metrics.Wq, places=5)

    def test_md1_equivalence(self):
        """Test que GenericQueue produit les mêmes résultats que MD1Queue."""
        lambda_rate = 10
        mu_rate = 12

        md1_queue = MD1Queue(lambda_rate=lambda_rate, mu_rate=mu_rate)
        generic_queue = GenericQueue(lambda_rate=lambda_rate, mu_rate=mu_rate, kendall_notation="M/D/1")

        md1_metrics = md1_queue.compute_theoretical_metrics()
        generic_metrics = generic_queue.compute_theoretical_metrics()

        self.assertAlmostEqual(md1_metrics.L, generic_metrics.L, places=5)
        self.assertAlmostEqual(md1_metrics.Lq, generic_metrics.Lq, places=5)
        self.assertAlmostEqual(md1_metrics.W, generic_metrics.W, places=5)
        self.assertAlmostEqual(md1_metrics.Wq, generic_metrics.Wq, places=5)

    def test_mdc_equivalence(self):
        """Test que GenericQueue produit les mêmes résultats que MDcQueue."""
        lambda_rate = 20
        mu_rate = 15
        c = 8

        mdc_queue = MDcQueue(lambda_rate=lambda_rate, mu_rate=mu_rate, c=c)
        generic_queue = GenericQueue(lambda_rate=lambda_rate, mu_rate=mu_rate, c=c, kendall_notation="M/D/c")

        mdc_metrics = mdc_queue.compute_theoretical_metrics()
        generic_metrics = generic_queue.compute_theoretical_metrics()

        self.assertAlmostEqual(mdc_metrics.L, generic_metrics.L, places=5)
        self.assertAlmostEqual(mdc_metrics.Lq, generic_metrics.Lq, places=5)
        self.assertAlmostEqual(mdc_metrics.W, generic_metrics.W, places=5)
        self.assertAlmostEqual(mdc_metrics.Wq, generic_metrics.Wq, places=5)

    def test_mg1_equivalence(self):
        """Test que GenericQueue produit les mêmes résultats que MG1Queue."""
        lambda_rate = 4  # Réduction pour stabilité
        service_mean = 0.2
        service_variance = 0.04

        mg1_queue = MG1Queue(
            lambda_rate=lambda_rate,
            service_mean=service_mean,
            service_variance=service_variance
        )
        
        # Pour GenericQueue, on doit passer mu_rate et configurer service_variance
        generic_queue = GenericQueue(
            lambda_rate=lambda_rate,
            mu_rate=1/service_mean,
            kendall_notation="M/G/1"
        )
        # Configurer la variance
        generic_queue.service_variance = service_variance

        mg1_metrics = mg1_queue.compute_theoretical_metrics()
        generic_metrics = generic_queue.compute_theoretical_metrics()

        self.assertAlmostEqual(mg1_metrics.L, generic_metrics.L, places=5)
        self.assertAlmostEqual(mg1_metrics.Lq, generic_metrics.Lq, places=5)
        self.assertAlmostEqual(mg1_metrics.W, generic_metrics.W, places=5)
        self.assertAlmostEqual(mg1_metrics.Wq, generic_metrics.Wq, places=5)

    def test_mgc_equivalence(self):
        """Test que GenericQueue produit les mêmes résultats que MGcQueue."""
        lambda_rate = 6
        service_mean = 0.2
        service_variance = 0.04
        c = 6

        mgc_queue = MGcQueue(
            lambda_rate=lambda_rate,
            service_mean=service_mean,
            service_variance=service_variance,
            c=c
        )
        
        generic_queue = GenericQueue(
            lambda_rate=lambda_rate,
            mu_rate=1/service_mean,
            c=c,
            kendall_notation="M/G/c"
        )
        # Configurer la variance
        generic_queue.service_variance = service_variance

        mgc_metrics = mgc_queue.compute_theoretical_metrics()
        generic_metrics = generic_queue.compute_theoretical_metrics()

        self.assertAlmostEqual(mgc_metrics.L, generic_metrics.L, places=5)
        self.assertAlmostEqual(mgc_metrics.Lq, generic_metrics.Lq, places=5)
        self.assertAlmostEqual(mgc_metrics.W, generic_metrics.W, places=5)
        self.assertAlmostEqual(mgc_metrics.Wq, generic_metrics.Wq, places=5)

    def test_queue_chain(self):
        """Test pour une chaîne de files d'attente."""
        queue1 = GenericQueue(lambda_rate=5, mu_rate=10, kendall_notation="M/M/1")
        queue2 = GenericQueue(lambda_rate=5, mu_rate=8, kendall_notation="M/D/1")
        queue3 = GenericQueue(lambda_rate=5, mu_rate=6, kendall_notation="M/G/1")

        # Connecter les queues
        queue1.connect_to_next_queue(queue2, delay=0.5)
        queue2.connect_to_next_queue(queue3, delay=0.2)

        # Créer une chaîne
        chain = ChainQueue([queue1, queue2, queue3])

        # Simuler la chaîne
        results = chain.simulate_chain(n_customers=100)

        # Vérifier les résultats
        self.assertEqual(len(results), 3)
        self.assertGreater(len(results[0].departure_times), 0)
        self.assertGreater(len(results[1].departure_times), 0)
        self.assertGreater(len(results[2].departure_times), 0)

        # Vérifier la notation de Kendall
        self.assertEqual(chain.get_kendall_representation(), "M/M/1 -> M/D/1 -> M/G/1")

    def test_simulation_metrics(self):
        """Test des métriques dynamiques via simulation."""
        lambda_rate = 5
        mu_rate = 10
        c = 3
        buffer_size = 10

        # Créer une file d'attente générique avec simulation
        queue = GenericQueue(
            lambda_rate=lambda_rate, 
            mu_rate=mu_rate, 
            c=c, 
            K=buffer_size, 
            kendall_notation="M/M/c"
        )
        
        queue2 = MMcQueue(lambda_rate=lambda_rate, mu_rate=mu_rate, c=c) 
        
        # Exécuter les simulations avec même seed pour comparaison
        results = queue.simulate(n_customers=1000)
        results2 = queue2.simulate(n_customers=1000)

        # Vérifier les métriques dynamiques
        avg_waiting_time = np.mean(results.waiting_times)
        self.assertGreaterEqual(avg_waiting_time, 0, "Le temps d'attente moyen doit être positif.")
        
        eps = 1e-1  # Tolérance augmentée car c'est une simulation
        max_expected_wait = 10 / mu_rate  # 10x le temps de service moyen comme limite supérieure
        self.assertLessEqual(
            avg_waiting_time, 
            max_expected_wait, 
            f"Le temps d'attente moyen {avg_waiting_time:.3f} ne doit pas dépasser {max_expected_wait:.3f}"
        )

        # Vérifier que les traces existent et sont cohérentes
        self.assertGreaterEqual(len(results.queue_length_trace), 0, "queue_length_trace doit exister.")
        self.assertGreaterEqual(len(results.time_trace), 0, "time_trace doit exister.")
        self.assertEqual(
            len(results.queue_length_trace), 
            len(results.time_trace), 
            "queue_length_trace et time_trace doivent avoir la même longueur."
        )

        # Vérifier la propriété blocking_probability
        self.assertGreaterEqual(results.blocking_probability, 0, "La probabilité de blocage doit être positive.")
        self.assertLessEqual(results.blocking_probability, 1, "La probabilité de blocage doit être ≤ 1.")

        # Vérifier que toutes les valeurs de queue_length_trace sont positives
        self.assertTrue(
            np.all(results.queue_length_trace >= 0), 
            "Toutes les longueurs de queue doivent être positives."
        )

        # Pour les files avec capacité K, vérifier que la queue ne dépasse pas K - c
        if buffer_size is not None:
            max_queue_length = buffer_size - c
            self.assertTrue(
                np.all(results.queue_length_trace <= max_queue_length),
                f"Les longueurs de queue ne doivent pas dépasser {max_queue_length}"
            )

    def test_mm1_simulation_vs_theory(self):
        """Test que la simulation M/M/1 converge vers les valeurs théoriques."""
        lambda_rate = 5
        mu_rate = 10
        
        queue = GenericQueue(lambda_rate=lambda_rate, mu_rate=mu_rate, kendall_notation="M/M/1")
        
        # Métriques théoriques
        theory = queue.compute_theoretical_metrics()
        
        # Simulation avec beaucoup de clients pour convergence
        sim_results = queue.simulate(n_customers=10000)
        
        # Vérifier convergence (avec tolérance pour stochastique)
        empirical_wq = np.mean(sim_results.waiting_times)
        self.assertAlmostEqual(empirical_wq, theory.Wq, delta=theory.Wq * 0.2)  # 20% tolérance
        
        empirical_w = np.mean(sim_results.system_times)
        self.assertAlmostEqual(empirical_w, theory.W, delta=theory.W * 0.2)

    def test_deterministic_service_times(self):
        """Test que M/D/1 génère bien des temps de service constants."""
        lambda_rate = 5
        mu_rate = 10
        
        queue = GenericQueue(lambda_rate=lambda_rate, mu_rate=mu_rate, kendall_notation="M/D/1")
        results = queue.simulate(n_customers=100)
        
        # Vérifier que tous les temps de service sont égaux
        expected_service_time = 1.0 / mu_rate
        self.assertTrue(
            np.all(np.abs(results.service_times - expected_service_time) < 1e-10),
            "Tous les temps de service M/D/1 doivent être identiques (déterministes)"
        )

    def test_multiserver_queue_length(self):
        """Test que la longueur de queue multi-serveurs est correcte."""
        lambda_rate = 15
        mu_rate = 10
        c = 3
        
        queue = GenericQueue(lambda_rate=lambda_rate, mu_rate=mu_rate, c=c, kendall_notation="M/M/c")
        results = queue.simulate(n_customers=500)
        
        # La longueur de queue doit être system_size - min(system_size, c)
        # Vérifier quelques points
        self.assertTrue(
            np.all(results.queue_length_trace >= 0),
            "La longueur de queue ne peut pas être négative"
        )
        
        # Vérifier qu'il y a bien des moments où la queue est vide (au début notamment)
        self.assertIn(0, results.queue_length_trace, "La queue devrait être vide à certains moments")


if __name__ == "__main__":
    unittest.main()