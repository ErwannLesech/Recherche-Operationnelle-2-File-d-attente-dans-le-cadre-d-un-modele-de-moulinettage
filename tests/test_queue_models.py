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
        lambda_rate = 5
        service_mean = 0.2
        service_variance = 0.04

        # Ajuster les paramètres pour éviter l'instabilité
        lambda_rate = 4  # Réduction du taux d'arrivée pour ρ < 1

        mg1_queue = MG1Queue(
            lambda_rate=lambda_rate,
            service_mean=service_mean,
            service_variance=service_variance
        )
        generic_queue = GenericQueue(
            lambda_rate=lambda_rate,
            mu_rate=1/service_mean,
            kendall_notation="M/G/1"
        )

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
        queue = GenericQueue(lambda_rate=lambda_rate, mu_rate=mu_rate, c=c, K=buffer_size, kendall_notation="M/M/c")
        queue2 = MMcQueue(lambda_rate=lambda_rate, mu_rate=mu_rate, c=c) 
        results = queue.simulate(n_customers=1000)
        results2 = queue2.simulate(n_customers=1000)

        # Vérifier les métriques dynamiques
        # Calculer le temps d'attente moyen à partir des données de simulation
        avg_waiting_time = np.mean(results.waiting_times)
        self.assertGreaterEqual(avg_waiting_time, 0, "Le temps d'attente moyen doit être positif.")
        eps = 1e-2
        self.assertLessEqual(avg_waiting_time, 1 / mu_rate + eps, "Le temps d'attente moyen ne doit pas dépasser le temps de service moyen.")

        self.assertGreaterEqual(len(results.queue_length_trace), 0, "L'occupation du buffer doit être positive.")
        self.assertListEqual(list(results.queue_length_trace), list(results2.queue_length_trace), "Les traces d'occupation du buffer doivent être identiques entre les deux modèles.")

        self.assertGreaterEqual(results.blocking_probability, 0, "La probabilité de blocage doit être positive.")
        self.assertLessEqual(results.blocking_probability, 1, "La probabilité de blocage doit être inférieure ou égale à 1.")

        # Vérifier que toutes les valeurs de queue_length_trace sont positives
        self.assertTrue(np.all(results.queue_length_trace >= 0), "Toutes les longueurs de queue doivent être positives.")

        # Vérifier que les valeurs de queue_length_trace ne dépassent pas la capacité du buffer
        self.assertTrue(np.all(results.queue_length_trace <= buffer_size), "Les longueurs de queue ne doivent pas dépasser la capacité du buffer.")

if __name__ == "__main__":
    unittest.main()