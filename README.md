# Moulinette Simulator - Recherche Opérationnelle 2

> Simulation et optimisation de files d'attente pour le système de correction automatique EPITA

Ce projet analyse la moulinette, infrastructure de correction automatique, sous l'angle des systèmes d'attente. Il vise à modéliser son fonctionnement, formuler des hypothèses et étudier ses performances à partir d'une version simplifiée mais réaliste.

## Fonctionnalités

- **Modèles de files d'attente**: M/M/1, M/M/c, M/M/c/K, M/D/1, M/G/1
- **Personas étudiants**: Prépa Sup/Spé, Ing1/2/3 avec comportements différenciés
- **Simulation de rush**: Périodes de deadline, semaines d'examens
- **Optimisation coût/QoS**: Frontière de Pareto, configuration optimale
- **Auto-scaling**: Recommandations dynamiques de dimensionnement
- **Interface Streamlit**: Visualisation interactive et heatmaps

## Installation

```bash
# Cloner le repo
git clone https://github.com/votre-repo/moulinette-simulator.git
cd moulinette-simulator

# Installer les dépendances
pip install -r requirements.txt
```

## Usage

### Lancer l'interface web

```bash
# Méthode 1: Script de lancement
python run_app.py

# Méthode 2: Directement avec Streamlit
streamlit run app/frontend/main_app.py
```

Ouvrez ensuite http://localhost:8501 dans votre navigateur.

### Utilisation programmatique

```python
from app import MM1Queue, MMcKQueue, PersonaFactory, CostOptimizer

# Créer une file M/M/1
queue = MM1Queue(lambda_rate=5, mu_rate=10)
metrics = queue.compute_theoretical_metrics()
print(f"Temps d'attente moyen: {metrics.W:.3f}")

# Simuler une file M/M/c/K
queue = MMcKQueue(lambda_rate=30, mu_rate=10, n_servers=4, buffer_size=100)
result = queue.simulate(duration=60)
print(f"Clients traités: {result.total_customers}")

# Optimiser le nombre de serveurs
optimizer = CostOptimizer(lambda_rate=30, mu_rate=10)
optimal = optimizer.optimize()
print(f"Serveurs optimaux: {optimal.optimal_servers}")
```

## Structure du projet

```
app/
├── __init__.py              # Exports principaux
├── models/                  # Modèles de files d'attente
│   ├── base_queue.py        # Classe abstraite BaseQueueModel
│   ├── mm1.py               # File M/M/1
│   ├── mmc.py               # File M/M/c
│   ├── mmck.py              # File M/M/c/K (avec blocage)
│   ├── md1.py               # File M/D/1 (service déterministe)
│   └── mg1.py               # File M/G/1 (service général)
├── personas/                # Modélisation des étudiants
│   ├── personas.py          # Types d'étudiants et comportements
│   └── usage_patterns.py    # Patterns temporels (rush, deadline)
├── simulation/              # Moteur de simulation
│   ├── rush_simulator.py    # Simulation de périodes de rush
│   └── moulinette_system.py # Système complet (build + test)
├── optimization/            # Optimisation et scaling
│   ├── cost_optimizer.py    # Optimisation coût/performance
│   └── scaling_advisor.py   # Recommandations d'auto-scaling
└── frontend/                # Interface Streamlit
    └── main_app.py          # Application web interactive
```

## Modèles mathématiques

### Notation de Kendall: A/S/c/K/N/D

- **A**: Loi des arrivées (M=Markov/Poisson, D=Déterministe, G=Générale)
- **S**: Loi de service (M, D, G)
- **c**: Nombre de serveurs
- **K**: Capacité du système (buffer)
- **N**: Population (∞ par défaut)
- **D**: Discipline (FIFO par défaut)

### Formules clés

| Modèle | Condition stabilité | Temps attente Wq |
|--------|---------------------|------------------|
| M/M/1 | ρ < 1 | ρ / (μ(1-ρ)) |
| M/M/c | ρ < c | (C(c,ρ) × ρ) / (cμ(1-ρ/c)) |
| M/D/1 | ρ < 1 | ρ / (2μ(1-ρ)) |
| M/G/1 | ρ < 1 | (ρ + λμσ²) / (2(1-ρ)) |

## Authors

- Florian Ruiz
- Victor Mandelaire
- Abel Aubron
- Nathan Claude
- Aymeric Le Riboter
- Erwann Lesech

## License

MIT License - voir [LICENSE](LICENSE)
