# Moulinette Simulator - Recherche Opérationnelle 2

> Simulation et optimisation de files d'attente pour le système de correction automatique EPITA

Ce projet analyse la moulinette, infrastructure de correction automatique, sous l'angle des systèmes d'attente. Il vise à modéliser son fonctionnement, formuler des hypothèses et étudier ses performances à partir d'une version simplifiée mais réaliste.

## Fonctionnalités

- **Modèles de files d'attente**: M/M/1, M/M/c, M/M/c/K, M/D/1, M/G/1, M/G/c
- **Personas étudiants**: Prépa Sup/Spé, Ing1/2/3 avec comportements différenciés
- **Simulation de rush**: Périodes de deadline, semaines d'examens
- **Optimisation coût/QoS**: Frontière de Pareto, configuration optimale
- **Auto-scaling**: Recommandations dynamiques de dimensionnement
- **Interface Streamlit**: Visualisation interactive avec graphiques et heatmaps
- **Sauvegarde automatique**: Export des simulations en format JSON détaillé

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/ErwannLesech/Recherche-Operationnelle-2-File-d-attente-dans-le-cadre-d-un-modele-de-moulinettage.git
cd Recherche-Operationnelle-2-File-d-attente-dans-le-cadre-d-un-modele-de-moulinettage

# Installer les dépendances
pip install -r requirements.txt
```

## Usage

### Application interactive Streamlit

L'application interactive permet de configurer et visualiser les simulations en temps réel.

**Lancement:**

```bash
# Méthode recommandée: Script de lancement
python run_app.py

# Alternative: Lancement direct avec Streamlit
streamlit run app/frontend/main_app.py
```

L'application s'ouvre automatiquement dans votre navigateur à l'adresse **http://localhost:8501**

**Fonctionnalités de l'interface:**

1. **Paramètres globaux** (barre latérale):
   - Taux de service μ1 et μ2 (exécution et résultats)
   - Nombre de serveurs (1-20)
   - Taille du buffer K (capacité maximale)

2. **Onglet Personas**:
   - Visualisation des profils étudiants (Prépa Sup/Spé, Ing1/2/3)
   - Comparaison des comportements de soumission
   - Patterns d'usage selon les périodes académiques

3. **Onglet Modèles de files**:
   - Comparaison des différents modèles théoriques (M/M/1, M/M/c, M/M/c/K, M/D/1, M/G/1)
   - Métriques théoriques vs simulation
   - Graphiques de convergence et distributions

4. **Onglet Simulation de rush**:
   - Configuration de périodes de rush (deadlines, examens)
   - Simulation de comportements réalistes par persona
   - Visualisation de l'évolution temporelle

5. **Onglet Optimisation**:
   - Analyse coût/performance
   - Frontière de Pareto
   - Recommandations d'auto-scaling

6. **Onglet Système complet**:
   - Simulation de la chaîne complète (build → test → résultats)
   - Métriques end-to-end
   - Analyse des goulots d'étranglement

**Sauvegarde des simulations:**

Toutes les simulations effectuées sont automatiquement sauvegardées dans le dossier `simulations/` au format JSON avec un horodatage précis:

```
simulations/
├── simulation_YYYYMMDD_HHMMSS.json        # Simulations générales
├── modele_comparative/
│   └── simulation_YYYYMMDD_HHMMSS.json    # Comparaisons de modèles
└── moulinette_simulations/
    └── rush_simulation_YYYYMMDD_HHMMSS.json # Simulations de rush
```

Chaque fichier JSON contient:
- Horodatage de la simulation
- Paramètres utilisés (λ, μ, nombre de serveurs, etc.)
- Résultats détaillés pour chaque modèle et run
- Métriques de performance (temps d'attente, longueur de queue, taux de rejet)
- Indicateurs de stabilité

## Structure du projet

```
app/
├── __init__.py              # Exports principaux
├── models/                  # Modèles de files d'attente
│   ├── base_queue.py        # Classes GenericQueue et ChainQueue
│   └── old/                 # Implémentations historiques (M/M/1, M/M/c, etc.)
├── personas/                # Modélisation des étudiants
│   ├── personas.py          # Types d'étudiants et comportements
│   └── usage_patterns.py    # Patterns temporels (rush, deadline)
├── simulation/              # Moteur de simulation
│   ├── rush_simulator.py    # Simulation de périodes de rush
│   └── moulinette_system.py # Système complet (build + test)
├── optimization/            # Optimisation et scaling
│   ├── cost_optimizer.py    # Optimisation coût/performance
│   └── scaling_advisor.py   # Recommandations d'auto-scaling
├── frontend/                # Interface Streamlit
│   └── main_app.py          # Application web interactive
└── config/                  # Configuration
    └── server_config.py     # Paramètres serveurs et coûts

simulations/                 # Résultats sauvegardés (JSON)
├── simulation_*.json        # Simulations générales
├── modele_comparative/      # Comparaisons de modèles
└── moulinette_simulations/  # Simulations de rush

tests/                       # Tests unitaires
├── test_models.py           # Tests des modèles de files
├── test_personas.py         # Tests des personas
├── test_optimization.py     # Tests de l'optimisation
└── test_queue_models.py     # Tests des modèles de base

docs/                        # Documentation
└── note_coaching.md         # Notes de coaching et méthodologie
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

Avec:
- ρ = λ/μ (utilisation du serveur)
- λ = taux d'arrivée
- μ = taux de service

## Tests

Lancer les tests unitaires:

```bash
# Tous les tests
python -m unittest discover -s tests

# Tests spécifiques
python -m unittest tests.test_queue_models
python -m unittest tests.test_personas
python -m unittest tests.test_optimization
```

## Auteurs

- Florian Ruiz
- Victor Mandelaire
- Abel Aubron
- Nathan Claude
- Aymeric Le Riboter
- Erwann Lesech

## Licence

MIT License - voir [LICENSE](LICENSE)
