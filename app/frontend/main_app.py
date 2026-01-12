"""
Application Streamlit pour la simulation de la moulinette EPITA.

Interface interactive permettant de:
- Visualiser les différents modèles de files d'attente
- Simuler des périodes de rush
- Analyser les coûts et l'optimisation
- Obtenir des recommandations de scaling

Lancer avec: streamlit run app/frontend/main_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Ajouter le chemin parent pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.base_queue import GenericQueue, ChainQueue
from app.personas import PersonaFactory, StudentType
from app.personas.usage_patterns import AcademicPeriod
from app.simulation import RushSimulator, MoulinetteSystem, SimulationConfig, ServerConfig
from app.optimization import CostOptimizer, ScalingAdvisor, CostModel, ScalingPolicy
import json
from datetime import datetime
import os


def run_app():
    """Point d'entrée principal."""
    main()


def main():
    """Application principale."""
    st.set_page_config(
        page_title="Moulinette Simulator - EPITA",
        page_icon="M",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personnalisé
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="main-header">Moulinette Simulator</p>', unsafe_allow_html=True)
    st.markdown("*Simulation et optimisation des files d'attente pour la moulinette EPITA*")
    
    # Sidebar pour les paramètres globaux
    with st.sidebar:
        st.header("Parametres globaux de la moulinette")
        
        mu_rate1 = st.slider(
            "Taux de service μ1 (par serveur/min) (Queue exécution des test-suites)",
            min_value=1.0, max_value=50.0, value=10.0, step=0.5
        )

        mu_rate2 = st.slider(
            "Taux de service μ2 (par serveur/min) (Queue renvoie des résultats)",
            min_value=1.0, max_value=50.0, value=10.0, step=0.5
        )
        
        n_servers = st.slider(
            "Nombre de serveurs",
            min_value=1, max_value=20, value=4
        )
        
        buffer_size = st.slider(
            "Taille du buffer K",
            min_value=10, max_value=50000, value=100, step=20
        )
    
    # Onglets principaux
    tabs = st.tabs([
        "Personas",
        "Simulation - Moulinette",
        "Optimisation - Moulinette",
        "Benchmark - Modeles de files",
        "Auto-scaling",
        "Heatmaps"
    ])
    
    
    with tabs[0]:
        render_personas_tab()
    
    with tabs[1]:
        render_rush_simulation_tab(mu_rate1, mu_rate2, n_servers, buffer_size)
    
    with tabs[2]:
        render_optimization_tab(mu_rate1, buffer_size)
    
    with tabs[3]:
        render_queue_models_tab(mu_rate1, n_servers, buffer_size)

    with tabs[4]:
        render_scaling_tab(mu_rate1, buffer_size, n_servers)

    with tabs[5]:
        render_heatmaps_tab(mu_rate1, buffer_size)


def render_queue_models_tab(mu_rate, n_servers, buffer_size):
    """Onglet de comparaison des modèles de files d'attente."""
    st.header("Comparaison des modeles de files d'attente")

    # Ensure proper type conversion for user inputs
    try:
        mu_rate = float(mu_rate)
        n_servers = int(n_servers)
        buffer_size = int(buffer_size)
    except ValueError:
        st.error("Veuillez entrer des valeurs numériques valides pour les paramètres.")
        return

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("Selection des modeles")
        st.markdown("**Modeles mono-serveur:**")
        show_mm1 = st.checkbox("M/M/1", value=True, key="th_mm1")
        show_md1 = st.checkbox("M/D/1", value=True, key="th_md1")
        show_mg1 = st.checkbox("M/G/1", value=True, key="th_mg1")

        st.markdown("**Modeles multi-serveurs:**")
        show_mmc = st.checkbox("M/M/c", value=True, key="th_mmc")
        show_mdc = st.checkbox("M/D/c", value=True, key="th_mdc")
        show_mgc = st.checkbox("M/G/c", value=True, key="th_mgc")

        cv_squared = st.slider(
            "CV2 service (pour M/G/1 et M/G/c)",
            min_value=0.0, max_value=2.0, value=1.0, step=0.1,
            key="th_cv2"
        )

    with col1:
        # Calculer les métriques théoriques pour chaque modèle
        models_data = []
        service_mean = 1.0 / mu_rate

        lambda_rate = st.slider(
            "Taux d'arrivée λ (soumissions/min)",
            min_value=1.0, max_value=100.0, value=30.0, step=1.0
        )

        mu_rate_model = st.slider("Taux de service μ (par serveur/min)",
            min_value=1.0, max_value=50.0, value=10.0, step=0.5
        )

        # Modèles mono-serveur (condition: lambda < mu)
        if show_mm1 and lambda_rate < mu_rate_model:
            queue = GenericQueue(lambda_rate, mu_rate_model, "M/M/1")
            metrics = queue.compute_theoretical_metrics()
            models_data.append({
                'Modele': 'M/M/1',
                'L (clients)': metrics.L,
                'Lq (en attente)': metrics.Lq,
                'W (min)': metrics.W,
                'Wq (min)': metrics.Wq,
                'rho': metrics.rho,
                'P_blocage': 0.0
            })

        if show_md1 and lambda_rate < mu_rate_model:
            queue = GenericQueue(lambda_rate, mu_rate_model, "M/D/1")
            metrics = queue.compute_theoretical_metrics()
            models_data.append({
                'Modele': 'M/D/1',
                'L (clients)': metrics.L,
                'Lq (en attente)': metrics.Lq,
                'W (min)': metrics.W,
                'Wq (min)': metrics.Wq,
                'rho': metrics.rho,
                'P_blocage': 0.0
            })

        if show_mg1 and lambda_rate < mu_rate_model:
            queue = GenericQueue(lambda_rate, mu_rate_model, "M/G/1")
            metrics = queue.compute_theoretical_metrics()
            models_data.append({
                'Modele': f'M/G/1 (CV2={cv_squared})',
                'L (clients)': metrics.L,
                'Lq (en attente)': metrics.Lq,
                'W (min)': metrics.W,
                'Wq (min)': metrics.Wq,
                'rho': metrics.rho,
                'P_blocage': 0.0
            })

        # Modèles multi-serveurs (condition: lambda < c*mu)
        if show_mmc and lambda_rate < n_servers * mu_rate_model:
            queue = GenericQueue(lambda_rate, mu_rate_model, "M/M/c", n_servers)
            metrics = queue.compute_theoretical_metrics()
            models_data.append({
                'Modele': f'M/M/{n_servers}',
                'L (clients)': metrics.L,
                'Lq (en attente)': metrics.Lq,
                'W (min)': metrics.W,
                'Wq (min)': metrics.Wq,
                'rho': metrics.rho,
                'P_blocage': 0.0
            })

        if show_mdc and lambda_rate < n_servers * mu_rate_model:
            queue = GenericQueue(lambda_rate, mu_rate_model, "M/D/c", n_servers)
            metrics = queue.compute_theoretical_metrics()
            models_data.append({
                'Modele': f'M/D/{n_servers}',
                'L (clients)': metrics.L,
                'Lq (en attente)': metrics.Lq,
                'W (min)': metrics.W,
                'Wq (min)': metrics.Wq,
                'rho': metrics.rho,
                'P_blocage': 0.0
            })

        if show_mgc and lambda_rate < n_servers * mu_rate_model:
            queue = GenericQueue(lambda_rate, mu_rate_model, "M/G/c", n_servers)
            metrics = queue.compute_theoretical_metrics()
            models_data.append({
                'Modele': f'M/G/{n_servers} (CV2={cv_squared})',
                'L (clients)': metrics.L,
                'Lq (en attente)': metrics.Lq,
                'W (min)': metrics.W,
                'Wq (min)': metrics.Wq,
                'rho': metrics.rho,
                'P_blocage': 0.0
            })

        if models_data:
            df = pd.DataFrame(models_data)
            st.subheader("Metriques theoriques")
            st.dataframe(df.style.format({
                'L (clients)': '{:.2f}',
                'Lq (en attente)': '{:.2f}',
                'W (min)': '{:.3f}',
                'Wq (min)': '{:.3f}',
                'rho': '{:.2%}',
                'P_blocage': '{:.4%}'
            }), use_container_width=True)
            
            # Graphique de comparaison
            fig = make_subplots(rows=1, cols=2, subplot_titles=['Temps d\'attente (Wq)', 'Longueur de queue (Lq)'])
            
            fig.add_trace(
                go.Bar(x=df['Modele'], y=df['Wq (min)'], name='Wq'),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(x=df['Modele'], y=df['Lq (en attente)'], name='Lq'),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucun modele selectionne ou parametres invalides (systeme instable)")
    
    # ==========================================
    # SECTION SIMULATION MONTE CARLO COMPARATIVE
    # ==========================================
    st.divider()
    st.subheader("Simulation Monte Carlo - Comparaison de tous les modeles")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        sim_n_customers = st.number_input("Nombre de clients", 100, 50000, 2000, step=100, key="sim_n_customers")
        n_runs = st.number_input("Nombre de runs", 1, 50, 3, key="sim_n_runs")
        save_simulation = st.checkbox("Sauvegarder les resultats", value=True, key="sim_save")
    
    with col2:
        st.markdown("**Modeles a simuler:**")
        sim_mm1 = st.checkbox("M/M/1", value=True, key="sim_mm1")
        sim_md1 = st.checkbox("M/D/1", value=True, key="sim_md1")
        sim_mg1 = st.checkbox("M/G/1", value=True, key="sim_mg1")
        sim_mmc = st.checkbox("M/M/c", value=True, key="sim_mmc")
        sim_mdc = st.checkbox("M/D/c", value=True, key="sim_mdc")
        sim_mgc = st.checkbox("M/G/c", value=True, key="sim_mgc")
    
    with col3:
        run_sim = st.button("Lancer simulation comparative", key="run_sim_compare", type="primary")
        st.markdown("""
        **Note:** Les systemes instables (queue croissante) 
        seront simules normalement sur un nombre fini de clients.
        """)
    
    # Résultats en pleine largeur (en dehors des colonnes)
    if run_sim:
        with st.spinner("Simulation de tous les modeles en cours..."):
            simulation_results = []
            temporal_data = {}  # Pour les graphiques temporels
            service_mean = 1.0 / mu_rate
            service_variance = cv_squared * (service_mean ** 2)
            
            # Configuration des modèles à simuler (tous les modèles cochés, sans restriction)
            models_config = []
            
            # Modèles mono-serveur (lancés sans condition)
            if sim_mm1:
                models_config.append(('M/M/1', GenericQueue(lambda_rate, mu_rate, "M/M/1"), lambda_rate < mu_rate))
            
            if sim_md1:
                models_config.append(('M/D/1', GenericQueue(lambda_rate, mu_rate, "M/D/1"), lambda_rate < mu_rate))
            
            if sim_mg1:
                models_config.append(('M/G/1', GenericQueue(lambda_rate, mu_rate, "M/G/1"), lambda_rate < mu_rate))
            
            # Modèles multi-serveurs (lancés sans condition)
            if sim_mmc:
                models_config.append((f'M/M/{n_servers}', GenericQueue(lambda_rate, mu_rate, "M/M/c", n_servers), lambda_rate < n_servers * mu_rate))
            
            if sim_mdc:
                models_config.append((f'M/D/{n_servers}', GenericQueue(lambda_rate, mu_rate, "M/D/c", n_servers), lambda_rate < n_servers * mu_rate))
            
            if sim_mgc:
                models_config.append((f'M/G/{n_servers}', GenericQueue(lambda_rate, mu_rate, "M/G/c", n_servers), lambda_rate < n_servers * mu_rate))
            
            # Afficher les avertissements pour les systèmes instables
            unstable_models = [name for name, _, is_stable in models_config if not is_stable]
            if unstable_models:
                st.info(f"Systemes instables (rho >= 1): {', '.join(unstable_models)} - La queue va croitre")
            
            if not models_config:
                st.error("Aucun modele selectionne.")
            else:
                progress_bar = st.progress(0)
                total_sims = len(models_config) * n_runs
                current_sim = 0
                
                for model_name, queue, is_stable in models_config:
                    temporal_data[model_name] = {'times': [], 'queue_lengths': [], 'waiting_times': [], 'arrival_times': []}
                    
                    for run in range(n_runs):
                        try:
                            result = queue.simulate(n_customers=sim_n_customers)
                            
                            if len(result.system_times) > 0:
                                simulation_results.append({
                                    'Modele': model_name,
                                    'Run': run + 1,
                                    'Clients servis': result.n_served,
                                    'Temps systeme (min)': float(np.mean(result.system_times)),
                                    'Temps attente (min)': float(np.mean(result.waiting_times)),
                                    'Longueur max queue': int(np.max(result.queue_length_trace)) if len(result.queue_length_trace) > 0 else 0,
                                    'Longueur moy queue': float(np.mean(result.queue_length_trace)) if len(result.queue_length_trace) > 0 else 0,
                                    'Taux rejet (%)': 100 * result.n_rejected / result.n_arrivals if result.n_arrivals > 0 else 0,
                                    'Stable': 'Oui' if is_stable else 'Non'
                                })
                                
                                # Stocker les données temporelles (dernier run uniquement)
                                if run == n_runs - 1:
                                    # Sous-échantillonner pour éviter trop de points, mais garder le dernier point
                                    step = max(1, len(result.time_trace) // 500)
                                    
                                    # Queue lengths avec dernier point
                                    ql = result.queue_length_trace[::step].tolist()
                                    if len(result.queue_length_trace) > 0 and result.queue_length_trace[-1] not in ql[-1:]:
                                        ql.append(float(result.queue_length_trace[-1]))
                                    temporal_data[model_name]['queue_lengths'] = ql
                                    
                                    # Times avec dernier point
                                    times = result.time_trace[::step].tolist() if len(result.time_trace) > 0 else []
                                    if len(result.time_trace) > 0 and result.time_trace[-1] not in times[-1:]:
                                        times.append(float(result.time_trace[-1]))
                                    temporal_data[model_name]['times'] = times
                                    
                                    # Waiting times et arrival times
                                    temporal_data[model_name]['waiting_times'] = result.waiting_times[::step].tolist() if len(result.waiting_times) > step else result.waiting_times.tolist()
                                    temporal_data[model_name]['arrival_times'] = result.arrival_times[::step].tolist() if len(result.arrival_times) > step else result.arrival_times.tolist()
                        except Exception as e:
                            st.warning(f"{model_name} Run {run+1}: {str(e)}")
                        
                        current_sim += 1
                        progress_bar.progress(current_sim / total_sims)
                
                progress_bar.empty()
                
                # Affichage des résultats
                if simulation_results:
                    df_sim = pd.DataFrame(simulation_results)
                    
                    # Résumé statistique
                    st.markdown("### Resume statistique (moyenne +/- ecart-type)")
                    summary = df_sim.groupby('Modele').agg({
                        'Clients servis': ['mean', 'std'],
                        'Temps systeme (min)': ['mean', 'std'],
                        'Temps attente (min)': ['mean', 'std'],
                        'Longueur moy queue': ['mean', 'std'],
                        'Taux rejet (%)': ['mean', 'std']
                    }).round(3)
                    
                    summary_display = pd.DataFrame()
                    for col in ['Clients servis', 'Temps systeme (min)', 'Temps attente (min)', 'Longueur moy queue', 'Taux rejet (%)']:
                        mean_vals = summary[col]['mean'].fillna(0)
                        std_vals = summary[col]['std'].fillna(0)
                        summary_display[col] = mean_vals.astype(str) + ' +/- ' + std_vals.astype(str)
                    
                    st.dataframe(summary_display, use_container_width=True)
                    
                    # Espacement vertical
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # ==========================================
                    # GRAPHIQUES TEMPORELS COMPARATIFS
                    # ==========================================
                    st.markdown("### Evolution temporelle comparee")
                    
                    # Couleurs distinctes pour chaque modèle
                    colors = px.colors.qualitative.Set1
                    color_map = {name: colors[i % len(colors)] for i, name in enumerate(temporal_data.keys())}
                    
                    # Graphique longueur de queue dans le temps
                    fig_temporal = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=['Longueur de queue au cours du temps', 'Temps d\'attente au cours du temps'],
                        vertical_spacing=0.18
                    )
                    
                    for model_name, data in temporal_data.items():
                        if data['queue_lengths'] and data['times']:
                            fig_temporal.add_trace(
                                go.Scatter(
                                    x=data['times'],
                                    y=data['queue_lengths'],
                                    mode='lines',
                                    name=model_name,
                                    line=dict(color=color_map[model_name]),
                                    legendgroup=model_name
                                ),
                                row=1, col=1
                            )
                        
                        if data['waiting_times'] and data.get('arrival_times'):
                            fig_temporal.add_trace(
                                go.Scatter(
                                    x=data['arrival_times'],
                                    y=data['waiting_times'],
                                    mode='lines',
                                    name=model_name,
                                    line=dict(color=color_map[model_name]),
                                    legendgroup=model_name,
                                    showlegend=False
                                ),
                                row=2, col=1
                            )
                    
                    fig_temporal.update_xaxes(title_text="Temps de simulation (min)", row=1, col=1)
                    fig_temporal.update_xaxes(title_text="Temps de simulation (min)", row=2, col=1)
                    fig_temporal.update_yaxes(title_text="Clients en queue", row=1, col=1)
                    fig_temporal.update_yaxes(title_text="Temps attente (min)", row=2, col=1)
                    fig_temporal.update_layout(
                        height=800, 
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        margin=dict(t=80, b=40)
                    )
                    
                    st.plotly_chart(fig_temporal, use_container_width=True)
                    
                    # Espacement vertical
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    
                    # Box plots de comparaison
                    st.markdown("### Distribution des metriques")
                    
                    fig_box = make_subplots(
                        rows=1, cols=3,
                        subplot_titles=['Temps attente (min)', 'Temps systeme (min)', 'Longueur moy queue']
                    )
                    
                    for model in df_sim['Modele'].unique():
                        model_data = df_sim[df_sim['Modele'] == model]
                        color = color_map.get(model, '#1f77b4')
                        
                        fig_box.add_trace(
                            go.Box(y=model_data['Temps attente (min)'], name=model, marker_color=color, showlegend=True),
                            row=1, col=1
                        )
                        fig_box.add_trace(
                            go.Box(y=model_data['Temps systeme (min)'], name=model, marker_color=color, showlegend=False),
                            row=1, col=2
                        )
                        fig_box.add_trace(
                            go.Box(y=model_data['Longueur moy queue'], name=model, marker_color=color, showlegend=False),
                            row=1, col=3
                        )
                    
                    fig_box.update_layout(height=450, margin=dict(t=40, b=40))
                    st.plotly_chart(fig_box, use_container_width=True)
                    
                    # Espacement vertical
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Sauvegarde des résultats
                    if save_simulation:
                        sim_dir = Path(__file__).parent.parent.parent / 'simulations' / 'modele_comparative'
                        sim_dir.mkdir(exist_ok=True)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"simulation_{timestamp}.json"
                        filepath = sim_dir / filename
                        
                        save_data = {
                            'timestamp': timestamp,
                            'parameters': {
                                'lambda_rate': lambda_rate,
                                'mu_rate': mu_rate,
                                'n_servers': n_servers,
                                'cv_squared': cv_squared,
                                'n_customers': sim_n_customers,
                                'n_runs': n_runs
                            },
                            'results': simulation_results,
                            'summary': {
                                model: {
                                    'mean_wait': float(df_sim[df_sim['Modele'] == model]['Temps attente (min)'].mean()),
                                    'std_wait': float(df_sim[df_sim['Modele'] == model]['Temps attente (min)'].std()),
                                    'mean_system': float(df_sim[df_sim['Modele'] == model]['Temps systeme (min)'].mean()),
                                    'mean_queue_length': float(df_sim[df_sim['Modele'] == model]['Longueur moy queue'].mean())
                                }
                                for model in df_sim['Modele'].unique()
                            }
                        }
                        
                        with open(filepath, 'w', encoding='utf-8') as f:
                            json.dump(save_data, f, indent=2, ensure_ascii=False)
                        
                        st.success(f"Simulation sauvegardee: {filepath}")
                    
                    # Tableau détaillé
                    with st.expander("Voir les resultats detailles de tous les runs"):
                        st.dataframe(df_sim, use_container_width=True)
                else:
                    st.error("Aucune simulation valide. Verifiez les parametres et conditions de stabilite.")


def render_personas_tab():
    """Onglet des personas étudiants."""
    st.header("Personas Etudiants")
    
    # Créer les personas
    personas = PersonaFactory.create_all_personas()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Comportement par type d'etudiant")
        
        # Tableau des caractéristiques
        data = []
        for student_type, persona in personas.items():
            data.append({
                'Type': persona.name,
                'Taux base (sub/h)': persona.base_submission_rate,
                'Variance (CV2)': f"{persona.variance_coefficient:.2f}",
                'Procrastination': f"{persona.procrastination_level:.0%}",
                'Heures pic': ', '.join(f"{h}h" for h in persona.peak_hours[:3])
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
    
    with col2:
        st.subheader("Taux d'arrivée sur 24h")
        
        # Générer les courbes de taux d'arrivée
        hours = list(range(24))
        fig = go.Figure()
        
        for student_type, persona in personas.items():
            rates = [persona.get_arrival_rate(h) for h in hours]
            fig.add_trace(go.Scatter(
                x=hours, y=rates,
                mode='lines+markers',
                name=persona.name
            ))
        
        fig.update_layout(
            xaxis_title='Heure',
            yaxis_title='Taux d\'arrivée (sub/h)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Section deadline
    st.subheader("Impact d'une deadline")
    
    hours_to_deadline = st.slider("Heures avant deadline", 0.0, 48.0, 24.0, 0.5)
    
    # Calculer l'impact
    impact_data = []
    for student_type, persona in personas.items():
        base_rate = persona.get_arrival_rate(14)  # 14h comme référence
        deadline_rate = persona.get_arrival_rate(14, hours_to_deadline=hours_to_deadline)
        multiplier = deadline_rate / base_rate if base_rate > 0 else 1.0
        
        impact_data.append({
            'Type': persona.name,
            'Taux normal': base_rate,
            'Taux avec deadline': deadline_rate,
            'Multiplicateur': multiplier
        })
    
    fig = go.Figure()
    df_impact = pd.DataFrame(impact_data)
    
    fig.add_trace(go.Bar(
        x=df_impact['Type'],
        y=df_impact['Taux normal'],
        name='Normal'
    ))
    fig.add_trace(go.Bar(
        x=df_impact['Type'],
        y=df_impact['Taux avec deadline'],
        name=f'Deadline dans {hours_to_deadline}h'
    ))
    
    fig.update_layout(barmode='group', height=350)
    st.plotly_chart(fig, use_container_width=True)

def render_rush_simulation_tab(mu_rate1, mu_rate2, n_servers, buffer_size):
    """Onglet de simulation de rush basé sur les Personas."""
    st.header("Simulation - Moulinette EPITA")

    # [CODE EXISTANT: Configuration MoulinetteSystem et Scénario]
    moulinette = MoulinetteSystem()
    queue1 = GenericQueue(lambda_rate=1, mu_rate=mu_rate1, c=n_servers, kendall_notation=f'M/M/{n_servers}', K=buffer_size)
    queue2 = GenericQueue(lambda_rate=1, mu_rate=mu_rate2, c=1, kendall_notation="M/M/1", K=buffer_size)
    moulinette.configure_chain([queue1, queue2])

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Scénario")
        sim_mode = st.radio("Mode de génération de charge", ["Synthétique (Gaussienne)", "Basé sur les Personas (Réaliste)"])
        duration = st.slider("Durée simulation (h)", 1.0, 48.0, 24.0, 1.0)
        
        if sim_mode == "Synthétique (Gaussienne)":
             base_lambda = st.slider("Taux de base λ (sub/h)", 10.0, 200.0, 50.0, 10.0)
             peak_multiplier = st.slider("Multiplicateur de pic", 1.5, 10.0, 3.0, 0.5)
             rush_center = st.slider("Pic du rush (fraction)", 0.1, 0.9, 0.7, 0.1)
             def lambda_profile_func(t):
                 x = t / duration
                 width = 0.15
                 rush = np.exp(-((x - rush_center) ** 2) / (2 * width ** 2))
                 return base_lambda * (1 + rush * (peak_multiplier - 1))
        else:
            st.info("La charge est calculée en fonction de la population étudiante définie dans 'Personas'.")
            hours_to_deadline = st.slider("Heures avant deadline (au début)", duration, duration + 48.0, duration, 1.0)
            start_hour = st.slider("Heure de début de simulation", 0, 23, 8)
            
            # Récupérer les personas
            personas = PersonaFactory.create_all_personas()
            
            # Permettre de désactiver certains types d'étudiants pour tester
            selected_types = st.multiselect(
                "Populations actives",
                [p.student_type.name for p in personas.values()],
                default=[p.student_type.name for p in personas.values()]
            )
            
            # Fonction lambda complexe (basée sur les agents)
            def lambda_profile_func(t):
                current_hour = (start_hour + int(t)) % 24
                # Temps restant avant deadline qui diminue au fur et à mesure que t avance
                remaining_time = max(0, hours_to_deadline - t)
                
                total_rate = 0.0
                for p in personas.values():
                    if p.student_type.name in selected_types:
                        total_rate += p.get_arrival_rate(
                            hour=current_hour, 
                            is_weekend=False, 
                            hours_to_deadline=remaining_time
                        )
                return total_rate
        
        st.divider()
        st.subheader("Configuration Auto-scaling")
        enable_autoscaling = st.checkbox("Activer l'auto-scaling dynamique")
        
        scaling_policy = None
        if enable_autoscaling:
            with st.expander("Paramètres de scaling", expanded=True):
                as_min_servers = st.number_input("Min Serveurs", 1, n_servers, 1)
                as_max_servers = st.number_input("Max Serveurs", n_servers, 50, 20)
                
                col_up, col_down = st.columns(2)
                as_up_thresh = col_up.slider("Seuil Scale UP (%)", 50, 95, 80, 5) / 100.0
                as_down_thresh = col_down.slider("Seuil Scale DOWN (%)", 10, 60, 30, 5) / 100.0
                
                # Création de la policy
                scaling_policy = ScalingPolicy(
                    scale_up_threshold=as_up_thresh,
                    scale_down_threshold=as_down_thresh,
                    min_servers=as_min_servers,
                    max_servers=as_max_servers,
                    scale_up_increment=2,
                    scale_down_increment=1
                )

        save_rush = st.checkbox("Sauvegarder la simulation Rush", value=True)
        run_button = st.button("Lancer la simulation", type="primary")

    with col2:
        st.subheader("Charge prévue λ(t)")
        t_preview = np.linspace(0, duration, 100)
        lambda_values = [lambda_profile_func(t) for t in t_preview]
        
        fig_profile = go.Figure()
        fig_profile.add_trace(go.Scatter(
            x=t_preview, y=lambda_values,
            mode='lines', fill='tozeroy',
            name='Taux d\'arrivée',
            line=dict(color='#1f77b4')
        ))
        
        # Ajouter une ligne rouge pour la capacité théorique totale du système
        total_capacity = (n_servers) * mu_rate1 * 60 # 4 serveurs * mu (converti en heure si mu est en min)
        # Note: Dans ton code mu_rate semble être par minute, donc * 60 pour l'heure
        
        fig_profile.add_hline(
            y=total_capacity, 
            line_dash="dash", line_color="red",
            annotation_text="Capacité théorique max"
        )
        
        fig_profile.update_layout(
            xaxis_title="Temps (h)",
            yaxis_title="Soumissions / heure",
            height=300,
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_profile, use_container_width=True)

    if run_button:
        # 1. EXECUTION STANDARD (BASELINE)
        with st.spinner("Simulation Baseline (Configuration statique)..."):
            queue1.c = n_servers
            report_baseline = moulinette.simulate_evolving(lambda_profile_func, duration, scaling_policy=None)
        
        # 2. EXECUTION SCALED (SI ACTIVE)
        report_scaled = None
        if enable_autoscaling and scaling_policy:
            with st.spinner("Simulation avec Auto-scaling..."):
                # On réinitialise pour le 2eme run
                moulinette = MoulinetteSystem() 
                q1_scaled = GenericQueue(lambda_rate=1, mu_rate=mu_rate1, c=n_servers, kendall_notation=f'M/M/{n_servers}', K=buffer_size)
                q2_scaled = GenericQueue(lambda_rate=1, mu_rate=mu_rate2, c=1, kendall_notation="M/M/1", K=buffer_size)
                moulinette.configure_chain([q1_scaled, q2_scaled])
                
                report_scaled = moulinette.simulate_evolving(lambda_profile_func, duration, scaling_policy=scaling_policy)

        # --- AFFICHAGE RESULTATS COMPARES ---
        st.divider()
        st.subheader("Résultats de la simulation")

        if report_scaled:
            # MODE COMPARATIF
            col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
            
            # Helper pour afficher delta
            def delta_metric(label, val_base, val_scaled, unit=""):
                delta = val_scaled - val_base
                delta_color = "normal"
                if "Attente" in label or "Rejet" in label:
                    delta_color = "inverse" # Moins c'est mieux
                
                col_kpi1.metric(f"{label} (Base)", f"{val_base:.2f}{unit}")
                col_kpi2.metric(f"{label} (Auto-scale)", f"{val_scaled:.2f}{unit}", f"{delta:.2f}{unit}", delta_color=delta_color)

            delta_metric("Temps Attente Moy.", report_baseline.avg_waiting_time * 60, report_scaled.avg_waiting_time * 60, " min")
            delta_metric("Taux Rejet", report_baseline.rejection_rate * 100, report_scaled.rejection_rate * 100, " %")
            
            # Coût estimatif (serveur * heures)
            # Baseline: n_servers * duration
            # Scaled: somme(server_count_trace * step_hours)
            step_h = duration / len(report_scaled.time_series["server_count"])
            avg_servers_scaled = np.mean(report_scaled.time_series["server_count"])
            col_kpi3.metric("Serveurs Moyens", f"{n_servers}", f"{avg_servers_scaled:.1f}", delta_color="off")

            # --- GRAPHIQUE AUTO-SCALING ---
            st.subheader("Dynamique de l'Auto-scaling")
            
            # On crée un graph avec 2 axes Y
            fig_scaling = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Axe X commun
            times = report_scaled.time_series["time"]
            
            # Trace 1 : Nombre de serveurs (Area chart)
            fig_scaling.add_trace(
                go.Scatter(
                    x=times, 
                    y=report_scaled.time_series["server_count"],
                    name="Nombre de serveurs actifs",
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='rgba(46, 204, 113, 0.6)'), # Vert
                    step='hv' # Step line pour montrer les paliers
                ),
                secondary_y=False
            )
            
            # Trace 2 : Queue Length Scaled (Ligne)
            fig_scaling.add_trace(
                go.Scatter(
                    x=times,
                    y=report_scaled.simulation_results[0].queue_length_trace, # Queue 1 length
                    name="Queue (Avec Scaling)",
                    mode='lines',
                    line=dict(color='blue', width=2)
                ),
                secondary_y=True
            )

            # Trace 3 : Queue Length Baseline (Pointillés pour comparaison)
            if len(report_baseline.time_series.get("time", [])) > 0:
                 fig_scaling.add_trace(
                    go.Scatter(
                        x=report_baseline.simulation_results[0].time_trace,
                        y=report_baseline.simulation_results[0].queue_length_trace,
                        name="Queue (Sans Scaling)",
                        mode='lines',
                        line=dict(color='red', dash='dot')
                    ),
                    secondary_y=True
                )

            fig_scaling.update_yaxes(title_text="Nombre de Serveurs", secondary_y=False, range=[0, scaling_policy.max_servers + 2])
            fig_scaling.update_yaxes(title_text="Clients en attente", secondary_y=True)
            fig_scaling.update_xaxes(title_text="Temps (h)")
            fig_scaling.update_layout(title="Adaptation des ressources vs Charge", height=500)
            
            
            st.plotly_chart(fig_scaling, use_container_width=True)

        else:
            # KPIs Globaux
            st.divider()
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Temps Attente Moy.", f"{report_baseline.avg_waiting_time*60:.1f} min")
            kpi2.metric("Queue Max (Clients)", f"{int(max([np.max(r.queue_length_trace) if len(r.queue_length_trace) > 0 else 0 for r in report_baseline.simulation_results]))}")
            kpi3.metric("Taux de Rejet", f"{report_baseline.rejection_rate:.2%}")
            kpi4.metric("Total Servis", f"{sum(r.n_served for r in report_baseline.simulation_results)}")

            # Graphiques temporels
            st.subheader("Dynamique des Queues")
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                subplot_titles=("Longueur des files d'attente", "Temps d'attente par client"))

            colors = ['#EF553B', '#FFA15A'] # Couleurs Plotly

            fig.update_xaxes(
                row=1, col=1,
                type="linear",
                showticklabels=True,
                range=[0, duration],
                tickmode="linear",
                dtick=2
            )
            fig.update_xaxes(
                row=2, col=1,
                type="linear",
                showticklabels=True,
                range=[0, duration],
                tickmode="linear",
                dtick=2
            )

            # Trace Queue Lengths
            for i, res in enumerate(report_baseline.simulation_results):
                if len(res.time_trace) > 0:
                    fig.add_trace(go.Scatter(
                        x=res.time_trace, 
                        y=res.queue_length_trace,
                        name=f"Queue {i+1} ({moulinette._queue_chain.queues[i].kendall_notation})",
                        mode='lines',
                        line=dict(width=2)
                    ), row=1, col=1)

            # Trace Waiting Times (Scatter)
            for i, res in enumerate(report_baseline.simulation_results):
                if len(res.arrival_times) > 0 and len(res.waiting_times) > 0:
                    # On sous-échantillonne si trop de points pour la performance
                    step = max(1, len(res.waiting_times) // 500)
                    fig.add_trace(go.Scatter(
                        x=res.arrival_times[::step], 
                        y=res.waiting_times[::step] * 60, # conversion en minutes
                        name=f"Attente Q{i+1}",
                        mode='markers',
                        marker=dict(size=4, opacity=0.5)
                    ), row=2, col=1)

            fig.update_layout(height=600, showlegend=True)
            fig.update_yaxes(title_text="Clients", row=1, col=1)
            fig.update_yaxes(title_text="Minutes", row=2, col=1)
            fig.update_xaxes(title_text="Temps de simulation (h)", row=2, col=1)
            fig.update_xaxes(title_text="Temps de simulation (h)", row=1, col=1)
            
            st.plotly_chart(fig, use_container_width=True)

        if save_rush:
            sim_dir = Path(__file__).parent.parent.parent / "simulations" / "moulinette_simulations"
            sim_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rush_simulation_{timestamp}.json"
            filepath = sim_dir / filename

            # Agrégations globales simples et robustes
            total_arrivals = sum(r.n_arrivals for r in report_baseline.simulation_results)
            total_served = sum(r.n_served for r in report_baseline.simulation_results)
            total_rejected = sum(r.n_rejected for r in report_baseline.simulation_results)

            all_waiting = np.concatenate([r.waiting_times for r in report_baseline.simulation_results if len(r.waiting_times) > 0]) \
                if any(len(r.waiting_times) > 0 for r in report_baseline.simulation_results) else np.array([])

            all_system = np.concatenate([r.system_times for r in report_baseline.simulation_results if len(r.system_times) > 0]) \
                if any(len(r.system_times) > 0 for r in report_baseline.simulation_results) else np.array([])

            save_data = {
                "timestamp": timestamp,

                # --- paramètres simul —
                "rush_parameters": {
                    "mu_rate_exec": mu_rate1,
                    "mu_rate_results": mu_rate2,
                    "n_servers": n_servers,
                    "buffer_size": buffer_size,
                    "duration_hours": duration,
                    "mode": sim_mode
                },

                # --- résultats globaux —
                "system_results": {
                    "total_arrivals": int(total_arrivals),
                    "total_served": int(total_served),
                    "total_rejected": int(total_rejected),
                    "reject_rate_percent":
                        float(100 * total_rejected / total_arrivals) if total_arrivals > 0 else 0.0,
                    "avg_waiting_time_minutes":
                        float(np.mean(all_waiting) * 60) if len(all_waiting) > 0 else 0.0,
                    "avg_system_time_minutes":
                        float(np.mean(all_system) * 60) if len(all_system) > 0 else 0.0
                },

                # --- détails par queue —
                "queues": [
                    {
                        "index": i + 1,
                        "kendall": moulinette._queue_chain.queues[i].kendall_notation,
                        "served": int(r.n_served),
                        "arrivals": int(r.n_arrivals),
                        "rejected": int(r.n_rejected),
                        "avg_waiting_minutes":
                            float(np.mean(r.waiting_times) * 60) if len(r.waiting_times) > 0 else 0.0,
                        "max_queue_length":
                            int(np.max(r.queue_length_trace)) if len(r.queue_length_trace) > 0 else 0
                    }
                    for i, r in enumerate(report_baseline.simulation_results)
                ],

                # --- timeline globale si existante —
                "timeline": {
                    "queues": [
                        {
                            "queue": i + 1,
                            "time": r.time_trace.tolist() if len(r.time_trace) > 0 else [],
                            "queue_length": r.queue_length_trace.tolist() if len(r.queue_length_trace) > 0 else []
                        }
                        for i, r in enumerate(report_baseline.simulation_results)
                    ]
                }
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            st.success(f"Simulation Rush sauvegardée dans : {filepath}")

def render_optimization_tab(mu_rate, buffer_size):
    """Onglet d'optimisation coût/performance."""
    st.header("Optimisation Cout / Performance")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Modèle de coût")
        
        cost_per_server = st.number_input("Coût/serveur/heure (€)", 0.1, 10.0, 0.5, 0.1)
        fixed_cost = st.number_input("Coût fixe/heure (€)", 0.0, 100.0, 5.0, 1.0)
        penalty_per_min = st.number_input("Pénalité/min attente (€)", 0.0, 1.0, 0.05, 0.01)
        
        alpha = st.slider("α (poids QoS)", 0.0, 1.0, 0.5, 0.05)
        beta = 1 - alpha
        st.write(f"β (poids coût): {beta:.2f}")
        
        max_servers = st.slider("Serveurs max à tester", 5, 30, 15)
        
        optimize_btn = st.button("Optimiser")
    
    with col2:
        if optimize_btn:
            with st.spinner("Recherche de la configuration optimale..."):
                cost_model = CostModel(
                    cost_per_server_hour=cost_per_server,
                    fixed_infrastructure_cost=fixed_cost,
                    cost_per_waiting_minute=penalty_per_min
                )
                
                optimizer = CostOptimizer(
                    lambda_rate=lambda_rate,
                    mu_rate=mu_rate,
                    cost_model=cost_model
                )
                
                # Rechercher l'optimum
                result = optimizer.optimize(
                    alpha=alpha,
                    c_range=(1, max_servers),
                    K_range=(buffer_size, buffer_size)
                )
                
                # Afficher le resultat
                st.success(f"Configuration optimale trouvee!")
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Serveurs optimaux", result.optimal_servers)
                col_b.metric("Buffer optimal", result.optimal_buffer)
                col_c.metric("Score objectif", f"{result.objective_value:.4f}")
                
                col_d, col_e, col_f = st.columns(3)
                col_d.metric("Utilisation rho", f"{result.utilization:.0%}")
                col_e.metric("Taux rejet", f"{result.rejection_rate:.2%}")
                col_f.metric("Wq estime", f"{result.avg_waiting_time:.2f} min")
                
                # Frontière Pareto
                st.subheader("Analyse cout vs performance")
                
                # Calculer pour différents nombres de serveurs
                servers_range = list(range(1, max_servers + 1))
                costs = []
                waiting_times = []
                
                for n in servers_range:
                    try:
                        queue = GenericQueue(lambda_rate, mu_rate, "M/M/c")
                        metrics = queue.compute_theoretical_metrics()
                        wq = metrics.Wq
                        
                        server_cost = n * cost_per_server + fixed_cost
                        wait_cost = wq * penalty_per_min * lambda_rate * 60;
                        
                        costs.append(server_cost + wait_cost)
                        waiting_times.append(wq)
                    except:
                        costs.append(None)
                        waiting_times.append(None)

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Scatter(x=servers_range, y=costs, name='Coût total/h', line=dict(color='blue')),
                    secondary_y=False
                )
                fig.add_trace(
                    go.Scatter(x=servers_range, y=waiting_times, name='Temps attente (min)', line=dict(color='red')),
                    secondary_y=True
                )
                fig.add_vline(x=result.optimal_servers, line_dash="dash", line_color="green", 
                             annotation_text=f"Optimal: {result.optimal_servers}")
                
                fig.update_layout(
                    title='Coût et temps d\'attente vs nombre de serveurs',
                    xaxis_title='Nombre de serveurs',
                    height=400
                )
                fig.update_yaxes(title_text="Coût (€/h)", secondary_y=False)
                fig.update_yaxes(title_text="Wq (min)", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)


def render_scaling_tab(mu_rate, buffer_size, current_servers):
    """Onglet de recommandations d'auto-scaling."""
    st.header("Recommandations d'Auto-scaling")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Politique de scaling")
        
        scale_up_threshold = st.slider("Seuil scale up (rho)", 0.5, 0.95, 0.75, 0.05, key="scale_up_th")
        scale_down_threshold = st.slider("Seuil scale down (rho)", 0.1, 0.5, 0.35, 0.05, key="scale_down_th")
        emergency_threshold = st.slider("Seuil urgence (rho)", 0.8, 1.0, 0.95, 0.01, key="emergency_th")
        
        cost_per_server = st.number_input("Cout/serveur/h", 0.1, 10.0, 0.5, key="scaling_cost")
        
        current_hour = st.slider("Heure actuelle", 0, 23, 14, key="scaling_hour")
        hours_to_deadline = st.slider("Heures avant deadline", 0.0, 48.0, 24.0, 0.5, key="scaling_deadline")
        
        get_reco_btn = st.button("Obtenir recommandation")
    
    with col2:
        if get_reco_btn:
            # Créer le conseiller
            policy = ScalingPolicy(
                scale_up_threshold=scale_up_threshold,
                scale_down_threshold=scale_down_threshold,
                emergency_threshold=emergency_threshold
            )
            
            advisor = ScalingAdvisor(
                mu_rate=mu_rate,
                buffer_size=buffer_size,
                policy=policy,
                cost_per_server_hour=cost_per_server
            )
            
            # Créer les personas
            personas = PersonaFactory.create_all_personas()
            
            # Obtenir la recommandation
            reco = advisor.get_recommendation(
                current_servers=current_servers,
                personas=personas,
                hour=current_hour,
                hours_to_deadline=hours_to_deadline
            )
            
            # Afficher la recommandation
            if reco.action.value == "none":
                st.success("[OK] " + reco.reason)
            elif reco.action.value == "scale_up":
                st.warning("[UP] " + reco.reason)
            elif reco.action.value == "emergency":
                st.error("[URGENT] " + reco.reason)
            else:
                st.info("[DOWN] " + reco.reason)
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Serveurs actuels", reco.current_servers)
            col_b.metric("Serveurs recommandés", reco.recommended_servers, reco.delta_servers)
            col_c.metric("Confiance", f"{reco.confidence:.0%}")
            
            col_d, col_e, col_f = st.columns(3)
            col_d.metric("Charge actuelle", f"{reco.current_load:.0%}")
            col_e.metric("Charge prévue", f"{reco.predicted_load:.0%}")
            col_f.metric("Impact coût", f"{reco.estimated_cost_impact:+.2f}€/h")
            
            # Planification sur 24h
            st.subheader("Planning de scaling sur 24h")
            
            analysis = advisor.analyze_scaling_opportunities(
                current_servers=current_servers,
                personas=personas,
                hours=24
            )
            
            st.info(analysis['recommendation'])
            
            # Graphique du planning
            schedule = analysis['schedule']
            hours = list(schedule.keys())
            servers = list(schedule.values())
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=hours, y=servers, name='Serveurs recommandés'))
            fig.add_hline(y=current_servers, line_dash="dash", line_color="red",
                         annotation_text=f"Actuel: {current_servers}")
            
            fig.update_layout(
                title='Nombre de serveurs recommandé par heure',
                xaxis_title='Heure',
                yaxis_title='Serveurs',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)


def render_heatmaps_tab(mu_rate, buffer_size):
    """Onglet des heatmaps de sensibilité."""
    st.header("Heatmaps de sensibilite")
    
    st.markdown("""
    Visualisation de l'impact des hyperparamètres sur les métriques clés.
    Ces heatmaps permettent d'identifier les zones de fonctionnement optimal.
    """)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        metric = st.selectbox(
            "Métrique à visualiser",
            ["Temps d'attente Wq", "Longueur queue Lq", "Probabilité blocage", "Coût total"]
        )
        
        cost_per_server = st.number_input("Coût/serveur/h (€)", 0.1, 5.0, 0.5, key="heatmap_cost")
        
        resolution = st.slider("Résolution", 10, 50, 25)
        
        generate_btn = st.button("Generer heatmap")
    
    with col2:
        if generate_btn:
            with st.spinner("Génération de la heatmap..."):
                # Paramètres
                lambda_range = np.linspace(5, 100, resolution)
                server_range = np.arange(1, 16)
                
                # Créer la matrice
                Z = np.zeros((len(server_range), len(lambda_range)))
                
                for i, n_servers in enumerate(server_range):
                    for j, lambda_rate in enumerate(lambda_range):
                        try:
                            queue = GenericQueue(lambda_rate, mu_rate, "M/M/c")
                            metrics = queue.compute_theoretical_metrics()
                            
                            if metric == "Temps d'attente Wq":
                                Z[i, j] = min(metrics.Wq, 10)  # Cap pour visualisation
                            elif metric == "Longueur queue Lq":
                                Z[i, j] = min(metrics.Lq, 50)
                            elif metric == "Probabilité blocage":
                                Z[i, j] = metrics.Pk
                            else:  # Coût total
                                server_cost = n_servers * cost_per_server
                                wait_cost = metrics.Wq * 0.05 * lambda_rate * 60
                                Z[i, j] = min(server_cost + wait_cost, 50)
                        except:
                            Z[i, j] = np.nan
                
                # Créer la heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=Z,
                    x=lambda_range,
                    y=server_range,
                    colorscale='RdYlGn_r' if 'Coût' in metric or 'attente' in metric else 'RdYlGn_r',
                    colorbar=dict(title=metric)
                ))
                
                fig.update_layout(
                    title=f'Heatmap: {metric}',
                    xaxis_title='Taux d\'arrivée λ (sub/min)',
                    yaxis_title='Nombre de serveurs',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Interprétation
                st.markdown("""
                **Lecture de la heatmap:**
                - Vert = bonnes performances / faible cout
                - Rouge = mauvaises performances / cout eleve
                - Identifiez la frontiere de stabilite (rho < 1)
                - Trouvez le compromis optimal entre ressources et performances
                """)


if __name__ == "__main__":
    main()
