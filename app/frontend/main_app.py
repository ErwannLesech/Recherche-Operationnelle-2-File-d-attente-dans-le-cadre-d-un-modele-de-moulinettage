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
        "Benchmark - Modeles de files"
    ])
    
    
    with tabs[0]:
        render_personas_tab()
    
    with tabs[1]:
        render_rush_simulation_tab(mu_rate1, mu_rate2, n_servers, buffer_size)
    
    with tabs[2]:
        render_optimization_tab(buffer_size)
    
    with tabs[3]:
        render_queue_models_tab(mu_rate1, n_servers, buffer_size)

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
                        sim_dir.mkdir(parents=True, exist_ok=True)
                        
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

    # 1. Configuration du Système (Chaîne de 2 queues)
    moulinette = MoulinetteSystem()
    
    # Configuration des serveurs (capacité totale répartie ou ajustée)
    # Queue 1 : Pré-tri / Compilation (M/M/c)
    queue1 = GenericQueue(lambda_rate=1, mu_rate=mu_rate1, c=n_servers, kendall_notation=f'M/M/{n_servers}', K=buffer_size)
    # Queue 2 : Tests / Validation (M/M/1 - goulot d'étranglement typique)
    queue2 = GenericQueue(lambda_rate=1, mu_rate=mu_rate2, c=1, kendall_notation="M/M/1", K=buffer_size)
    
    moulinette.configure_chain([queue1, queue2])

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Scénario")
        
        # Choix du mode de simulation
        sim_mode = st.radio(
            "Mode de génération de charge",
            ["Synthétique (Gaussienne)", "Basé sur les Personas (Réaliste)"]
        )
        
        duration = st.slider("Durée simulation (h)", 1.0, 48.0, 24.0, 1.0)
        
        if sim_mode == "Synthétique (Gaussienne)":
            base_lambda = st.slider("Taux de base λ (sub/h)", 10.0, 200.0, 50.0, 10.0)
            peak_multiplier = st.slider("Multiplicateur de pic", 1.5, 10.0, 3.0, 0.5)
            rush_center = st.slider("Pic du rush (fraction du temps)", 0.1, 0.9, 0.7, 0.1)
            
            # Fonction lambda simple (mathématique)
            def lambda_profile_func(t):
                x = t / duration
                width = 0.15
                rush = np.exp(-((x - rush_center) ** 2) / (2 * width ** 2))
                return base_lambda * (1 + rush * (peak_multiplier - 1))
                
        else: # Mode Personas
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

        save_rush = st.checkbox("Sauvegarder la simulation Rush", value=True)
        run_button = st.button("Lancer la simulation", type="primary")

    with col2:
        # Prévisualisation de la courbe de charge
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
        with st.spinner("Simulation du système de moulinette en cours..."):
            try:
                # Lancement de la simulation via le système évolutif
                # On passe directement la fonction lambda_profile_func
                report = moulinette.simulate_evolving(lambda_profile_func, duration)

                # --- AFFICHAGE DES RÉSULTATS (Code existant conservé et nettoyé) ---
                
                # KPIs Globaux
                st.divider()
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Temps Attente Moy.", f"{report.avg_waiting_time*60:.1f} min")
                kpi2.metric("Queue Max (Clients)", f"{int(max([np.max(r.queue_length_trace) if len(r.queue_length_trace) > 0 else 0 for r in report.simulation_results]))}")
                kpi3.metric("Taux de Rejet", f"{report.rejection_rate:.2%}")
                kpi4.metric("Total Servis", f"{sum(r.n_served for r in report.simulation_results)}")

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
                for i, res in enumerate(report.simulation_results):
                    if len(res.time_trace) > 0:
                        fig.add_trace(go.Scatter(
                            x=res.time_trace, 
                            y=res.queue_length_trace,
                            name=f"Queue {i+1} ({moulinette._queue_chain.queues[i].kendall_notation})",
                            mode='lines',
                            line=dict(width=2)
                        ), row=1, col=1)

                # Trace Waiting Times (Scatter)
                for i, res in enumerate(report.simulation_results):
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
                    sim_dir.mkdir(parents=True, exist_ok=True)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"rush_simulation_{timestamp}.json"
                    filepath = sim_dir / filename

                    # Agrégations globales simples et robustes
                    total_arrivals = sum(r.n_arrivals for r in report.simulation_results)
                    total_served = sum(r.n_served for r in report.simulation_results)
                    total_rejected = sum(r.n_rejected for r in report.simulation_results)

                    all_waiting = np.concatenate([r.waiting_times for r in report.simulation_results if len(r.waiting_times) > 0]) \
                        if any(len(r.waiting_times) > 0 for r in report.simulation_results) else np.array([])

                    all_system = np.concatenate([r.system_times for r in report.simulation_results if len(r.system_times) > 0]) \
                        if any(len(r.system_times) > 0 for r in report.simulation_results) else np.array([])

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
                            for i, r in enumerate(report.simulation_results)
                        ],

                        # --- timeline globale si existante —
                        "timeline": {
                            "queues": [
                                {
                                    "queue": i + 1,
                                    "time": r.time_trace.tolist() if len(r.time_trace) > 0 else [],
                                    "queue_length": r.queue_length_trace.tolist() if len(r.queue_length_trace) > 0 else []
                                }
                                for i, r in enumerate(report.simulation_results)
                            ]
                        }
                    }

                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(save_data, f, indent=2, ensure_ascii=False)

                    st.success(f"Simulation Rush sauvegardée dans : {filepath}")

            except Exception as e:
                st.error(f"Erreur durant la simulation: {str(e)}")
                # Afficher la stacktrace pour le debug
                import traceback
                st.code(traceback.format_exc())

def render_optimization_tab(buffer_size):
    """Onglet d'optimisation coût/performance avec heatmaps."""
    st.header("Optimisation Cout / Performance")
    
    st.markdown("""
    Visualisation de l'impact des paramètres sur les métriques clés.
    Ces heatmaps permettent d'identifier les zones de fonctionnement optimal.
    """)
    
    # --- Section de configuration du taux d'arrivée ---
    st.subheader("Configuration du taux d'arrivée")
    
    col_mode1, col_mode2 = st.columns([1, 2])
    
    with col_mode1:
        input_mode = st.radio(
            "Mode de saisie",
            ["Manuel", "Depuis Personas"],
            key="optim_input_mode"
        )
    
    with col_mode2:
        if input_mode == "Manuel":
            lambda_rate = st.number_input(
                "Taux d'arrivée λ (soumissions/min)",
                min_value=1.0, max_value=200.0, value=30.0, step=1.0,
                key="optim_lambda_manual"
            )
            selected_personas_names = ["Manuel"]
        else:
            # Mode Personas - sélection multiple (même logique que Simulation Rush)
            # Récupérer tous les personas
            personas = PersonaFactory.create_all_personas()
            
            selected_types = st.multiselect(
                "Populations actives",
                [p.student_type.name for p in personas.values()],
                default=[p.student_type.name for p in personas.values()],
                key="optim_persona_multiselect"
            )
            
            if not selected_types:
                st.warning("Veuillez sélectionner au moins un persona.")
                lambda_rate = 0.0
                selected_personas_names = []
            else:
                # Calculer le taux d'arrivée combiné de tous les personas sélectionnés
                lambda_rate = 0.0
                total_population = 0
                details = []
                selected_personas_names = []
                
                for p in personas.values():
                    if p.student_type.name in selected_types:
                        # Taux d'arrivée à 14h (heure de référence typique)
                        persona_lambda = p.get_arrival_rate(hour=14) / 60.0  # Convertir en sub/min
                        lambda_rate += persona_lambda
                        total_population += p.population_size
                        selected_personas_names.append(p.name)
                        details.append(f"{p.name}: {persona_lambda:.2f} sub/min ({p.population_size} étudiants)")
                
                st.info(f"**Taux d'arrivée combiné**: λ = {lambda_rate:.2f} soumissions/min")
                st.caption(f"Population totale: {total_population} étudiants")
                with st.expander("Détail par persona"):
                    for detail in details:
                        st.write(f"• {detail}")
    
    st.divider()
    
    # --- Section des paramètres de génération ---
    st.subheader("Paramètres des Heatmaps")
    
    # Paramètres de coût
    st.markdown("**Modèle de coût**")
    col_cost1, col_cost2, col_cost3, col_cost4 = st.columns([1, 1, 1, 1])
    
    with col_cost1:
        cost_per_server = st.number_input(
            "Coût horaire/serveur (€/h)", 
            min_value=0.1, max_value=100.0, value=0.5, step=0.1,
            key="optim_cost_server",
            help="Coût de fonctionnement par serveur par heure"
        )
    
    with col_cost2:
        fixed_cost_per_server = st.number_input(
            "Coût fixe/serveur (€)", 
            min_value=0.0, max_value=500.0, value=0.0, step=1.0,
            key="optim_fixed_cost",
            help="Coût fixe d'installation/maintenance par serveur (amorti sur 1h)"
        )
    
    with col_cost3:
        cost_per_mu = st.number_input(
            "Coût/unité μ/serveur (€/h)",
            min_value=0.0, max_value=10.0, value=0.1, step=0.05,
            key="optim_cost_mu",
            help="Coût supplémentaire par unité de performance (μ élevé = serveurs plus puissants/chers)"
        )
    
    with col_cost4:
        penalty_per_min = st.number_input(
            "Pénalité/min attente (€)", 
            min_value=0.0, max_value=10.0, value=0.05, step=0.01,
            key="optim_penalty",
            help="Coût de pénalité par minute d'attente par client"
        )
    
    # Pondération coût vs temps
    st.markdown("**Contraintes de qualité**")
    col_qos1, col_qos2 = st.columns([1, 1])
    with col_qos1:
        acceptable_wait_time = st.slider(
            "Temps d'attente acceptable Wq (min)",
            min_value=1.0, max_value=15.0, value=5.0, step=0.5,
            key="optim_acceptable_wait",
            help="Temps d'attente considéré comme acceptable. Au-dessus, forte pénalité."
        )
    with col_qos2:
        target_rho = st.slider(
            "Taux utilisation cible ρ",
            min_value=0.5, max_value=0.9, value=0.7, step=0.05,
            key="optim_target_rho",
            help="Taux d'utilisation optimal visé pour minimiser les coûts"
        )
    
    st.markdown("**Pondération du compromis**")
    col_weight1, col_weight2 = st.columns([2, 1])
    with col_weight1:
        alpha = st.slider(
            "Poids du coût vs temps d'attente",
            min_value=0.0, max_value=1.0, value=0.9, step=0.05,
            key="optim_alpha",
            help="0 = priorité temps d'attente, 1 = priorité coût. Valeur élevée = favorise économies"
        )
    with col_weight2:
        st.markdown(f"""  
        - Poids coût: **{alpha:.0%}**  
        - Poids temps: **{1-alpha:.0%}**
        """)
    
    st.markdown("**Paramètres de simulation**")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        max_servers = st.slider(
            "Nombre max de serveurs", 
            min_value=5, max_value=30, value=15,
            key="optim_max_servers"
        )
    
    with col2:
        mu_min = st.number_input(
            "Taux de service min μ (par serveur/min)",
            min_value=1.0, max_value=20.0, value=5.0, step=1.0,
            key="optim_mu_min"
        )
        mu_max = st.number_input(
            "Taux de service max μ (par serveur/min)",
            min_value=5.0, max_value=50.0, value=25.0, step=1.0,
            key="optim_mu_max"
        )
    
    with col3:
        resolution = st.slider(
            "Résolution (points par axe)", 
            min_value=10, max_value=50, value=20,
            key="optim_resolution"
        )
    
    generate_btn = st.button("Générer les Heatmaps", type="primary", key="optim_generate")
    
    # --- Génération des heatmaps ---
    if generate_btn:
        with st.spinner("Génération des heatmaps en cours..."):
            # Définir les plages
            server_range = np.arange(1, max_servers + 1)
            mu_range = np.linspace(mu_min, mu_max, resolution)
            
            # Créer les matrices pour chaque métrique
            Z_cost = np.zeros((len(mu_range), len(server_range)))
            Z_wait = np.zeros((len(mu_range), len(server_range)))
            Z_rho = np.zeros((len(mu_range), len(server_range)))  # Taux d'utilisation
            
            # Remplir les matrices
            for i, mu_rate in enumerate(mu_range):
                for j, n_servers in enumerate(server_range):
                    try:
                        # Vérifier la stabilité: lambda < c * mu
                        if lambda_rate >= n_servers * mu_rate:
                            # Système instable
                            Z_cost[i, j] = np.nan
                            Z_wait[i, j] = np.nan
                            Z_rho[i, j] = np.nan
                        else:
                            queue = GenericQueue(lambda_rate, mu_rate, "M/M/c", int(n_servers))
                            metrics = queue.compute_theoretical_metrics()
                            
                            # Taux d'utilisation
                            rho = lambda_rate / (n_servers * mu_rate)
                            Z_rho[i, j] = rho
                            
                            # Temps d'attente Wq
                            wq = metrics.Wq
                            Z_wait[i, j] = wq  # Pas de cap pour le calcul
                            
                            # Coût total = coût fixe + coût horaire + coût performance + pénalité attente
                            # Coût fixe par serveur (amorti sur 1h)
                            fixed_cost = n_servers * fixed_cost_per_server
                            # Coût horaire de fonctionnement
                            hourly_cost = n_servers * cost_per_server
                            # Coût de performance : serveurs plus rapides coûtent plus cher
                            performance_cost = n_servers * mu_rate * cost_per_mu
                            # Coût de pénalité d'attente (par client par minute, multiplié par le débit)
                            # lambda_rate est en sub/min, donc lambda_rate * 60 = clients/heure
                            # wq est en minutes
                            wait_cost = wq * penalty_per_min * lambda_rate * 60
                            
                            # Pénalité pour taux d'utilisation non optimal
                            # Si rho est trop bas : gaspillage de ressources
                            # Si rho est trop haut : risque de saturation
                            # Pénalité proportionnelle à l'écart par rapport au target_rho
                            rho_penalty_factor = abs(rho - target_rho) * 2.0  # Facteur de pénalité
                            utilization_penalty = (fixed_cost + hourly_cost + performance_cost) * rho_penalty_factor
                            
                            total_cost = fixed_cost + hourly_cost + performance_cost + wait_cost + utilization_penalty
                            Z_cost[i, j] = total_cost
                    except Exception:
                        Z_cost[i, j] = np.nan
                        Z_wait[i, j] = np.nan
                        Z_rho[i, j] = np.nan
            
            # Normaliser pour la moyenne (échelles différentes)
            # Normalisation min-max pour chaque matrice
            def normalize_matrix(Z):
                Z_flat = Z[~np.isnan(Z)]
                if len(Z_flat) == 0:
                    return Z
                z_min, z_max = np.nanmin(Z_flat), np.nanmax(Z_flat)
                if z_max - z_min == 0:
                    return np.zeros_like(Z)
                return (Z - z_min) / (z_max - z_min)
            
            # Normalisation spéciale pour le temps d'attente
            # On pénalise surtout les temps > acceptable_wait_time
            def normalize_wait_with_threshold(Z_wait, threshold):
                Z_flat = Z_wait[~np.isnan(Z_wait)]
                if len(Z_flat) == 0:
                    return Z_wait
                
                # Créer une pénalité progressive
                Z_normalized = np.zeros_like(Z_wait)
                for i in range(Z_wait.shape[0]):
                    for j in range(Z_wait.shape[1]):
                        wq = Z_wait[i, j]
                        if np.isnan(wq):
                            Z_normalized[i, j] = np.nan
                        elif wq <= threshold:
                            # Temps acceptable : score faible (bon)
                            Z_normalized[i, j] = wq / threshold * 0.3  # Max 0.3 si <= seuil
                        else:
                            # Temps trop élevé : pénalité forte
                            excess_ratio = (wq - threshold) / threshold
                            Z_normalized[i, j] = 0.3 + min(excess_ratio, 2.0) * 0.35  # 0.3 à 1.0
                
                return Z_normalized
            
            Z_cost_norm = normalize_matrix(Z_cost)
            Z_wait_norm = normalize_wait_with_threshold(Z_wait, acceptable_wait_time)
            # Score combiné pondéré: alpha * coût + (1-alpha) * temps
            Z_avg = alpha * Z_cost_norm + (1 - alpha) * Z_wait_norm
            
            # Appliquer un cap pour la visualisation
            Z_wait_display = np.minimum(Z_wait, 10)  # Cap à 10 min pour affichage
            Z_cost_display = np.minimum(Z_cost, np.nanpercentile(Z_cost[~np.isnan(Z_cost)], 95) if not np.all(np.isnan(Z_cost)) else 100)
            
            # --- Affichage des 3 heatmaps ---
            st.markdown("---")
            st.subheader("Résultats")
            st.markdown(f"**Taux d'arrivée utilisé**: λ = {lambda_rate:.2f} soumissions/min")
            st.markdown(f"**Pondération**: {alpha:.0%} coût / {1-alpha:.0%} temps d'attente")
            st.markdown(f"**Temps d'attente acceptable**: ≤ {acceptable_wait_time} min (au-delà = forte pénalité)")
            
            # Heatmap 1: Coût
            st.markdown("### 1. Heatmap du Coût Total (€/h)")
            st.caption(f"Coût = (fixe: {fixed_cost_per_server}€/srv) + (horaire: {cost_per_server}€/h/srv) + (performance: {cost_per_mu}€/h/μ/srv) + (pénalité: {penalty_per_min}€/min/client)")
            fig_cost = go.Figure(data=go.Heatmap(
                z=Z_cost_display,
                x=server_range,
                y=mu_range,
                colorscale='RdYlGn_r',
                colorbar=dict(title="Coût (€/h)"),
                hoverongaps=False
            ))
            fig_cost.update_layout(
                xaxis_title='Nombre de serveurs',
                yaxis_title='Taux de service μ (par serveur/min)',
                height=450
            )
            st.plotly_chart(fig_cost, use_container_width=True)
            
            # Heatmap 2: Temps d'attente
            st.markdown("### 2. Heatmap du Temps d'Attente Wq (min)")
            fig_wait = go.Figure(data=go.Heatmap(
                z=Z_wait_display,
                x=server_range,
                y=mu_range,
                colorscale='RdYlGn_r',
                colorbar=dict(title="Wq (min)"),
                hoverongaps=False
            ))
            fig_wait.update_layout(
                xaxis_title='Nombre de serveurs',
                yaxis_title='Taux de service μ (par serveur/min)',
                height=450
            )
            st.plotly_chart(fig_wait, use_container_width=True)
            
            # Heatmap 3: Score moyen normalisé
            st.markdown("### 3. Heatmap Score Combiné Pondéré")
            st.caption(f"Score = {alpha:.0%} × Coût(normalisé) + {1-alpha:.0%} × Temps(normalisé). Vert = optimal.")
            fig_avg = go.Figure(data=go.Heatmap(
                z=Z_avg,
                x=server_range,
                y=mu_range,
                colorscale='RdYlGn_r',
                colorbar=dict(title="Score (0-1)"),
                hoverongaps=False
            ))
            fig_avg.update_layout(
                xaxis_title='Nombre de serveurs',
                yaxis_title='Taux de service μ (par serveur/min)',
                height=450
            )
            st.plotly_chart(fig_avg, use_container_width=True)
            
            # Interprétation
            st.markdown("""
            **Lecture des heatmaps:**
            - **Vert** = bonnes performances / faible coût
            - **Rouge** = mauvaises performances / coût élevé
            - **Zones blanches** = système instable (ρ ≥ 1)
            - La frontière de stabilité est définie par: λ < c × μ
            - Cherchez le compromis optimal (zone verte) entre ressources et performances
            """)
            
            # --- Configuration optimale ---
            st.markdown("---")
            st.subheader("Configuration Optimale Recommandée")
            
            # Trouver le minimum du score combiné (meilleur compromis)
            if not np.all(np.isnan(Z_avg)):
                # Trouver l'indice du minimum
                min_idx = np.nanargmin(Z_avg)
                min_i, min_j = np.unravel_index(min_idx, Z_avg.shape)
                
                optimal_mu = mu_range[min_i]
                optimal_servers = server_range[min_j]
                optimal_cost = Z_cost[min_i, min_j]
                optimal_wait = Z_wait[min_i, min_j]
                optimal_score = Z_avg[min_i, min_j]
                
                # Calculer rho pour la config optimale
                optimal_rho = lambda_rate / (optimal_servers * optimal_mu)
                
                # Décomposition des coûts pour la config optimale
                opt_fixed_cost = optimal_servers * fixed_cost_per_server
                opt_hourly_cost = optimal_servers * cost_per_server
                opt_performance_cost = optimal_servers * optimal_mu * cost_per_mu
                opt_wait_cost = optimal_wait * penalty_per_min * lambda_rate * 60
                opt_rho_penalty = abs(optimal_rho - target_rho) * 2.0 * (opt_fixed_cost + opt_hourly_cost + opt_performance_cost)
                
                # Afficher les résultats
                personas_str = ", ".join(selected_personas_names) if input_mode != "Manuel" else "Mode manuel"
                
                # Avertissement si trop loin de la cible
                rho_diff = abs(optimal_rho - target_rho)
                if rho_diff > 0.15:
                    st.warning(f"**Meilleur compromis coût/temps d'attente pour: {personas_str}**\n\nAttention: Le taux d'utilisation ({optimal_rho:.1%}) s'écarte de la cible ({target_rho:.0%}). Ajustez les paramètres pour approcher la cible.")
                else:
                    st.success(f"**Meilleur compromis coût/temps d'attente pour: {personas_str}**")
                
                col_opt1, col_opt2, col_opt3 = st.columns(3)
                with col_opt1:
                    st.metric("Nombre de serveurs optimal", f"{optimal_servers}")
                    rho_delta = optimal_rho - target_rho
                    st.metric("Taux d'utilisation ρ", f"{optimal_rho:.1%}", 
                             delta=f"{rho_delta:+.1%} vs cible",
                             delta_color="inverse")
                with col_opt2:
                    st.metric("Taux de service μ optimal", f"{optimal_mu:.2f} /min")
                    st.metric(f"Temps d'attente Wq", f"{optimal_wait:.2f} min",
                             delta=f"{optimal_wait - acceptable_wait_time:+.2f} min vs acceptable",
                             delta_color="inverse")
                with col_opt3:
                    st.metric("Coût total", f"{optimal_cost:.2f} €/h")
                    st.metric("Score combiné", f"{optimal_score:.3f}")
                
                st.markdown(f"""
                **Récapitulatif:**
                - Taux d'arrivée λ = {lambda_rate:.2f} sub/min
                - Configuration: **{optimal_servers} serveurs** avec μ = **{optimal_mu:.2f}** services/min/serveur
                - Capacité totale: {optimal_servers * optimal_mu:.2f} services/min
                - Marge de capacité: {(optimal_servers * optimal_mu - lambda_rate):.2f} services/min
                """)
                
                # Détail des coûts
                st.markdown("**Décomposition du coût total:**")
                col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns(5)
                with col_c1:
                    st.metric("Coût fixe", f"{opt_fixed_cost:.2f} €", 
                              help=f"{optimal_servers} × {fixed_cost_per_server}€")
                with col_c2:
                    st.metric("Coût horaire", f"{opt_hourly_cost:.2f} €/h",
                              help=f"{optimal_servers} × {cost_per_server}€/h")
                with col_c3:
                    st.metric("Coût performance", f"{opt_performance_cost:.2f} €/h",
                              help=f"{optimal_servers} × {optimal_mu:.2f} × {cost_per_mu}€/h")
                with col_c4:
                    st.metric("Pénalité attente", f"{opt_wait_cost:.2f} €/h",
                              help=f"{optimal_wait:.3f}min × {penalty_per_min}€ × {lambda_rate*60:.0f} clients/h")
                with col_c5:
                    st.metric("Pénalité utilisation", f"{opt_rho_penalty:.2f} €/h",
                              help=f"Pénalité pour écart de ρ par rapport à la cible {target_rho:.0%}")
            else:
                st.error("Aucune configuration stable trouvée. Essayez d'augmenter le nombre de serveurs ou le taux de service.")

if __name__ == "__main__":
    main()
