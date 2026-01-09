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

from app.models import MM1Queue, MMcQueue, MMcKQueue, MD1Queue, MG1Queue
from app.personas import PersonaFactory, StudentType
from app.personas.usage_patterns import AcademicPeriod
from app.simulation import RushSimulator, MoulinetteSystem, SimulationConfig, ServerConfig
from app.optimization import CostOptimizer, ScalingAdvisor, CostModel, ScalingPolicy


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
        st.header("Parametres globaux")
        
        lambda_rate = st.slider(
            "Taux d'arrivée λ (soumissions/min)",
            min_value=1.0, max_value=100.0, value=30.0, step=1.0
        )
        
        mu_rate = st.slider(
            "Taux de service μ (par serveur/min)",
            min_value=1.0, max_value=50.0, value=10.0, step=0.5
        )
        
        n_servers = st.slider(
            "Nombre de serveurs",
            min_value=1, max_value=20, value=4
        )
        
        buffer_size = st.slider(
            "Taille du buffer K",
            min_value=10, max_value=500, value=100, step=10
        )
        
        st.divider()
        st.markdown("### Etat du systeme")
        rho = lambda_rate / (n_servers * mu_rate)
        
        if rho < 0.7:
            status_color = "[OK]"
            status_text = "Stable"
        elif rho < 0.9:
            status_color = "[!]"
            status_text = "Charge"
        else:
            status_color = "[X]"
            status_text = "Critique"
        
        st.metric("Utilisation ρ", f"{rho:.2%}", status_text)
    
    # Onglets principaux
    tabs = st.tabs([
        "Modeles de files",
        "Personas",
        "Simulation Rush",
        "Optimisation",
        "Auto-scaling",
        "Heatmaps"
    ])
    
    with tabs[0]:
        render_queue_models_tab(lambda_rate, mu_rate, n_servers, buffer_size)
    
    with tabs[1]:
        render_personas_tab()
    
    with tabs[2]:
        render_rush_simulation_tab(mu_rate, buffer_size)
    
    with tabs[3]:
        render_optimization_tab(lambda_rate, mu_rate, buffer_size)
    
    with tabs[4]:
        render_scaling_tab(mu_rate, buffer_size, n_servers, lambda_rate)
    
    with tabs[5]:
        render_heatmaps_tab(mu_rate, buffer_size)


def render_queue_models_tab(lambda_rate, mu_rate, n_servers, buffer_size):
    """Onglet de comparaison des modèles de files d'attente."""
    st.header("Comparaison des modeles de files d'attente")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Sélection des modèles")
        show_mm1 = st.checkbox("M/M/1", value=True)
        show_mmc = st.checkbox("M/M/c", value=True)
        show_mmck = st.checkbox("M/M/c/K", value=True)
        show_md1 = st.checkbox("M/D/1", value=True)
        show_mg1 = st.checkbox("M/G/1", value=True)
        
        cv_squared = st.slider(
            "CV² service (pour M/G/1)",
            min_value=0.0, max_value=2.0, value=1.0, step=0.1
        )
    
    with col1:
        # Calculer les métriques pour chaque modèle
        models_data = []
        
        if show_mm1 and n_servers == 1 and lambda_rate < mu_rate:
            queue = MM1Queue(lambda_rate, mu_rate)
            metrics = queue.compute_theoretical_metrics()
            models_data.append({
                'Modèle': 'M/M/1',
                'L (clients)': metrics.L,
                'Lq (en attente)': metrics.Lq,
                'W (min)': metrics.W,
                'Wq (min)': metrics.Wq,
                'ρ': metrics.rho,
                'P_blocage': 0.0
            })
        
        if show_mmc and lambda_rate < n_servers * mu_rate:
            queue = MMcQueue(lambda_rate, mu_rate, n_servers)
            metrics = queue.compute_theoretical_metrics()
            models_data.append({
                'Modèle': f'M/M/{n_servers}',
                'L (clients)': metrics.L,
                'Lq (en attente)': metrics.Lq,
                'W (min)': metrics.W,
                'Wq (min)': metrics.Wq,
                'ρ': metrics.rho,
                'P_blocage': 0.0
            })
        
        if show_mmck:
            queue = MMcKQueue(lambda_rate, mu_rate, n_servers, buffer_size)
            metrics = queue.compute_theoretical_metrics()
            models_data.append({
                'Modèle': f'M/M/{n_servers}/{buffer_size}',
                'L (clients)': metrics.L,
                'Lq (en attente)': metrics.Lq,
                'W (min)': metrics.W,
                'Wq (min)': metrics.Wq,
                'ρ': metrics.rho,
                'P_blocage': metrics.Pk
            })
        
        if show_md1 and lambda_rate < mu_rate:
            queue = MD1Queue(lambda_rate, mu_rate)
            metrics = queue.compute_theoretical_metrics()
            models_data.append({
                'Modèle': 'M/D/1',
                'L (clients)': metrics.L,
                'Lq (en attente)': metrics.Lq,
                'W (min)': metrics.W,
                'Wq (min)': metrics.Wq,
                'ρ': metrics.rho,
                'P_blocage': 0.0
            })
        
        if show_mg1 and lambda_rate < mu_rate:
            queue = MG1Queue(lambda_rate, mu_rate, cv_squared)
            metrics = queue.compute_theoretical_metrics()
            models_data.append({
                'Modèle': f'M/G/1 (CV²={cv_squared})',
                'L (clients)': metrics.L,
                'Lq (en attente)': metrics.Lq,
                'W (min)': metrics.W,
                'Wq (min)': metrics.Wq,
                'ρ': metrics.rho,
                'P_blocage': 0.0
            })
        
        if models_data:
            df = pd.DataFrame(models_data)
            st.dataframe(df.style.format({
                'L (clients)': '{:.2f}',
                'Lq (en attente)': '{:.2f}',
                'W (min)': '{:.3f}',
                'Wq (min)': '{:.3f}',
                'ρ': '{:.2%}',
                'P_blocage': '{:.4%}'
            }), use_container_width=True)
            
            # Graphique de comparaison
            fig = make_subplots(rows=1, cols=2, subplot_titles=['Temps d\'attente (Wq)', 'Longueur de queue (Lq)'])
            
            fig.add_trace(
                go.Bar(x=df['Modèle'], y=df['Wq (min)'], name='Wq'),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(x=df['Modèle'], y=df['Lq (en attente)'], name='Lq'),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucun modele selectionne ou parametres invalides (systeme instable)")
    
    # Section simulation
    st.subheader("Simulation Monte Carlo")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        sim_duration = st.number_input("Durée simulation (min)", 10, 1000, 60)
        selected_model = st.selectbox("Modèle à simuler", ['M/M/1', 'M/M/c', 'M/M/c/K', 'M/D/1', 'M/G/1'])
        run_sim = st.button("Lancer simulation")
    
    with col2:
        if run_sim:
            with st.spinner("Simulation en cours..."):
                queue = None
                error_msg = None
                
                if selected_model == 'M/M/1':
                    if n_servers != 1:
                        error_msg = "M/M/1 necessite exactement 1 serveur"
                    elif lambda_rate >= mu_rate:
                        error_msg = f"Systeme instable: lambda ({lambda_rate}) >= mu ({mu_rate})"
                    else:
                        queue = MM1Queue(lambda_rate, mu_rate)
                elif selected_model == 'M/M/c':
                    if lambda_rate >= n_servers * mu_rate:
                        error_msg = f"Systeme instable: lambda ({lambda_rate}) >= c*mu ({n_servers * mu_rate})"
                    else:
                        queue = MMcQueue(lambda_rate, mu_rate, n_servers)
                elif selected_model == 'M/M/c/K':
                    queue = MMcKQueue(lambda_rate, mu_rate, n_servers, buffer_size)
                elif selected_model == 'M/D/1':
                    if lambda_rate >= mu_rate:
                        error_msg = f"Systeme instable: lambda ({lambda_rate}) >= mu ({mu_rate})"
                    else:
                        queue = MD1Queue(lambda_rate, mu_rate)
                elif selected_model == 'M/G/1':
                    if lambda_rate >= mu_rate:
                        error_msg = f"Systeme instable: lambda ({lambda_rate}) >= mu ({mu_rate})"
                    else:
                        queue = MG1Queue(lambda_rate, mu_rate, cv_squared)
                
                if error_msg:
                    st.error(error_msg)
                elif queue:
                    result = queue.simulate(sim_duration)
                    
                    # Calculer les métriques depuis les données brutes
                    n_served = result.n_served
                    avg_system_time = float(np.mean(result.system_times)) if len(result.system_times) > 0 else 0.0
                    avg_waiting_time = float(np.mean(result.waiting_times)) if len(result.waiting_times) > 0 else 0.0
                    max_queue_len = int(np.max(result.queue_length_trace)) if len(result.queue_length_trace) > 0 else 0
                    
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("Clients traites", n_served)
                    col_b.metric("Temps moyen systeme", f"{avg_system_time:.2f} min")
                    col_c.metric("Temps moyen attente", f"{avg_waiting_time:.2f} min")
                    col_d.metric("Longueur max queue", max_queue_len)
                    
                    # Graphique de l'évolution
                    if len(result.time_trace) > 0 and len(result.queue_length_trace) > 0:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=result.time_trace, y=result.queue_length_trace, mode='lines', name='Queue'))
                        fig.update_layout(
                            title='Evolution de la longueur de queue',
                            xaxis_title='Temps (min)',
                            yaxis_title='Clients en attente'
                        )
                        st.plotly_chart(fig, use_container_width=True)


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


def render_rush_simulation_tab(mu_rate, buffer_size):
    """Onglet de simulation de rush."""
    st.header("Simulation de periode de rush")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration du rush")
        
        rush_type = st.selectbox(
            "Type de rush",
            ["Deadline projet", "Semaine d'examens", "Rush soirée standard"]
        )
        
        base_load = st.slider("Charge de base (λ)", 10, 100, 30)
        rush_multiplier = st.slider("Multiplicateur rush", 1.0, 5.0, 2.5, 0.1)
        rush_duration_h = st.slider("Durée du rush (heures)", 1, 24, 4)
        n_build_servers = st.slider("Serveurs build", 1, 10, 4)
        n_test_servers = st.slider("Serveurs test", 1, 10, 2)
        
        simulate_btn = st.button("Simuler le rush")
    
    with col2:
        if simulate_btn:
            with st.spinner("Simulation en cours..."):
                # Configuration serveur
                server_config = ServerConfig(
                    n_servers=n_build_servers,
                    service_rate=mu_rate,
                    buffer_size=buffer_size
                )
                
                # Configuration simulation
                config = SimulationConfig(
                    duration_hours=rush_duration_h,
                    server_config=server_config,
                    seed=42
                )
                
                # Creer le simulateur
                simulator = RushSimulator(config)
                
                # Simuler
                report = simulator.run()
                
                # Afficher les resultats
                st.subheader("Resultats du rush")
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Debit (clients/h)", f"{report.throughput:.1f}")
                col_b.metric("Temps moyen systeme", f"{report.avg_system_time:.1f} min")
                col_c.metric("Temps moyen attente", f"{report.avg_waiting_time:.2f} min")
                
                col_d, col_e, col_f = st.columns(3)
                col_d.metric("Utilisation rho", f"{report.utilization:.0%}")
                col_e.metric("Longueur max queue", f"{report.max_queue_length}")
                col_f.metric("Taux rejet", f"{report.rejection_rate:.2%}")
                
                # Graphique d'évolution
                st.subheader("Evolution de la charge")
                
                times = np.linspace(0, rush_duration_h * 60, 100)
                
                # Simuler l'évolution (simplifié)
                build_loads = []
                test_loads = []
                
                for t in times:
                    # Charge qui monte puis descend
                    progress = t / (rush_duration_h * 60)
                    rush_factor = rush_multiplier * np.sin(np.pi * progress)
                    
                    build_load = base_load * (1 + rush_factor)
                    build_loads.append(build_load / (n_build_servers * mu_rate))
                    test_loads.append(build_load * 0.9 / (n_test_servers * mu_rate * 0.5))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=times, y=build_loads, name='ρ Build', fill='tozeroy'))
                fig.add_trace(go.Scatter(x=times, y=test_loads, name='ρ Test', fill='tozeroy'))
                fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Saturation")
                fig.add_hline(y=0.7, line_dash="dash", line_color="orange", annotation_text="Optimal")
                
                fig.update_layout(
                    xaxis_title='Temps (min)',
                    yaxis_title='Utilisation ρ',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)


def render_optimization_tab(lambda_rate, mu_rate, buffer_size):
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
                        queue = MMcKQueue(lambda_rate, mu_rate, n, buffer_size)
                        metrics = queue.compute_theoretical_metrics()
                        wq = metrics.Wq
                        
                        server_cost = n * cost_per_server + fixed_cost
                        wait_cost = wq * penalty_per_min * lambda_rate * 60
                        
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


def render_scaling_tab(mu_rate, buffer_size, current_servers, current_lambda):
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
                current_lambda=current_lambda,
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
                            queue = MMcKQueue(lambda_rate, mu_rate, n_servers, buffer_size)
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
