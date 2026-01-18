"""
Application Streamlit pour la simulation de la moulinette EPITA.

Interface interactive permettant de:
- Visualiser le modèle Waterfall (files infinies/finies)
- Simuler les mécanismes de backup et leur impact
- Analyser les populations différenciées (Channels & Dams)
- Optimiser coût/qualité de service
- Obtenir des recommandations de scaling

Basé sur le rapport de projet ERO2 - Janvier 2026

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
from typing import Callable, Dict, List, Tuple, Optional
import math
import heapq

# Ajouter le chemin parent pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.base_queue import GenericQueue, ChainQueue, QueueMetrics, SimulationResults
from app.personas import PersonaFactory, StudentType
from app.personas.usage_patterns import AcademicPeriod
from app.simulation import RushSimulator, MoulinetteSystem, SimulationConfig, ServerConfig
from app.optimization import CostOptimizer, ScalingAdvisor, CostModel, ScalingPolicy
import json
from datetime import datetime
import os


def apply_dark_theme(fig):
    """Applique un thème sombre aux graphiques Plotly pour améliorer la lisibilité."""
    fig.update_layout(
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#2d2d2d',
        font=dict(color='#e0e0e0'),
        xaxis=dict(
            gridcolor='#404040',
            zerolinecolor='#404040'
        ),
        yaxis=dict(
            gridcolor='#404040',
            zerolinecolor='#404040'
        ),
        legend=dict(
            bgcolor='rgba(45, 45, 45, 0.8)',
            bordercolor='#404040',
            borderwidth=1
        )
    )
    return fig


def run_app():
    """Point d'entrée principal."""
    main()


def main():
    """Application principale."""
    st.set_page_config(
        page_title="Moulinette Simulator - EPITA",
        page_icon="🎯",
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
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .scenario-box {
        background: #2b3e50;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
        color: #ecf0f1;
    }
    .formula-box {
        background: #34495e;
        border: 1px solid #3498db;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        text-align: center;
        margin: 1rem 0;
        color: #ecf0f1;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="main-header">🎯 Moulinette Simulator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Optimisation des files d\'attente pour la moulinette EPITA - Projet ERO2</p>', unsafe_allow_html=True)
    
    # Sidebar pour les paramètres globaux
    with st.sidebar:
        st.header("⚙️ Paramètres Globaux")
        st.markdown("---")
        
        st.subheader("File 1: Exécution Tests")
        mu_rate1 = st.slider(
            "μ₁ - Taux de service (tests/min)",
            min_value=1.0, max_value=50.0, value=10.0, step=0.5,
            help="Vitesse de traitement des test-suites par serveur"
        )
        n_servers = st.slider(
            "c - Nombre de runners",
            min_value=1, max_value=20, value=4
        )
        K1 = st.slider(
            "K₁ - Capacité buffer File 1",
            min_value=10, max_value=1000, value=100, step=10
        )
        
        st.markdown("---")
        st.subheader("File 2: Renvoi Résultats")
        mu_rate2 = st.slider(
            "μ₂ - Taux de service (résultats/min)",
            min_value=1.0, max_value=100.0, value=20.0, step=1.0,
            help="Vitesse d'envoi des résultats au frontend"
        )
        K2 = st.slider(
            "K₂ - Capacité buffer File 2",
            min_value=10, max_value=500, value=50, step=10
        )
        
        st.markdown("---")
        st.subheader("Modèle File 2")
        file2_model = st.radio(
            "Type de service",
            ["M/M/1 (Exponentiel)", "M/D/1 (Déterministe)"],
            help="M/D/1 réduit le temps d'attente de 50% vs M/M/1"
        )
    
    # Onglets principaux
    tabs = st.tabs([
        "📊 Scénario 1: Waterfall",
        "💾 Scénario 2: Backup",
        "👥 Channels & Dams",
        "💰 Optimisation Coût/QoS",
        "📈 Auto-Scaling",
        "🔬 Benchmark Modèles"
    ])
    
    with tabs[0]:
        render_waterfall_scenario(mu_rate1, mu_rate2, n_servers, K1, K2, file2_model)
    
    with tabs[1]:
        render_backup_scenario(mu_rate1, mu_rate2, n_servers, K1, K2)
    
    with tabs[2]:
        render_channels_dams_tab(mu_rate1, n_servers, K1)
    
    with tabs[3]:
        render_optimization_tab(mu_rate1, n_servers, K1)
    
    with tabs[4]:
        render_autoscaling_tab(mu_rate1, mu_rate2, n_servers, K1, K2)
    
    with tabs[5]:
        render_benchmark_tab(mu_rate1, n_servers, K1)


# ==============================================================================
# SCÉNARIO 1: MODÈLE WATERFALL
# ==============================================================================

def render_waterfall_scenario(mu_rate1: float, mu_rate2: float, n_servers: int, K1: int, K2: int, file2_model: str):
    """Scénario 1: Modèle Waterfall avec files infinies puis finies."""
    st.header("📊 Scénario 1: Modèle Waterfall")
    
    st.markdown("""
    <div class="scenario-box">
    <strong>Architecture en cascade de la moulinette:</strong><br>
    <code>Étudiants → Buffer₁ (K₁) → Runners (c) → Buffer₂ (K₂) → Frontend (1 serveur)</code>
    </div>
    """, unsafe_allow_html=True)
    
    # Schéma du système
    col_diagram, col_params = st.columns([2, 1])
    
    with col_params:
        st.subheader("Paramètres d'entrée")
        lambda_rate = st.slider(
            "λ - Taux d'arrivée (tags/min)",
            min_value=1.0, max_value=100.0, value=30.0, step=1.0,
            key="waterfall_lambda"
        )
        
        capacity_mode = st.radio(
            "Mode de capacité",
            ["Files infinies (K=∞)", "Files finies (K limité)"],
            key="waterfall_capacity"
        )
        
        use_md1 = "M/D/1" in file2_model
    
    with col_diagram:
        # Créer un schéma visuel du système
        fig_diagram = create_waterfall_diagram(lambda_rate, mu_rate1, mu_rate2, n_servers, K1, K2, capacity_mode)
        st.plotly_chart(apply_dark_theme(fig_diagram), use_container_width=True)
    
    st.divider()
    
    # Section analyse théorique
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📐 File 1: Exécution des tests (M/M/c)")
        
        # Condition de stabilité
        rho1 = lambda_rate / (n_servers * mu_rate1)
        stable1 = rho1 < 1 or capacity_mode == "Files finies (K limité)"
        
        st.markdown(f"""
        **Condition de stabilité:** ρ₁ = λ/(c×μ₁) = {lambda_rate:.1f}/({n_servers}×{mu_rate1:.1f}) = **{rho1:.3f}**
        """)
        
        if rho1 < 1:
            st.success(f"✅ Système stable (ρ₁ = {rho1:.2%} < 100%)")
        else:
            if capacity_mode == "Files finies (K limité)":
                st.warning(f"⚠️ ρ₁ = {rho1:.2%} ≥ 100% mais stable grâce au buffer fini K₁={K1}")
            else:
                st.error(f"❌ Système instable (ρ₁ = {rho1:.2%} ≥ 100%)")
        
        if stable1 or capacity_mode == "Files finies (K limité)":
            try:
                if capacity_mode == "Files infinies (K=∞)":
                    queue1 = GenericQueue(lambda_rate, mu_rate1, f"M/M/{n_servers}", c=n_servers)
                    metrics1 = queue1.compute_theoretical_metrics()
                    P_K1 = 0.0
                    lambda_eff = lambda_rate
                else:
                    # M/M/c/K - calcul avec file finie
                    metrics1, P_K1 = compute_mmck_metrics(lambda_rate, mu_rate1, n_servers, K1)
                    lambda_eff = lambda_rate * (1 - P_K1)
                
                # Afficher les métriques
                display_queue_metrics(metrics1, "File 1", P_K1)
                
            except Exception as e:
                st.error(f"Erreur de calcul: {e}")
                metrics1 = None
                lambda_eff = 0
        else:
            metrics1 = None
            lambda_eff = 0
    
    with col2:
        st.subheader(f"📐 File 2: Renvoi des résultats ({'M/D/1' if use_md1 else 'M/M/1'})")
        
        if lambda_eff > 0:
            # Le taux d'entrée en file 2 = taux de sortie de file 1
            lambda2 = lambda_eff
            rho2 = lambda2 / mu_rate2
            
            st.markdown(f"""
            **Taux d'arrivée effectif:** λ₂ = λ_eff = {lambda2:.2f} tags/min
            
            **Condition de stabilité:** ρ₂ = λ₂/μ₂ = {lambda2:.2f}/{mu_rate2:.1f} = **{rho2:.3f}**
            """)
            
            if rho2 < 1:
                st.success(f"✅ Système stable (ρ₂ = {rho2:.2%} < 100%)")
            else:
                if capacity_mode == "Files finies (K limité)":
                    st.warning(f"⚠️ ρ₂ = {rho2:.2%} ≥ 100% mais stable grâce au buffer fini K₂={K2}")
                else:
                    st.error(f"❌ Système instable (ρ₂ = {rho2:.2%} ≥ 100%)")
            
            try:
                if capacity_mode == "Files infinies (K=∞)" and rho2 < 1:
                    notation = "M/D/1" if use_md1 else "M/M/1"
                    queue2 = GenericQueue(lambda2, mu_rate2, notation)
                    metrics2 = queue2.compute_theoretical_metrics()
                    P_K2 = 0.0
                else:
                    # M/M/1/K ou M/D/1/K
                    metrics2, P_K2 = compute_mm1k_metrics(lambda2, mu_rate2, K2, use_md1)
                
                display_queue_metrics(metrics2, "File 2", P_K2)
                
                # Avantage M/D/1 vs M/M/1
                if use_md1:
                    st.info("💡 **M/D/1:** Temps d'attente réduit de ~50% grâce au service déterministe")
                
            except Exception as e:
                st.error(f"Erreur de calcul: {e}")
                metrics2 = None
                P_K2 = 0
        else:
            st.warning("Calculez d'abord la File 1")
            metrics2 = None
            P_K2 = 0
    
    st.divider()
    
    # Métriques globales du système
    st.subheader("🎯 Performance Globale du Système Waterfall")
    
    if metrics1 is not None and 'metrics2' in dir() and metrics2 is not None:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        W_total = metrics1.W + metrics2.W
        P_rejet = P_K1 if 'P_K1' in dir() else 0.0
        P_blank = (1 - P_rejet) * P_K2 if 'P_K2' in dir() else 0.0
        
        with col_m1:
            st.metric("Temps de séjour total (W)", f"{W_total:.3f} min")
        with col_m2:
            st.metric("Taux de rejet (P_K₁)", f"{P_rejet:.2%}")
        with col_m3:
            st.metric("Pages blanches (P_blank)", f"{P_blank:.4%}")
        with col_m4:
            throughput = lambda_rate * (1 - P_rejet) * (1 - P_K2)
            st.metric("Débit effectif", f"{throughput:.2f} tags/min")
        
        # Formules
        st.markdown("""
        <div class="formula-box">
        <strong>Formules clés:</strong><br>
        W_total = W₁ + W₂<br>
        P_rejet = P(file 1 pleine) = π_K₁<br>
        P_blank = (1 - P_rejet) × P_K₂
        </div>
        """, unsafe_allow_html=True)
    
    # Simulation Monte Carlo
    st.divider()
    st.subheader("🎲 Simulation Monte Carlo")
    
    col_sim1, col_sim2 = st.columns([1, 2])
    
    with col_sim1:
        n_customers = st.number_input("Nombre de clients", 100, 10000, 2000, step=100, key="waterfall_n")
        n_runs = st.number_input("Nombre de répétitions", 1, 20, 5, key="waterfall_runs")
        run_sim = st.button("▶️ Lancer simulation Waterfall", type="primary", key="waterfall_run")
    
    with col_sim2:
        if run_sim:
            run_waterfall_simulation(lambda_rate, mu_rate1, mu_rate2, n_servers, K1, K2, 
                                    n_customers, n_runs, capacity_mode, use_md1)


def create_waterfall_diagram(lambda_rate, mu_rate1, mu_rate2, n_servers, K1, K2, capacity_mode):
    """Crée un schéma visuel du système Waterfall."""
    fig = go.Figure()
    
    # Paramètres de positionnement
    y_base = 0.5
    
    # Source (étudiants)
    fig.add_trace(go.Scatter(
        x=[0], y=[y_base],
        mode='markers+text',
        marker=dict(size=40, color='#2ecc71', symbol='circle'),
        text=['👨‍🎓'], textposition='middle center',
        name='Étudiants', showlegend=False
    ))
    fig.add_annotation(x=0, y=y_base-0.15, text=f"λ={lambda_rate}", showarrow=False)
    
    # Flèche vers Buffer 1
    fig.add_annotation(x=0.5, y=y_base, ax=0.15, ay=y_base,
                      xref="x", yref="y", axref="x", ayref="y",
                      showarrow=True, arrowhead=2, arrowsize=1.5)
    
    # Buffer 1
    k1_text = "∞" if "infinies" in capacity_mode else str(K1)
    fig.add_trace(go.Scatter(
        x=[1], y=[y_base],
        mode='markers+text',
        marker=dict(size=50, color='#3498db', symbol='square'),
        text=[f'📦'], textposition='middle center',
        name='Buffer 1', showlegend=False
    ))
    fig.add_annotation(x=1, y=y_base-0.15, text=f"K₁={k1_text}", showarrow=False)
    
    # Flèche vers Runners
    fig.add_annotation(x=1.5, y=y_base, ax=1.15, ay=y_base,
                      xref="x", yref="y", axref="x", ayref="y",
                      showarrow=True, arrowhead=2, arrowsize=1.5)
    
    # Runners (multi-serveurs)
    fig.add_trace(go.Scatter(
        x=[2], y=[y_base],
        mode='markers+text',
        marker=dict(size=60, color='#e74c3c', symbol='square'),
        text=['🖥️'], textposition='middle center',
        name='Runners', showlegend=False
    ))
    fig.add_annotation(x=2, y=y_base-0.15, text=f"c={n_servers}, μ₁={mu_rate1}", showarrow=False)
    
    # Flèche vers Buffer 2
    fig.add_annotation(x=2.5, y=y_base, ax=2.15, ay=y_base,
                      xref="x", yref="y", axref="x", ayref="y",
                      showarrow=True, arrowhead=2, arrowsize=1.5)
    
    # Buffer 2
    k2_text = "∞" if "infinies" in capacity_mode else str(K2)
    fig.add_trace(go.Scatter(
        x=[3], y=[y_base],
        mode='markers+text',
        marker=dict(size=50, color='#9b59b6', symbol='square'),
        text=['📦'], textposition='middle center',
        name='Buffer 2', showlegend=False
    ))
    fig.add_annotation(x=3, y=y_base-0.15, text=f"K₂={k2_text}", showarrow=False)
    
    # Flèche vers Frontend
    fig.add_annotation(x=3.5, y=y_base, ax=3.15, ay=y_base,
                      xref="x", yref="y", axref="x", ayref="y",
                      showarrow=True, arrowhead=2, arrowsize=1.5)
    
    # Frontend
    fig.add_trace(go.Scatter(
        x=[4], y=[y_base],
        mode='markers+text',
        marker=dict(size=50, color='#f39c12', symbol='circle'),
        text=['🖥️'], textposition='middle center',
        name='Frontend', showlegend=False
    ))
    fig.add_annotation(x=4, y=y_base-0.15, text=f"μ₂={mu_rate2}", showarrow=False)
    
    fig.update_layout(
        height=200,
        xaxis=dict(showgrid=False, showticklabels=False, range=[-0.5, 4.5]),
        yaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
        margin=dict(l=20, r=20, t=30, b=20),
        title="Architecture du système Waterfall"
    )
    
    return apply_dark_theme(fig)


def display_queue_metrics(metrics: QueueMetrics, name: str, P_K: float = 0.0):
    """Affiche les métriques d'une file."""
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"L (clients dans système)", f"{metrics.L:.2f}")
        st.metric(f"W (temps de séjour)", f"{metrics.W:.3f} min")
    with col2:
        st.metric(f"Lq (en attente)", f"{metrics.Lq:.2f}")
        st.metric(f"Wq (temps d'attente)", f"{metrics.Wq:.3f} min")
    
    if P_K > 0:
        st.metric(f"P(blocage)", f"{P_K:.4%}")


def compute_mmck_metrics(lambda_rate: float, mu_rate: float, c: int, K: int) -> Tuple[QueueMetrics, float]:
    """Calcule les métriques pour M/M/c/K."""
    rho = lambda_rate / (c * mu_rate)
    a = lambda_rate / mu_rate
    
    # Calcul de π₀
    sum1 = sum((a ** n) / math.factorial(n) for n in range(c))
    if rho != 1:
        sum2 = ((a ** c) / math.factorial(c)) * (1 - rho ** (K - c + 1)) / (1 - rho)
    else:
        sum2 = ((a ** c) / math.factorial(c)) * (K - c + 1)
    
    P0 = 1 / (sum1 + sum2)
    
    # Calcul de π_K (probabilité de blocage)
    if rho != 1:
        P_K = ((a ** K) / (math.factorial(c) * (c ** (K - c)))) * P0
    else:
        P_K = ((a ** K) / math.factorial(K)) * P0
    
    # Lambda effectif
    lambda_eff = lambda_rate * (1 - P_K)
    
    # Calcul de L
    L = 0
    for n in range(1, c + 1):
        L += n * ((a ** n) / math.factorial(n)) * P0
    
    if rho != 1:
        numerator = (a ** c * rho * P0) / math.factorial(c)
        L += numerator * (1 - rho ** (K - c + 1) - (1 - rho) * (K - c + 1) * rho ** (K - c)) / ((1 - rho) ** 2)
    
    # Métriques via Little
    Lq = max(0, L - (1 - P0) * c / c)  # Approximation
    W = L / lambda_eff if lambda_eff > 0 else 0
    Wq = Lq / lambda_eff if lambda_eff > 0 else 0
    
    metrics = QueueMetrics(
        rho=rho if rho < 1 else 1.0,
        L=L,
        Lq=Lq,
        W=W,
        Wq=Wq,
        Ws=1/mu_rate,
        P0=P0,
        Pk=P_K,
        lambda_eff=lambda_eff,
        throughput=lambda_eff
    )
    
    return metrics, P_K


def compute_mm1k_metrics(lambda_rate: float, mu_rate: float, K: int, deterministic: bool = False) -> Tuple[QueueMetrics, float]:
    """Calcule les métriques pour M/M/1/K ou M/D/1/K."""
    rho = lambda_rate / mu_rate
    
    if rho == 1:
        P0 = 1 / (K + 1)
        P_K = 1 / (K + 1)
        L = K / 2
    else:
        P0 = (1 - rho) / (1 - rho ** (K + 1))
        P_K = P0 * (rho ** K)
        L = rho * (1 - (K + 1) * rho ** K + K * rho ** (K + 1)) / ((1 - rho) * (1 - rho ** (K + 1)))
    
    lambda_eff = lambda_rate * (1 - P_K)
    
    W = L / lambda_eff if lambda_eff > 0 else 0
    Wq = W - 1/mu_rate
    Lq = lambda_eff * Wq if Wq > 0 else 0
    
    # Pour M/D/1: réduction de 50% du temps d'attente
    if deterministic:
        Wq = Wq / 2
        Lq = Lq / 2
        W = Wq + 1/mu_rate
    
    metrics = QueueMetrics(
        rho=min(rho, 1.0),
        L=L,
        Lq=Lq,
        W=W,
        Wq=Wq,
        Ws=1/mu_rate,
        P0=P0,
        Pk=P_K,
        lambda_eff=lambda_eff,
        throughput=lambda_eff
    )
    
    return metrics, P_K


def run_waterfall_simulation(lambda_rate, mu_rate1, mu_rate2, n_servers, K1, K2, 
                             n_customers, n_runs, capacity_mode, use_md1):
    """Exécute une simulation Monte Carlo du système Waterfall."""
    with st.spinner("Simulation en cours..."):
        results_all = []
        
        for run in range(n_runs):
            # File 1: M/M/c ou M/M/c/K
            if "infinies" in capacity_mode:
                queue1 = GenericQueue(lambda_rate, mu_rate1, f"M/M/{n_servers}", c=n_servers)
            else:
                queue1 = GenericQueue(lambda_rate, mu_rate1, f"M/M/{n_servers}", c=n_servers, K=K1)
            
            res1 = queue1.simulate(n_customers=n_customers)
            
            # File 2: utilise les temps de départ de File 1 comme arrivées
            if len(res1.departure_times) > 0:
                notation2 = "M/D/1" if use_md1 else "M/M/1"
                if "infinies" in capacity_mode:
                    queue2 = GenericQueue(lambda_rate, mu_rate2, notation2)
                else:
                    queue2 = GenericQueue(lambda_rate, mu_rate2, notation2, K=K2)
                
                res2 = queue2.simulate(external_arrival_times=res1.departure_times)
                
                results_all.append({
                    'run': run + 1,
                    'file1_served': res1.n_served,
                    'file1_rejected': res1.n_rejected,
                    'file1_wait': np.mean(res1.waiting_times) if len(res1.waiting_times) > 0 else 0,
                    'file1_system': np.mean(res1.system_times) if len(res1.system_times) > 0 else 0,
                    'file2_served': res2.n_served,
                    'file2_rejected': res2.n_rejected,
                    'file2_wait': np.mean(res2.waiting_times) if len(res2.waiting_times) > 0 else 0,
                    'file2_system': np.mean(res2.system_times) if len(res2.system_times) > 0 else 0,
                    'total_wait': (np.mean(res1.waiting_times) if len(res1.waiting_times) > 0 else 0) + 
                                  (np.mean(res2.waiting_times) if len(res2.waiting_times) > 0 else 0),
                    'total_system': (np.mean(res1.system_times) if len(res1.system_times) > 0 else 0) + 
                                    (np.mean(res2.system_times) if len(res2.system_times) > 0 else 0),
                })
        
        if results_all:
            df = pd.DataFrame(results_all)
            
            # Statistiques
            st.markdown("### Résultats de simulation")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Temps d'attente moyen total", 
                         f"{df['total_wait'].mean():.3f} ± {df['total_wait'].std():.3f} min")
            with col2:
                st.metric("Temps de séjour moyen total",
                         f"{df['total_system'].mean():.3f} ± {df['total_system'].std():.3f} min")
            with col3:
                reject_rate = df['file1_rejected'].sum() / (df['file1_served'].sum() + df['file1_rejected'].sum())
                st.metric("Taux de rejet File 1", f"{reject_rate:.2%}")
            
            # Graphique
            fig = make_subplots(rows=1, cols=2, subplot_titles=['Temps par file', 'Distribution des temps totaux'])
            
            fig.add_trace(go.Box(y=df['file1_wait'], name='Attente F1'), row=1, col=1)
            fig.add_trace(go.Box(y=df['file2_wait'], name='Attente F2'), row=1, col=1)
            fig.add_trace(go.Box(y=df['file1_system'], name='Séjour F1'), row=1, col=1)
            fig.add_trace(go.Box(y=df['file2_system'], name='Séjour F2'), row=1, col=1)
            
            fig.add_trace(go.Histogram(x=df['total_system'], name='Temps total', nbinsx=20), row=1, col=2)
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(apply_dark_theme(fig), use_container_width=True)


# ==============================================================================
# SCÉNARIO 2: MÉCANISMES DE BACKUP
# ==============================================================================

def simulate_waterfall_with_backup(
    lambda_rate: float,
    mu_rate1: float,
    mu_rate2: float,
    mu_backup: float,
    n_servers: int,
    K1: int,
    K2: int,
    n_customers: int,
    p_backup: float = 1.0,
    seed: Optional[int] = None
) -> dict:
    """
    Simule le système Waterfall avec mécanisme de backup entre les deux files.
    
    Architecture:
    File 1 (M/M/c/K1) → [BACKUP] → File 2 (M/M/1/K2) → Frontend
    
    Le backup sauvegarde les résultats AVANT l'envoi vers la file 2.
    Si la file 2 rejette (pleine), les données sont récupérées depuis le backup.
    
    Args:
        lambda_rate: Taux d'arrivée des tags
        mu_rate1: Taux de service de la file 1 (moulinette)
        mu_rate2: Taux de service de la file 2 (envoi résultats)
        mu_backup: Taux de sauvegarde (backup/min)
        n_servers: Nombre de serveurs dans la file 1
        K1: Capacité de la file 1
        K2: Capacité de la file 2
        n_customers: Nombre de clients à simuler
        p_backup: Probabilité de faire un backup (0 à 1)
        seed: Graine aléatoire
    
    Returns:
        dict avec les métriques de simulation
    """
    rng = np.random.default_rng(seed)
    
    # Génération des arrivées
    interarrival_times = rng.exponential(1/lambda_rate, n_customers)
    arrival_times = np.cumsum(interarrival_times)
    
    # Structures pour File 1
    queue1 = []
    busy_servers1 = 0
    event_heap = []
    
    # Structures pour File 2
    queue2 = []
    busy_server2 = False
    
    # Backup storage
    backup_storage = {}  # client_id -> (result_data, backup_time)
    
    # Compteurs
    n_rejected_f1 = 0
    n_rejected_f2 = 0
    n_recovered_from_backup = 0
    n_backed_up = 0
    
    # Ensemble pour tracker les clients déjà rejetés de File 2 (une seule fois par client)
    rejected_f2_clients = set()
    
    # Métriques temporelles
    file1_wait_times = []
    file1_system_times = []
    file2_wait_times = []
    file2_system_times = []
    backup_times = []
    total_system_times = []
    
    # Service times (pré-générés)
    service_times_f1 = rng.exponential(1/mu_rate1, n_customers)
    service_times_f2 = rng.exponential(1/mu_rate2, n_customers)
    backup_service_times = rng.exponential(1/mu_backup, n_customers)
    
    # Décisions de backup (pré-générées)
    do_backup = rng.random(n_customers) < p_backup
    
    # Initialiser les arrivées
    for i, t in enumerate(arrival_times):
        heapq.heappush(event_heap, (t, "arrival_f1", i))
    
    # Compteur système
    system_size_f1 = 0
    system_size_f2 = 0
    
    # Traces temporelles
    time_trace = []
    f1_size_trace = []
    f2_size_trace = []
    backup_size_trace = []
    
    # Temps d'entrée dans le système pour chaque client
    client_arrival_f1 = {}
    client_exit_f1 = {}
    client_arrival_f2 = {}
    client_completed = set()
    
    while event_heap:
        time, event_type, client_id = heapq.heappop(event_heap)
        
        if event_type == "arrival_f1":
            # Arrivée dans File 1
            if system_size_f1 >= K1:
                n_rejected_f1 += 1
                continue
            
            system_size_f1 += 1
            client_arrival_f1[client_id] = time
            
            if busy_servers1 < n_servers:
                # Service immédiat
                busy_servers1 += 1
                depart_time = time + service_times_f1[client_id]
                heapq.heappush(event_heap, (depart_time, "depart_f1", client_id))
                file1_wait_times.append(0.0)
            else:
                queue1.append(client_id)
        
        elif event_type == "depart_f1":
            # Départ de File 1 → Backup → File 2
            system_size_f1 -= 1
            busy_servers1 -= 1
            
            client_exit_f1[client_id] = time
            file1_system_times.append(time - client_arrival_f1[client_id])
            
            # Faire le backup AVANT d'envoyer à la file 2
            if do_backup[client_id]:
                backup_time = backup_service_times[client_id]
                backup_storage[client_id] = {
                    'backup_time': time,
                    'data_ready': time + backup_time
                }
                backup_times.append(backup_time)
                n_backed_up += 1
                # Le backup est asynchrone, on continue vers File 2 immédiatement
                # mais les données sont sauvegardées
            
            # Essayer d'entrer dans File 2
            heapq.heappush(event_heap, (time, "try_enter_f2", client_id))
            
            # Servir le prochain dans File 1
            if queue1:
                next_client = queue1.pop(0)
                busy_servers1 += 1
                wait_time = time - client_arrival_f1[next_client]
                file1_wait_times.append(wait_time)
                depart_time = time + service_times_f1[next_client]
                heapq.heappush(event_heap, (depart_time, "depart_f1", next_client))
        
        elif event_type == "try_enter_f2":
            # Tentative d'entrée dans File 2
            if system_size_f2 >= K2:
                # File 2 pleine !
                # Compter le rejet seulement si c'est la première fois pour ce client
                if client_id not in rejected_f2_clients:
                    print(rejected_f2_clients)
                    n_rejected_f2 += 1
                    rejected_f2_clients.add(client_id)
                
                # Récupération depuis le backup si disponible
                if client_id in backup_storage:
                    n_recovered_from_backup += 1
                    # On planifie une ré-tentative après un délai
                    retry_time = time + rng.exponential(1/mu_rate2)  # Attendre avant retry
                    heapq.heappush(event_heap, (retry_time, "retry_from_backup", client_id))
                # Sinon: page blanche (données perdues)
                continue
            
            system_size_f2 += 1
            client_arrival_f2[client_id] = time
            
            if not busy_server2:
                # Service immédiat
                busy_server2 = True
                depart_time = time + service_times_f2[client_id]
                heapq.heappush(event_heap, (depart_time, "depart_f2", client_id))
                file2_wait_times.append(0.0)
            else:
                queue2.append(client_id)
        
        elif event_type == "retry_from_backup":
            # Ré-tentative depuis le backup
            if system_size_f2 >= K2:
                # Toujours pleine, on re-planifie
                retry_time = time + rng.exponential(1/mu_rate2)
                heapq.heappush(event_heap, (retry_time, "retry_from_backup", client_id))
                continue
            
            system_size_f2 += 1
            client_arrival_f2[client_id] = time
            
            if not busy_server2:
                busy_server2 = True
                depart_time = time + service_times_f2[client_id]
                heapq.heappush(event_heap, (depart_time, "depart_f2", client_id))
                file2_wait_times.append(0.0)
            else:
                queue2.append(client_id)
        
        elif event_type == "depart_f2":
            # Départ de File 2 → Client complété
            system_size_f2 -= 1
            busy_server2 = False
            
            if client_id in client_arrival_f2:
                file2_system_times.append(time - client_arrival_f2[client_id])
            
            if client_id in client_arrival_f1:
                total_system_times.append(time - client_arrival_f1[client_id])
            
            client_completed.add(client_id)
            
            # Nettoyer le backup
            if client_id in backup_storage:
                del backup_storage[client_id]
            
            # Servir le prochain dans File 2
            if queue2:
                next_client = queue2.pop(0)
                busy_server2 = True
                wait_time = time - client_arrival_f2[next_client]
                file2_wait_times.append(wait_time)
                depart_time = time + service_times_f2[next_client]
                heapq.heappush(event_heap, (depart_time, "depart_f2", next_client))
        
        # Mise à jour des traces
        time_trace.append(time)
        f1_size_trace.append(system_size_f1)
        f2_size_trace.append(system_size_f2)
        backup_size_trace.append(len(backup_storage))
    
    # Calcul des métriques finales
    n_completed = len(client_completed)
    n_lost = n_rejected_f2 - n_recovered_from_backup  # Vraies pages blanches
    
    return {
        'n_arrivals': n_customers,
        'n_rejected_f1': n_rejected_f1,
        'n_rejected_f2': n_rejected_f2,
        'n_backed_up': n_backed_up,
        'n_recovered_from_backup': n_recovered_from_backup,
        'n_completed': n_completed,
        'n_lost': max(0, n_lost),
        'p_blank': max(0, n_lost) / n_customers if n_customers > 0 else 0,
        'file1_wait_mean': np.mean(file1_wait_times) if file1_wait_times else 0,
        'file1_system_mean': np.mean(file1_system_times) if file1_system_times else 0,
        'file2_wait_mean': np.mean(file2_wait_times) if file2_wait_times else 0,
        'file2_system_mean': np.mean(file2_system_times) if file2_system_times else 0,
        'backup_time_mean': np.mean(backup_times) if backup_times else 0,
        'total_system_mean': np.mean(total_system_times) if total_system_times else 0,
        'time_trace': np.array(time_trace),
        'f1_size_trace': np.array(f1_size_trace),
        'f2_size_trace': np.array(f2_size_trace),
        'backup_size_trace': np.array(backup_size_trace),
    }


def render_backup_scenario(mu_rate1: float, mu_rate2: float, n_servers: int, K1: int, K2: int):
    """Scénario 2: Impact du backup sur les pages blanches."""
    st.header("💾 Scénario 2: Mécanismes de Backup")
    
    st.markdown("""
    <div class="scenario-box">
    <strong>Problématique:</strong> Quand la file 2 est saturée, les résultats sont perdus → <em>pages blanches</em><br><br>
    <strong>Solution:</strong> Sauvegarder les résultats de moulinettage <em>avant</em> l'envoi vers la file 2.<br>
    Si la file 2 rejette, les données peuvent être récupérées depuis le backup.
    </div>
    """, unsafe_allow_html=True)
    
    # Schéma du système avec backup
    st.subheader("🏗️ Architecture avec Backup")
    
    fig_arch = go.Figure()
    y_base = 0.5
    
    # File 1
    fig_arch.add_trace(go.Scatter(x=[0], y=[y_base], mode='markers+text',
        marker=dict(size=40, color='#2ecc71'), text=['👨‍🎓'], textposition='middle center', showlegend=False))
    fig_arch.add_annotation(x=0, y=y_base-0.2, text="Étudiants", showarrow=False)
    
    fig_arch.add_annotation(x=0.4, y=y_base, ax=0.1, ay=y_base, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2)
    
    fig_arch.add_trace(go.Scatter(x=[0.8], y=[y_base], mode='markers+text',
        marker=dict(size=50, color='#3498db', symbol='square'), text=['🖥️'], textposition='middle center', showlegend=False))
    fig_arch.add_annotation(x=0.8, y=y_base-0.2, text=f"File 1\n(M/M/{n_servers}/K₁)", showarrow=False)
    
    # Backup (point de sauvegarde)
    fig_arch.add_annotation(x=1.3, y=y_base, ax=1.0, ay=y_base, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2)
    
    fig_arch.add_trace(go.Scatter(x=[1.6], y=[y_base], mode='markers+text',
        marker=dict(size=50, color='#f39c12', symbol='diamond'), text=['💾'], textposition='middle center', showlegend=False))
    fig_arch.add_annotation(x=1.6, y=y_base-0.2, text="BACKUP\n(sauvegarde)", showarrow=False, font=dict(color='#f39c12'))
    
    # Flèche vers File 2
    fig_arch.add_annotation(x=2.1, y=y_base, ax=1.8, ay=y_base, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2)
    
    # File 2
    fig_arch.add_trace(go.Scatter(x=[2.4], y=[y_base], mode='markers+text',
        marker=dict(size=50, color='#9b59b6', symbol='square'), text=['📤'], textposition='middle center', showlegend=False))
    fig_arch.add_annotation(x=2.4, y=y_base-0.2, text=f"File 2\n(M/M/1/K₂)", showarrow=False)
    
    # Flèche de récupération (backup -> retry)
    fig_arch.add_trace(go.Scatter(x=[1.6, 2.0, 2.4], y=[y_base+0.25, y_base+0.35, y_base+0.25], 
        mode='lines', line=dict(color='#e74c3c', dash='dash', width=2), showlegend=False))
    fig_arch.add_annotation(x=2.0, y=y_base+0.4, text="Récupération si rejet", showarrow=False, font=dict(color='#e74c3c', size=10))
    
    # Frontend
    fig_arch.add_annotation(x=2.9, y=y_base, ax=2.6, ay=y_base, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2)
    
    fig_arch.add_trace(go.Scatter(x=[3.2], y=[y_base], mode='markers+text',
        marker=dict(size=40, color='#27ae60'), text=['✅'], textposition='middle center', showlegend=False))
    fig_arch.add_annotation(x=3.2, y=y_base-0.2, text="Résultat\naffiché", showarrow=False)
    
    fig_arch.update_layout(
        height=250,
        xaxis=dict(showgrid=False, showticklabels=False, range=[-0.3, 3.5]),
        yaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(apply_dark_theme(fig_arch), use_container_width=True)
    
    st.subheader("⚙️ Configuration de la simulation")
    
    col_config1, col_config2, col_config3 = st.columns(3)
    
    with col_config1:
        lambda_rate = st.slider("λ - Taux d'arrivée (tags/min)", 10.0, 100.0, 40.0, key="backup_lambda")
    
    with col_config2:
        n_customers = st.number_input("Nombre de clients", 100, 10000, 2000, step=100, key="backup_sim_n")
    
    with col_config3:
        n_runs = st.number_input("Nombre de répétitions", 1, 20, 5, key="backup_sim_runs")
    
    mu_backup = 50.0  # Valeur fixe pour le taux de sauvegarde
    
    st.divider()
    
    # Section Simulation Monte Carlo avec Backup
    st.subheader("🎲 Simulation Monte Carlo - Analyse de sensibilité du backup")
    
    st.markdown("""
    La simulation compare automatiquement 5 valeurs de probabilité de backup:
    - **P=0%**: Aucun backup (référence)
    - **P=25%**: Backup aléatoire faible
    - **P=50%**: Backup aléatoire moyen
    - **P=75%**: Backup aléatoire élevé
    - **P=100%**: Backup systématique (optimal)
    """)
    
    run_backup_sim = st.button("▶️ Lancer simulation complète", type="primary", key="backup_sim_run", use_container_width=True)
    
    if run_backup_sim:
        with st.spinner("Simulation en cours pour P = 0%, 25%, 50%, 75%, 100%..."):
            # Valeurs de P à tester
            p_values = [0.0, 0.25, 0.5, 0.75, 1.0]
            all_results = {p: [] for p in p_values}
            
            # Exécuter les simulations pour chaque valeur de P
            for p_val in p_values:
                for run in range(n_runs):
                    res = simulate_waterfall_with_backup(
                        lambda_rate, mu_rate1, mu_rate2, mu_backup,
                        n_servers, K1, K2, n_customers, p_backup=p_val, seed=run
                    )
                    all_results[p_val].append(res)
                
            # Affichage des résultats - Tableau comparatif
            st.markdown("### 📊 Résultats comparatifs")
            
            # Créer le tableau de résultats
            summary_data = []
            for p_val in p_values:
                results = all_results[p_val]
                summary_data.append({
                    'Probabilité P': f"{p_val:.0%}",
                    'Pages blanches (%)': f"{np.mean([r['p_blank'] for r in results]) * 100:.3f}",
                    'Clients perdus': f"{np.mean([r['n_lost'] for r in results]):.1f}",
                    'Récupérés backup': f"{np.mean([r['n_recovered_from_backup'] for r in results]):.1f}",
                    'Sauvegardés': f"{np.mean([r['n_backed_up'] for r in results]):.1f}",
                    'Rejets File 1': f"{np.mean([r['n_rejected_f1'] for r in results]):.1f}",
                    'Rejets File 2': f"{np.mean([r['n_rejected_f2'] for r in results]):.1f}",
                    'Temps moyen (min)': f"{np.mean([r['total_system_mean'] for r in results]):.3f}",
                    'Complétés': f"{np.mean([r['n_completed'] for r in results]):.1f}"
                })
            
            df_summary = pd.DataFrame(summary_data)
            
            # Afficher le tableau avec style
            st.dataframe(
                df_summary,
                use_container_width=True,
                hide_index=True
            )
            
            st.divider()
            
            # Graphiques de visualisation
            st.markdown("### 📈 Visualisations comparatives")
            
            # Graphique 1: Pages blanches et clients perdus
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                fig_blank = go.Figure()
                blank_rates = [np.mean([r['p_blank'] for r in all_results[p]]) * 100 for p in p_values]
                
                fig_blank.add_trace(go.Scatter(
                    x=[f"{p:.0%}" for p in p_values],
                    y=blank_rates,
                    mode='lines+markers',
                    name='Pages blanches',
                    line=dict(color='#e74c3c', width=3),
                    marker=dict(size=12),
                    fill='tozeroy'
                ))
                
                fig_blank.update_layout(
                    title="Impact de P sur les pages blanches",
                    xaxis_title="Probabilité de backup (P)",
                    yaxis_title="Pages blanches (%)",
                    height=350
                )
                st.plotly_chart(apply_dark_theme(fig_blank), use_container_width=True)
            
            with col_g2:
                fig_recovery = go.Figure()
                
                saved = [np.mean([r['n_backed_up'] for r in all_results[p]]) for p in p_values]
                recovered = [np.mean([r['n_recovered_from_backup'] for r in all_results[p]]) for p in p_values]
                lost = [np.mean([r['n_lost'] for r in all_results[p]]) for p in p_values]
                
                fig_recovery.add_trace(go.Scatter(
                    x=[f"{p:.0%}" for p in p_values],
                    y=saved,
                    mode='lines+markers',
                    name='Sauvegardés',
                    line=dict(color='#f39c12', width=2),
                    marker=dict(size=10)
                ))
                
                fig_recovery.add_trace(go.Scatter(
                    x=[f"{p:.0%}" for p in p_values],
                    y=recovered,
                    mode='lines+markers',
                    name='Récupérés du backup',
                    line=dict(color='#27ae60', width=2),
                    marker=dict(size=10)
                ))
                
                fig_recovery.add_trace(go.Scatter(
                    x=[f"{p:.0%}" for p in p_values],
                    y=lost,
                    mode='lines+markers',
                    name='Perdus définitivement',
                    line=dict(color='#c0392b', width=2),
                    marker=dict(size=10)
                ))
                
                fig_recovery.update_layout(
                    title="Efficacité du mécanisme de backup",
                    xaxis_title="Probabilité de backup (P)",
                    yaxis_title="Nombre de clients",
                    height=350
                )
                st.plotly_chart(apply_dark_theme(fig_recovery), use_container_width=True)
            
            st.divider()
            
            # Graphique 3: Box plots comparatifs (pleine largeur)
            st.markdown("### 📦 Distribution des résultats")
            
            fig_boxes = make_subplots(
                rows=1, cols=2,
                subplot_titles=['Distribution des pages blanches (%)', 'Distribution des temps de séjour (min)']
            )
            
            colors = ['#c0392b', '#e67e22', '#f39c12', '#27ae60', '#2ecc71']
            
            for i, p_val in enumerate(p_values):
                results = all_results[p_val]
                blanks = [r['p_blank'] * 100 for r in results]
                times = [r['total_system_mean'] for r in results]
                
                fig_boxes.add_trace(go.Box(
                    y=blanks,
                    name=f"P={p_val:.0%}",
                    marker_color=colors[i],
                    showlegend=True
                ), row=1, col=1)
                
                fig_boxes.add_trace(go.Box(
                    y=times,
                    name=f"P={p_val:.0%}",
                    marker_color=colors[i],
                    showlegend=False
                ), row=1, col=2)
            
            fig_boxes.update_yaxes(title_text="Pages blanches (%)", row=1, col=1)
            fig_boxes.update_yaxes(title_text="Temps (min)", row=1, col=2)
            fig_boxes.update_layout(height=450, showlegend=True)
            
            st.plotly_chart(apply_dark_theme(fig_boxes), use_container_width=True)
            
            st.divider()
            
            # Graphique 4: Évolution temporelle (dernière simulation avec P=100%)
            last_res = all_results[1.0][-1]  # Dernière simulation avec P=100%
            if len(last_res['time_trace']) > 0:
                st.markdown("### 📈 Évolution temporelle (simulation avec P=100%)")
                
                # Sous-échantillonnage si trop de points
                max_points = 1000
                step = max(1, len(last_res['time_trace']) // max_points)
                
                fig_trace = make_subplots(
                    rows=2, cols=1, 
                    shared_xaxes=True,
                    subplot_titles=['Taille des files d\'attente', 'Données en backup'],
                    vertical_spacing=0.12
                )
                
                fig_trace.add_trace(go.Scatter(
                    x=last_res['time_trace'][::step], 
                    y=last_res['f1_size_trace'][::step],
                    name='File 1 (Moulinette)', 
                    line=dict(color='#3498db', width=2)
                ), row=1, col=1)
                
                fig_trace.add_trace(go.Scatter(
                    x=last_res['time_trace'][::step], 
                    y=last_res['f2_size_trace'][::step],
                    name='File 2 (Envoi résultats)', 
                    line=dict(color='#9b59b6', width=2)
                ), row=1, col=1)
                
                fig_trace.add_trace(go.Scatter(
                    x=last_res['time_trace'][::step], 
                    y=last_res['backup_size_trace'][::step],
                    name='Stockage backup', 
                    line=dict(color='#f39c12', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(243, 156, 18, 0.3)'
                ), row=2, col=1)
                
                fig_trace.update_xaxes(title_text="Temps (min)", row=2, col=1)
                fig_trace.update_yaxes(title_text="Nb clients en file", row=1, col=1)
                fig_trace.update_yaxes(title_text="Nb en backup", row=2, col=1)
                fig_trace.update_layout(height=500, showlegend=True)
                
                st.plotly_chart(apply_dark_theme(fig_trace), use_container_width=True)
    
    st.divider()
    
    st.info("""
    💡 **Mécanisme de backup implémenté:**
    1. Les résultats de moulinettage sont sauvegardés **avant** l'envoi vers la file 2
    2. Si la file 2 rejette (pleine), les données sont **récupérées depuis le backup**
    3. Une nouvelle tentative d'envoi est planifiée jusqu'au succès
    4. Cela élimine (presque) totalement les pages blanches
    """)


# ==============================================================================
# CHANNELS & DAMS: POPULATIONS DIFFÉRENCIÉES
# ==============================================================================

def render_channels_dams_tab(mu_rate1: float, n_servers: int, K1: int):
    """Onglet Channels & Dams pour populations différenciées."""
    st.header("👥 Channels & Dams: Populations Différenciées")
    
    st.markdown("""
    <div class="scenario-box">
    <strong>Constat:</strong> Les différentes populations ont des besoins distincts<br>
    <ul>
        <li><strong>Prépa SUP/SPÉ</strong>: Soumissions groupées à la deadline</li>
        <li><strong>ING1</strong>: Soumissions fréquentes, traitement immédiat</li>
        <li><strong>Admin/Staff</strong>: Priorité pour tests continus</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Créer les personas
    personas = PersonaFactory.create_all_personas()
    
    # Afficher les caractéristiques
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Caractéristiques des populations")
        
        data = []
        for student_type, persona in personas.items():
            data.append({
                'Population': persona.name,
                'Effectif': persona.population_size,
                'Taux base (sub/h)': persona.base_submission_rate,
                'Variance': f"{persona.variance_coefficient:.2f}",
                'Procrastination': f"{persona.procrastination_level:.0%}"
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
    
    with col2:
        st.subheader("📈 Patterns d'arrivée sur 24h")
        
        hours = list(range(24))
        fig = go.Figure()
        
        for student_type, persona in personas.items():
            rates = [persona.get_arrival_rate(h) for h in hours]
            fig.add_trace(go.Scatter(
                x=hours, y=rates,
                mode='lines+markers',
                name=persona.name,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            xaxis_title='Heure',
            yaxis_title='Taux d\'arrivée (sub/h)',
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(apply_dark_theme(fig), use_container_width=True)
    
    st.divider()
    
    # Stratégies de régulation
    st.subheader("🎛️ Stratégies de Régulation")
    
    strategy = st.radio(
        "Stratégie",
        ["File unique (actuel)", "Dam - Blocage périodique", "Channels - Files séparées", "Hybride - Pool avec priorités"],
        horizontal=True
    )
    
    if strategy == "Dam - Blocage périodique":
        st.markdown("""
        **Principe:** Bloquer périodiquement la moulinette pour grouper les traitements
        
        - Fermée pendant t_b
        - Ouverte pendant t_b/2
        """)
        
        t_block = st.slider("Durée de blocage t_b (min)", 5, 60, 15)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Période fermée", f"{t_block} min")
        with col2:
            st.metric("Période ouverte", f"{t_block/2} min")
        
        # Visualisation du cycle
        fig = go.Figure()
        x = []
        y = []
        for i in range(5):
            # Fermé
            x.extend([i * 1.5 * t_block, i * 1.5 * t_block + t_block])
            y.extend([0, 0])
            # Ouvert
            x.extend([i * 1.5 * t_block + t_block, i * 1.5 * t_block + 1.5 * t_block])
            y.extend([1, 1])
        
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', fill='tozeroy', name='État'))
        fig.update_layout(
            xaxis_title="Temps (min)",
            yaxis=dict(tickvals=[0, 1], ticktext=['Fermé', 'Ouvert']),
            height=200
        )
        st.plotly_chart(apply_dark_theme(fig), use_container_width=True)
    
    elif strategy == "Channels - Files séparées":
        st.markdown("""
        **Principe:** Une file dédiée par population avec ressources allouées
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_servers = n_servers
        
        with col1:
            servers_sup = st.number_input("Serveurs SUP", 1, total_servers, max(1, total_servers//4))
        with col2:
            servers_spe = st.number_input("Serveurs SPÉ", 1, total_servers, max(1, total_servers//4))
        with col3:
            servers_ing = st.number_input("Serveurs ING1", 1, total_servers, max(1, total_servers//4))
        with col4:
            servers_admin = st.number_input("Serveurs Admin", 1, total_servers, max(1, total_servers//8))
        
        total_alloc = servers_sup + servers_spe + servers_ing + servers_admin
        st.info(f"Total alloué: {total_alloc} serveurs (disponibles: {total_servers})")
    
    elif strategy == "Hybride - Pool avec priorités":
        st.markdown("""
        **Principe:** Pool partagé avec ordonnancement à priorités
        
        Les admins/staff ont la priorité la plus haute, suivis des ING1, puis des Prépas.
        """)
        
        priorities = st.multiselect(
            "Ordre de priorité (haut → bas)",
            ["Admin/Staff", "ING1", "Prépa SPÉ", "Prépa SUP"],
            default=["Admin/Staff", "ING1", "Prépa SPÉ", "Prépa SUP"]
        )
        
        if priorities:
            st.markdown("**Ordre de traitement:**")
            for i, p in enumerate(priorities, 1):
                st.write(f"{i}. {p}")
    
    st.divider()
    
    # Impact deadline
    st.subheader("⏰ Impact d'une Deadline")
    
    hours_to_deadline = st.slider("Heures avant deadline", 0.0, 48.0, 24.0, 0.5)
    
    impact_data = []
    for student_type, persona in personas.items():
        base_rate = persona.get_arrival_rate(14)
        deadline_rate = persona.get_arrival_rate(14, hours_to_deadline=hours_to_deadline)
        multiplier = deadline_rate / base_rate if base_rate > 0 else 1.0
        
        impact_data.append({
            'Population': persona.name,
            'Taux normal': base_rate,
            'Taux deadline': deadline_rate,
            'Multiplicateur': multiplier
        })
    
    df_impact = pd.DataFrame(impact_data)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_impact['Population'], y=df_impact['Taux normal'], name='Normal'))
    fig.add_trace(go.Bar(x=df_impact['Population'], y=df_impact['Taux deadline'], name=f'Deadline {hours_to_deadline:.0f}h'))
    fig.update_layout(barmode='group', height=350, yaxis_title="Soumissions/heure")
    st.plotly_chart(apply_dark_theme(fig), use_container_width=True)


# ==============================================================================
# OPTIMISATION COÛT / QUALITÉ DE SERVICE
# ==============================================================================

def render_optimization_tab(mu_rate: float, n_servers: int, buffer_size: int):
    """Onglet d'optimisation coût/performance."""
    st.header("💰 Optimisation Coût / Qualité de Service")
    
    st.markdown("""
    <div class="formula-box">
    <strong>Fonction objectif:</strong><br>
    min [ α × E[T] + β × Coût(K, c, μ) ] avec α + β = 1
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    - **E[T]** = Temps moyen de séjour (temps d'attente + service)
    - **Coût** = Coût serveurs + Coût rejets + Coût insatisfaction
    - **α** = Poids accordé à la performance
    - **β** = Poids accordé au coût
    """)
    
    st.divider()
    
    # Configuration
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Taux d'arrivée")
        
        input_mode = st.radio("Source", ["Manuel", "Personas"], horizontal=True)
        
        if input_mode == "Manuel":
            lambda_rate = st.number_input("λ (soumissions/min)", 1.0, 200.0, 30.0, key="opt_lambda")
        else:
            personas = PersonaFactory.create_all_personas()
            selected = st.multiselect(
                "Populations actives",
                [p.name for p in personas.values()],
                default=[p.name for p in personas.values()]
            )
            
            lambda_rate = 0.0
            for p in personas.values():
                if p.name in selected:
                    lambda_rate += p.get_arrival_rate(14) / 60.0
            
            st.info(f"λ combiné = {lambda_rate:.2f} sub/min")
    
    with col2:
        st.subheader("⚖️ Pondération")
        
        alpha = st.slider("α (poids coût)", 0.0, 1.0, 0.7, 0.05)
        st.write(f"β (poids temps) = {1-alpha:.2f}")
        
        st.markdown("---")
        st.subheader("💵 Modèle de coût")
        
        cost_server = st.number_input("Coût serveur (€/h)", 0.1, 10.0, 0.50, 0.1)
        cost_reject = st.number_input("Coût rejet (€)", 0.01, 1.0, 0.05, 0.01)
        cost_wait = st.number_input("Pénalité attente (€/min)", 0.001, 0.1, 0.01, 0.001)
    
    st.divider()
    
    # Paramètres de recherche
    st.subheader("🔍 Espace de recherche")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        max_servers = st.slider("Max serveurs", 5, 30, 15)
    with col2:
        mu_min = st.number_input("μ min", 5.0, 20.0, 5.0)
        mu_max = st.number_input("μ max", 10.0, 50.0, 25.0)
    with col3:
        resolution = st.slider("Résolution", 10, 40, 20)
    
    if st.button("🚀 Lancer l'optimisation", type="primary"):
        run_optimization(lambda_rate, alpha, cost_server, cost_reject, cost_wait,
                        max_servers, mu_min, mu_max, resolution)


def run_optimization(lambda_rate, alpha, cost_server, cost_reject, cost_wait,
                    max_servers, mu_min, mu_max, resolution):
    """Exécute l'optimisation et affiche les heatmaps."""
    
    with st.spinner("Optimisation en cours..."):
        server_range = np.arange(1, max_servers + 1)
        mu_range = np.linspace(mu_min, mu_max, resolution)
        
        Z_cost = np.zeros((len(mu_range), len(server_range)))
        Z_time = np.zeros((len(mu_range), len(server_range)))
        Z_score = np.zeros((len(mu_range), len(server_range)))
        
        for i, mu in enumerate(mu_range):
            for j, c in enumerate(server_range):
                rho = lambda_rate / (c * mu)
                
                if rho >= 1:
                    Z_cost[i, j] = np.nan
                    Z_time[i, j] = np.nan
                    Z_score[i, j] = np.nan
                else:
                    try:
                        queue = GenericQueue(lambda_rate, mu, f"M/M/{int(c)}", c=int(c))
                        metrics = queue.compute_theoretical_metrics()
                        
                        # Coût horaire
                        server_cost = c * cost_server
                        wait_cost = metrics.Wq * cost_wait * lambda_rate * 60
                        total_cost = server_cost + wait_cost
                        
                        Z_cost[i, j] = total_cost
                        Z_time[i, j] = metrics.W
                        
                        # Score normalisé
                        Z_score[i, j] = alpha * total_cost + (1 - alpha) * metrics.W
                        
                    except:
                        Z_cost[i, j] = np.nan
                        Z_time[i, j] = np.nan
                        Z_score[i, j] = np.nan
        
        # Normaliser le score pour affichage
        valid = ~np.isnan(Z_score)
        if valid.any():
            Z_score_norm = (Z_score - np.nanmin(Z_score)) / (np.nanmax(Z_score) - np.nanmin(Z_score))
        else:
            st.error("Aucune configuration stable trouvée")
            return
        
        # Afficher les heatmaps
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💵 Heatmap Coût Total (€/h)")
            fig_cost = go.Figure(data=go.Heatmap(
                z=Z_cost,
                x=server_range,
                y=mu_range,
                colorscale='RdYlGn_r',
                colorbar=dict(title="€/h")
            ))
            fig_cost.update_layout(
                xaxis_title='Nombre de serveurs (c)',
                yaxis_title='Taux de service μ',
                height=400
            )
            st.plotly_chart(apply_dark_theme(fig_cost), use_container_width=True)
        
        with col2:
            st.markdown("### ⏱️ Heatmap Temps de Séjour (min)")
            fig_time = go.Figure(data=go.Heatmap(
                z=Z_time,
                x=server_range,
                y=mu_range,
                colorscale='RdYlGn_r',
                colorbar=dict(title="min")
            ))
            fig_time.update_layout(
                xaxis_title='Nombre de serveurs (c)',
                yaxis_title='Taux de service μ',
                height=400
            )
            st.plotly_chart(apply_dark_theme(fig_time), use_container_width=True)
        
        # Score combiné
        st.markdown("### 🎯 Heatmap Score Combiné (α×Coût + β×Temps)")
        fig_score = go.Figure(data=go.Heatmap(
            z=Z_score_norm,
            x=server_range,
            y=mu_range,
            colorscale='RdYlGn_r',
            colorbar=dict(title="Score")
        ))
        fig_score.update_layout(
            xaxis_title='Nombre de serveurs (c)',
            yaxis_title='Taux de service μ',
            height=400
        )
        st.plotly_chart(apply_dark_theme(fig_score), use_container_width=True)
        
        # Configuration optimale
        min_idx = np.nanargmin(Z_score)
        min_i, min_j = np.unravel_index(min_idx, Z_score.shape)
        
        opt_mu = mu_range[min_i]
        opt_c = server_range[min_j]
        opt_cost = Z_cost[min_i, min_j]
        opt_time = Z_time[min_i, min_j]
        
        st.success(f"""
        ### ✅ Configuration Optimale
        
        | Paramètre | Valeur |
        |-----------|--------|
        | **Serveurs (c)** | {opt_c} |
        | **Taux service (μ)** | {opt_mu:.2f}/min |
        | **Coût total** | {opt_cost:.2f} €/h |
        | **Temps de séjour** | {opt_time:.3f} min |
        | **Utilisation (ρ)** | {lambda_rate/(opt_c*opt_mu):.1%} |
        """)


# ==============================================================================
# AUTO-SCALING
# ==============================================================================

def render_autoscaling_tab(mu_rate1: float, mu_rate2: float, n_servers: int, K1: int, K2: int):
    """Onglet des stratégies d'auto-scaling."""
    st.header("📈 Stratégies d'Auto-Scaling")
    
    st.markdown("""
    <div class="scenario-box">
    <strong>Objectif:</strong> Adapter dynamiquement le nombre de serveurs à la charge
    </div>
    """, unsafe_allow_html=True)
    
    # Types de scaling
    scaling_type = st.radio(
        "Stratégie de scaling",
        ["🔒 Fixe", "📅 Programmé", "⚡ Réactif", "🔮 Prédictif"],
        horizontal=True
    )
    
    st.divider()
    
    if scaling_type == "🔒 Fixe":
        st.markdown("""
        **Scaling Fixe:** Nombre constant de serveurs
        
        ✅ Simple à gérer  
        ❌ Pas d'adaptation à la charge
        """)
        
        st.metric("Serveurs fixes", n_servers)
    
    elif scaling_type == "📅 Programmé":
        st.markdown("""
        **Scaling Programmé:** Nombre de serveurs selon l'heure
        
        ✅ Adapté aux patterns connus  
        ❌ Ne gère pas les pics imprévus
        """)
        
        st.subheader("Configuration horaire")
        
        schedule = {}
        cols = st.columns(6)
        for i, h in enumerate([0, 4, 8, 12, 16, 20]):
            with cols[i]:
                schedule[h] = st.number_input(f"{h}h-{h+4}h", 1, 20, n_servers, key=f"sched_{h}")
        
        # Visualisation
        hours = list(range(24))
        servers = []
        for h in hours:
            for start in sorted(schedule.keys(), reverse=True):
                if h >= start:
                    servers.append(schedule[start])
                    break
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hours, y=servers, mode='lines+markers', fill='tozeroy', name='Serveurs'))
        fig.update_layout(xaxis_title="Heure", yaxis_title="Serveurs", height=300)
        st.plotly_chart(apply_dark_theme(fig), use_container_width=True)
    
    elif scaling_type == "⚡ Réactif":
        st.markdown("""
        **Scaling Réactif:** Ajustement basé sur la charge actuelle
        
        ✅ Réagit aux pics  
        ❌ Temps de réaction (cooldown)
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            scale_up = st.slider("Seuil scale-up (ρ)", 0.5, 0.95, 0.8)
            scale_up_inc = st.number_input("Serveurs à ajouter", 1, 5, 2)
        with col2:
            scale_down = st.slider("Seuil scale-down (ρ)", 0.1, 0.5, 0.3)
            scale_down_inc = st.number_input("Serveurs à retirer", 1, 3, 1)
        
        cooldown = st.slider("Cooldown (min)", 1, 30, 10)
        
        st.markdown(f"""
        **Règles:**
        - Si ρ > {scale_up:.0%} → +{scale_up_inc} serveurs
        - Si ρ < {scale_down:.0%} → -{scale_down_inc} serveur(s)
        - Délai entre ajustements: {cooldown} min
        """)
    
    elif scaling_type == "🔮 Prédictif":
        st.markdown("""
        **Scaling Prédictif:** Anticipation basée sur les données historiques
        
        ✅ Évite les temps de réaction  
        ❌ Nécessite des données historiques
        """)
        
        st.info("Le scaling prédictif utilise les patterns de soumission historiques pour anticiper la charge.")
        
        # Simulation d'une prédiction
        st.subheader("Prédiction de charge (exemple)")
        
        hours = list(range(24))
        predicted_load = [20 + 30 * np.sin((h - 14) * np.pi / 12) ** 2 for h in hours]
        predicted_servers = [max(2, int(load / mu_rate1) + 1) for load in predicted_load]
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=hours, y=predicted_load, name="Charge prédite", line=dict(color='blue')), secondary_y=False)
        fig.add_trace(go.Scatter(x=hours, y=predicted_servers, name="Serveurs recommandés", line=dict(color='green', dash='dash')), secondary_y=True)
        fig.update_layout(height=350)
        fig.update_yaxes(title_text="Charge (tags/min)", secondary_y=False)
        fig.update_yaxes(title_text="Serveurs", secondary_y=True)
        st.plotly_chart(apply_dark_theme(fig), use_container_width=True)
    
    st.divider()
    
    # Comparaison des stratégies
    st.subheader("📊 Comparaison des stratégies")
    
    comparison_data = {
        'Stratégie': ['Fixe', 'Programmé', 'Réactif', 'Prédictif'],
        'Réactivité': ['❌ Aucune', '⚠️ Limitée', '✅ Bonne', '✅ Excellente'],
        'Complexité': ['✅ Simple', '⚠️ Moyenne', '⚠️ Moyenne', '❌ Complexe'],
        'Coût': ['⚠️ Variable', '✅ Optimisé', '✅ Optimisé', '✅ Très optimisé'],
        'Pics imprévus': ['❌ Non géré', '❌ Non géré', '✅ Géré', '⚠️ Partiellement']
    }
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)


# ==============================================================================
# BENCHMARK DES MODÈLES
# ==============================================================================

def render_benchmark_tab(mu_rate: float, n_servers: int, buffer_size: int):
    """Onglet de benchmark comparatif des modèles de files."""
    st.header("🔬 Benchmark des Modèles de Files d'Attente")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Sélection des modèles")
        
        st.markdown("**Mono-serveur:**")
        show_mm1 = st.checkbox("M/M/1", value=True, key="b_mm1")
        show_md1 = st.checkbox("M/D/1", value=True, key="b_md1")
        show_mg1 = st.checkbox("M/G/1", value=True, key="b_mg1")
        
        st.markdown("**Multi-serveurs:**")
        show_mmc = st.checkbox("M/M/c", value=True, key="b_mmc")
        show_mdc = st.checkbox("M/D/c", value=True, key="b_mdc")
        show_mgc = st.checkbox("M/G/c", value=True, key="b_mgc")
        
        cv_squared = st.slider("CV² (pour M/G/*)", 0.0, 2.0, 1.0, 0.1)
    
    with col1:
        lambda_rate = st.slider("λ - Taux d'arrivée", 1.0, 100.0, 30.0, key="b_lambda")
        mu_rate_b = st.slider("μ - Taux de service", 1.0, 50.0, 10.0, key="b_mu")
        
        models_data = []
        
        # Mono-serveur
        if show_mm1 and lambda_rate < mu_rate_b:
            queue = GenericQueue(lambda_rate, mu_rate_b, "M/M/1")
            m = queue.compute_theoretical_metrics()
            models_data.append({'Modèle': 'M/M/1', 'L': m.L, 'Lq': m.Lq, 'W': m.W, 'Wq': m.Wq, 'ρ': m.rho})
        
        if show_md1 and lambda_rate < mu_rate_b:
            queue = GenericQueue(lambda_rate, mu_rate_b, "M/D/1")
            m = queue.compute_theoretical_metrics()
            models_data.append({'Modèle': 'M/D/1', 'L': m.L, 'Lq': m.Lq, 'W': m.W, 'Wq': m.Wq, 'ρ': m.rho})
        
        if show_mg1 and lambda_rate < mu_rate_b:
            queue = GenericQueue(lambda_rate, mu_rate_b, "M/G/1")
            queue.service_variance = cv_squared * (1/mu_rate_b)**2
            m = queue.compute_theoretical_metrics()
            models_data.append({'Modèle': f'M/G/1 (CV²={cv_squared})', 'L': m.L, 'Lq': m.Lq, 'W': m.W, 'Wq': m.Wq, 'ρ': m.rho})
        
        # Multi-serveurs
        if show_mmc and lambda_rate < n_servers * mu_rate_b:
            queue = GenericQueue(lambda_rate, mu_rate_b, f"M/M/{n_servers}", c=n_servers)
            m = queue.compute_theoretical_metrics()
            models_data.append({'Modèle': f'M/M/{n_servers}', 'L': m.L, 'Lq': m.Lq, 'W': m.W, 'Wq': m.Wq, 'ρ': m.rho})
        
        if show_mdc and lambda_rate < n_servers * mu_rate_b:
            queue = GenericQueue(lambda_rate, mu_rate_b, f"M/D/{n_servers}", c=n_servers)
            m = queue.compute_theoretical_metrics()
            models_data.append({'Modèle': f'M/D/{n_servers}', 'L': m.L, 'Lq': m.Lq, 'W': m.W, 'Wq': m.Wq, 'ρ': m.rho})
        
        if show_mgc and lambda_rate < n_servers * mu_rate_b:
            queue = GenericQueue(lambda_rate, mu_rate_b, f"M/G/{n_servers}", c=n_servers)
            queue.service_variance = cv_squared * (1/mu_rate_b)**2
            m = queue.compute_theoretical_metrics()
            models_data.append({'Modèle': f'M/G/{n_servers} (CV²={cv_squared})', 'L': m.L, 'Lq': m.Lq, 'W': m.W, 'Wq': m.Wq, 'ρ': m.rho})
        
        if models_data:
            df = pd.DataFrame(models_data)
            
            st.subheader("📊 Comparaison théorique")
            st.dataframe(df.style.format({
                'L': '{:.2f}',
                'Lq': '{:.2f}',
                'W': '{:.4f}',
                'Wq': '{:.4f}',
                'ρ': '{:.2%}'
            }), use_container_width=True)
            
            # Graphiques
            fig = make_subplots(rows=1, cols=2, subplot_titles=['Temps d\'attente (Wq)', 'Longueur de queue (Lq)'])
            
            colors = px.colors.qualitative.Set2
            
            fig.add_trace(go.Bar(x=df['Modèle'], y=df['Wq'], marker_color=colors[:len(df)]), row=1, col=1)
            fig.add_trace(go.Bar(x=df['Modèle'], y=df['Lq'], marker_color=colors[:len(df)]), row=1, col=2)
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(apply_dark_theme(fig), use_container_width=True)
        else:
            st.warning("Aucun modèle stable sélectionné. Vérifiez que λ < c×μ")
    
    st.divider()
    
    # Simulation comparative
    st.subheader("🎲 Simulation Monte Carlo Comparative")
    
    col_s1, col_s2 = st.columns([1, 3])
    
    with col_s1:
        sim_n = st.number_input("Clients", 100, 10000, 2000, key="b_sim_n")
        sim_runs = st.number_input("Runs", 1, 20, 5, key="b_sim_runs")
        run_benchmark = st.button("▶️ Lancer benchmark", type="primary")
    
    with col_s2:
        if run_benchmark:
            run_model_benchmark(lambda_rate, mu_rate_b, n_servers, cv_squared, sim_n, sim_runs,
                               show_mm1, show_md1, show_mg1, show_mmc, show_mdc, show_mgc)


def run_model_benchmark(lambda_rate, mu_rate, n_servers, cv_squared, n_customers, n_runs,
                       show_mm1, show_md1, show_mg1, show_mmc, show_mdc, show_mgc):
    """Exécute un benchmark Monte Carlo des modèles."""
    
    with st.spinner("Benchmark en cours..."):
        results = []
        
        models_to_test = []
        if show_mm1 and lambda_rate < mu_rate:
            models_to_test.append(("M/M/1", GenericQueue(lambda_rate, mu_rate, "M/M/1")))
        if show_md1 and lambda_rate < mu_rate:
            models_to_test.append(("M/D/1", GenericQueue(lambda_rate, mu_rate, "M/D/1")))
        if show_mg1 and lambda_rate < mu_rate:
            q = GenericQueue(lambda_rate, mu_rate, "M/G/1")
            q.service_variance = cv_squared * (1/mu_rate)**2
            models_to_test.append(("M/G/1", q))
        if show_mmc and lambda_rate < n_servers * mu_rate:
            models_to_test.append((f"M/M/{n_servers}", GenericQueue(lambda_rate, mu_rate, f"M/M/{n_servers}", c=n_servers)))
        if show_mdc and lambda_rate < n_servers * mu_rate:
            models_to_test.append((f"M/D/{n_servers}", GenericQueue(lambda_rate, mu_rate, f"M/D/{n_servers}", c=n_servers)))
        if show_mgc and lambda_rate < n_servers * mu_rate:
            q = GenericQueue(lambda_rate, mu_rate, f"M/G/{n_servers}", c=n_servers)
            q.service_variance = cv_squared * (1/mu_rate)**2
            models_to_test.append((f"M/G/{n_servers}", q))
        
        progress = st.progress(0)
        total = len(models_to_test) * n_runs
        current = 0
        
        for name, queue in models_to_test:
            for run in range(n_runs):
                try:
                    res = queue.simulate(n_customers=n_customers)
                    results.append({
                        'Modèle': name,
                        'Run': run + 1,
                        'Wq (sim)': np.mean(res.waiting_times) if len(res.waiting_times) > 0 else 0,
                        'W (sim)': np.mean(res.system_times) if len(res.system_times) > 0 else 0,
                        'Lq max': np.max(res.queue_length_trace) if len(res.queue_length_trace) > 0 else 0
                    })
                except Exception as e:
                    st.warning(f"{name}: {e}")
                
                current += 1
                progress.progress(current / total)
        
        progress.empty()
        
        if results:
            df = pd.DataFrame(results)
            
            # Résumé
            summary = df.groupby('Modèle').agg({
                'Wq (sim)': ['mean', 'std'],
                'W (sim)': ['mean', 'std'],
                'Lq max': ['mean', 'std']
            }).round(4)
            
            st.markdown("### Résultats")
            
            # Créer un DataFrame plus lisible
            summary_display = pd.DataFrame()
            for col in ['Wq (sim)', 'W (sim)', 'Lq max']:
                mean = summary[col]['mean']
                std = summary[col]['std']
                summary_display[col] = mean.astype(str) + ' ± ' + std.astype(str)
            
            st.dataframe(summary_display, use_container_width=True)
            
            # Box plots
            fig = make_subplots(rows=1, cols=2, subplot_titles=['Temps d\'attente', 'Temps de séjour'])
            
            for model in df['Modèle'].unique():
                model_data = df[df['Modèle'] == model]
                fig.add_trace(go.Box(y=model_data['Wq (sim)'], name=model), row=1, col=1)
                fig.add_trace(go.Box(y=model_data['W (sim)'], name=model, showlegend=False), row=1, col=2)
            
            fig.update_layout(height=400)
            st.plotly_chart(apply_dark_theme(fig), use_container_width=True)


if __name__ == "__main__":
    main()
