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
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personnalisé
    st.markdown("""
    <style>
    .main-header {
        font-size: 5.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        margin-top: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        letter-spacing: 2px;
        animation: fadeInDown 1s ease-out;
    }
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .sub-header {
        font-size: 1.4rem;
        color: #888;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
        letter-spacing: 1px;
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
    
    st.markdown('<p class="main-header">Moulinette Simulator</p>', unsafe_allow_html=True)
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
        "Scénario 1: Waterfall",
        "Backup",
        "Scénario 2: Channels & Dams",
        "Optimisation Coût/QoS",
        "Auto-Scaling"
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

# ==============================================================================
# SCÉNARIO 1: MODÈLE WATERFALL
# ==============================================================================

def render_waterfall_scenario(mu_rate1: float, mu_rate2: float, n_servers: int, K1: int, K2: int, file2_model: str):
    """Scénario 1: Modèle Waterfall avec files infinies puis finies."""
    st.header("Scénario 1: Modèle Waterfall")
    
    st.markdown("""
    <div class="scenario-box">
    <strong>Architecture en cascade de la moulinette:</strong><br>
    <code>Étudiants → Buffer₁ (K₁) → Runners (c) → Buffer₂ (K₂) → Frontend (1 serveur)</code><br><br>
    <strong>Modèle théorique:</strong> Chaîne de files M/M/c/K₁ → M/M/1/K₂ (ou M/D/1/K₂)
    </div>
    """, unsafe_allow_html=True)

    
    # Configuration de la simulation
    st.subheader("Configuration de la Simulation")
    
    col_config1, col_config2, col_config3 = st.columns(3)
    
    with col_config1:
        lambda_rate = st.slider(
            "λ - Taux d'arrivée (tags/min)",
            min_value=1.0, max_value=100.0, value=30.0, step=1.0,
            key="waterfall_lambda",
            help="Nombre de tags soumis par minute"
        )
    
    with col_config2:
        capacity_mode = st.radio(
            "Mode de capacité",
            ["Files infinies (K=∞)", "Files finies (K limité)"],
            key="waterfall_capacity",
            help="Files infinies: théorie classique. Files finies: modèle réaliste avec rejets."
        )
    
    with col_config3:
        use_md1 = "M/D/1" in file2_model
        st.metric("Type File 2", "M/D/1 (Déterministe)" if use_md1 else "M/M/1 (Exponentiel)")
        if use_md1:
            st.info("✓ Service déterministe: -50% temps d'attente")
    
    st.divider()
    
    # Section analyse théorique
    st.subheader("Analyse Théorique des Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### File 1: Exécution des tests")
        st.markdown(f"<div style='background:#2d4059;padding:1rem;border-radius:8px;margin-bottom:1rem;'>"
                   f"<strong>Modèle:</strong> M/M/{n_servers}{'/' + str(K1) if capacity_mode == 'Files finies (K limité)' else ''}<br>"
                   f"<strong>Serveurs:</strong> {n_servers} runners parallèles<br>"
                   f"<strong>Service:</strong> μ₁ = {mu_rate1} tests/min/runner"
                   f"</div>", unsafe_allow_html=True)
        
        # Condition de stabilité
        rho1 = lambda_rate / (n_servers * mu_rate1)
        stable1 = rho1 < 1 or capacity_mode == "Files finies (K limité)"
        
        st.markdown(f"""
        <div class="formula-box">
        <strong>Condition de stabilité:</strong><br>
        ρ₁ = λ/(c×μ₁) = {lambda_rate:.1f}/({n_servers}×{mu_rate1:.1f}) = <strong>{rho1:.3f}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        if rho1 < 1:
            st.success(f"✓ Système stable (ρ₁ = {rho1:.2%} < 100%)")
        else:
            if capacity_mode == "Files finies (K limité)":
                st.warning(f"ρ₁ = {rho1:.2%} ≥ 100% mais stable grâce au buffer fini K₁={K1}")
            else:
                st.error(f"✗ Système instable (ρ₁ = {rho1:.2%} ≥ 100%)")
        
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
        st.subheader(f"File 2: Renvoi des résultats ({'M/D/1' if use_md1 else 'M/M/1'})")
        
        if lambda_eff > 0:
            # Le taux d'entrée en file 2 = taux de sortie de file 1
            lambda2 = lambda_eff
            rho2 = lambda2 / mu_rate2
            
            st.markdown(f"""
            **Taux d'arrivée effectif:** λ₂ = λ_eff = {lambda2:.2f} tags/min
            
            **Condition de stabilité:** ρ₂ = λ₂/μ₂ = {lambda2:.2f}/{mu_rate2:.1f} = **{rho2:.3f}**
            """)
            
            if rho2 < 1:
                st.success(f"Système stable (ρ₂ = {rho2:.2%} < 100%)")
            else:
                if capacity_mode == "Files finies (K limité)":
                    st.warning(f"ρ₂ = {rho2:.2%} ≥ 100% mais stable grâce au buffer fini K₂={K2}")
                else:
                    st.error(f"Système instable (ρ₂ = {rho2:.2%} ≥ 100%)")
            
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
                    st.info("**M/D/1:** Temps d'attente réduit de ~50% grâce au service déterministe")
                
            except Exception as e:
                st.error(f"Erreur de calcul: {e}")
                metrics2 = None
                P_K2 = 0
        else:
            st.warning("Calculez d'abord la File 1")
            metrics2 = None
            P_K2 = 0
    
    st.divider()
    
    st.subheader("Simulation Monte Carlo")
    
    col_sim1, col_sim2 = st.columns([1, 2])
    
    with col_sim1:
        n_customers = st.number_input("Nombre de clients", 100, 10000, 2000, step=100, key="waterfall_n")
        n_runs = st.number_input("Nombre de répétitions", 1, 20, 5, key="waterfall_runs")
        run_sim = st.button("Lancer simulation Waterfall", type="primary", key="waterfall_run")
    
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
        mode='markers',
        marker=dict(size=40, color='#2ecc71', symbol='circle'),
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
        mode='markers',
        marker=dict(size=50, color='#3498db', symbol='square'),
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
        mode='markers',
        marker=dict(size=60, color='#e74c3c', symbol='square'),
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
        mode='markers',
        marker=dict(size=50, color='#9b59b6', symbol='square'),
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
        mode='markers',
        marker=dict(size=50, color='#f39c12', symbol='circle'),
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
            fig = make_subplots(rows=1, subplot_titles=['Temps par file'])
            
            fig.add_trace(go.Box(y=df['file1_wait'], name='Attente F1'), row=1, col=1)
            fig.add_trace(go.Box(y=df['file2_wait'], name='Attente F2'), row=1, col=1)
            fig.add_trace(go.Box(y=df['file1_system'], name='Séjour F1'), row=1, col=1)
            fig.add_trace(go.Box(y=df['file2_system'], name='Séjour F2'), row=1, col=1)
                        
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
    st.header("Mécanismes de Backup")
    
    st.markdown("""
    <div class="scenario-box">
    <strong>Problématique:</strong> Quand la file 2 est saturée, les résultats sont perdus → <em>pages blanches</em><br><br>
    <strong>Solution:</strong> Sauvegarder les résultats de moulinettage <em>avant</em> l'envoi vers la file 2.<br>
    Si la file 2 rejette, les données peuvent être récupérées depuis le backup.
    </div>
    """, unsafe_allow_html=True)
    
    # Schéma du système avec backup
    st.subheader("Architecture avec Backup")
    
    fig_arch = go.Figure()
    y_base = 0.5
    
    # File 1
    fig_arch.add_trace(go.Scatter(x=[0], y=[y_base], mode='markers',
        marker=dict(size=40, color='#2ecc71'), showlegend=False))
    fig_arch.add_annotation(x=0, y=y_base-0.2, text="Étudiants", showarrow=False)
    
    fig_arch.add_annotation(x=0.4, y=y_base, ax=0.1, ay=y_base, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2)
    
    fig_arch.add_trace(go.Scatter(x=[0.8], y=[y_base], mode='markers',
        marker=dict(size=50, color='#3498db', symbol='square'), showlegend=False))
    fig_arch.add_annotation(x=0.8, y=y_base-0.2, text=f"File 1\n(M/M/{n_servers}/K₁)", showarrow=False)
    
    # Backup (point de sauvegarde)
    fig_arch.add_annotation(x=1.3, y=y_base, ax=1.0, ay=y_base, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2)
    
    fig_arch.add_trace(go.Scatter(x=[1.6], y=[y_base], mode='markers',
        marker=dict(size=50, color='#f39c12', symbol='diamond'), showlegend=False))
    fig_arch.add_annotation(x=1.6, y=y_base-0.2, text="BACKUP\n(sauvegarde)", showarrow=False, font=dict(color='#f39c12'))
    
    # Flèche vers File 2
    fig_arch.add_annotation(x=2.1, y=y_base, ax=1.8, ay=y_base, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2)
    
    # File 2
    fig_arch.add_trace(go.Scatter(x=[2.4], y=[y_base], mode='markers',
        marker=dict(size=50, color='#9b59b6', symbol='square'), showlegend=False))
    fig_arch.add_annotation(x=2.4, y=y_base-0.2, text=f"File 2\n(M/M/1/K₂)", showarrow=False)
    
    # Flèche de récupération (backup -> retry)
    fig_arch.add_trace(go.Scatter(x=[1.6, 2.0, 2.4], y=[y_base+0.25, y_base+0.35, y_base+0.25], 
        mode='lines', line=dict(color='#e74c3c', dash='dash', width=2), showlegend=False))
    fig_arch.add_annotation(x=2.0, y=y_base+0.4, text="Récupération si rejet", showarrow=False, font=dict(color='#e74c3c', size=10))
    
    # Frontend
    fig_arch.add_annotation(x=2.9, y=y_base, ax=2.6, ay=y_base, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2)
    
    fig_arch.add_trace(go.Scatter(x=[3.2], y=[y_base], mode='markers',
        marker=dict(size=40, color='#27ae60'), showlegend=False))
    fig_arch.add_annotation(x=3.2, y=y_base-0.2, text="Résultat\naffiché", showarrow=False)
    
    fig_arch.update_layout(
        height=250,
        xaxis=dict(showgrid=False, showticklabels=False, range=[-0.3, 3.5]),
        yaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(apply_dark_theme(fig_arch), use_container_width=True)
    
    st.subheader("Configuration de la simulation")
    
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
    st.subheader("Simulation Monte Carlo - Analyse de sensibilité du backup")
    
    st.markdown("""
    La simulation compare automatiquement 5 valeurs de probabilité de backup:
    - **P=0%**: Aucun backup (référence)
    - **P=25%**: Backup aléatoire faible
    - **P=50%**: Backup aléatoire moyen
    - **P=75%**: Backup aléatoire élevé
    - **P=100%**: Backup systématique (optimal)
    """)
    
    run_backup_sim = st.button("Lancer simulation complète", type="primary", key="backup_sim_run", use_container_width=True)
    
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
            st.markdown("### Résultats comparatifs")
            
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
            st.markdown("### Visualisations comparatives")
            
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
            st.markdown("### Distribution des résultats")
            
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
                st.markdown("### Évolution temporelle (simulation avec P=100%)")
                
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
    **Mécanisme de backup implémenté:**
    1. Les résultats de moulinettage sont sauvegardés **avant** l'envoi vers la file 2
    2. Si la file 2 rejette (pleine), les données sont **récupérées depuis le backup**
    3. Une nouvelle tentative d'envoi est planifiée jusqu'au succès
    4. Cela élimine (presque) totalement les pages blanches
    """)


# ==============================================================================
# CHANNELS & DAMS: POPULATIONS DIFFÉRENCIÉES
# ==============================================================================

def render_channels_dams_tab(mu_rate1: float, n_servers: int, K1: int):
    """Onglet Channels & Dams - Analyse par persona avec files séparées et priorisées."""
    st.header("Scénario 2: Channels & Dams")
    
    # Créer les personas
    personas = PersonaFactory.create_all_personas()
    persona_names = [p.name for p in personas.values()]
    
    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    st.subheader("⚙️ Configuration")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Sélection du persona
        selected_persona_name = st.selectbox(
            "👤 Sélectionner un Persona",
            persona_names,
            index=0
        )
        
        # Récupérer le persona sélectionné
        selected_persona = None
        for p in personas.values():
            if p.name == selected_persona_name:
                selected_persona = p
                break
        
        if selected_persona is None:
            st.error("Persona non trouvé")
            return
        
        # Afficher les infos du persona
        st.markdown("---")
        st.markdown(f"**Caractéristiques de {selected_persona.name}:**")
        st.write(f"• Population: **{selected_persona.population_size}** utilisateurs")
        st.write(f"• Taux de base: **{selected_persona.base_submission_rate:.2f}** tags/h/user")
        st.write(f"• Complexité: **{selected_persona.avg_test_complexity:.2f}** min/job")
        
        service_rate_per_server = 1.0 / selected_persona.avg_test_complexity
        st.write(f"• Vitesse traitement: **{service_rate_per_server:.1f}** jobs/min/serveur")
    
    with col2:
        # Paramètres de simulation
        st.markdown("**Paramètres de simulation:**")
        
        simulation_duration = st.slider("Durée de simulation (heures)", 1, 24, 4)
        start_hour = st.slider("Heure de début", 0, 23, 14)
        num_servers = st.slider("Nombre de serveurs", 1, 50, n_servers)
        
        # Calculer la capacité totale
        total_capacity = num_servers * service_rate_per_server
        st.metric("Capacité totale", f"{total_capacity:.1f} jobs/min")
    
    st.divider()
    
    # ============================================================================
    # SIMULATION PERSONNALISÉE PAR PERSONA
    # ============================================================================
    st.subheader(f"📊 Simulation pour {selected_persona.name}")
    
    # Générer les données de simulation
    time_steps = simulation_duration * 60  # minutes
    times = np.arange(time_steps)
    
    # Comportement différent selon le persona
    if selected_persona.student_type == StudentType.PREPA:
        # PRÉPA: Burst de tous les tags en 15 minutes au début, puis traitement
        st.info("🎓 **Mode Prépa**: Tous les étudiants rendent en même temps (burst de 15 min au début)")
        
        arrivals = np.zeros(time_steps)
        burst_duration = 15  # 15 minutes de burst
        
        # Tous les prépas rendent pendant les 15 premières minutes
        total_submissions = selected_persona.population_size  # 1 tag par étudiant
        arrivals_per_minute = total_submissions / burst_duration
        
        for t in range(min(burst_duration, time_steps)):
            arrivals[t] = arrivals_per_minute
        
    else:
        # INGÉNIEUR et ADMIN: Flux continu basé sur le taux d'arrivée horaire avec variabilité
        arrivals = np.zeros(time_steps)
        np.random.seed(42)  # Pour reproductibilité
        
        for t in range(time_steps):
            current_hour = (start_hour + t // 60) % 24
            # Taux d'arrivée en jobs/min (conversion depuis jobs/heure)
            hourly_rate = selected_persona.get_arrival_rate(current_hour)
            base_rate = hourly_rate / 60.0
            
            # Ajouter de la variabilité aléatoire
            # Pour les ingénieurs: variance plus élevée (comportement plus erratique)
            # Pour admin: variance plus faible (comportement plus régulier)
            if selected_persona.student_type == StudentType.INGENIEUR:
                # Variabilité ±40% autour du taux de base
                noise_factor = np.random.uniform(0.6, 1.4)
            else:
                # Variabilité ±20% pour admin
                noise_factor = np.random.uniform(0.8, 1.2)
            
            arrivals[t] = base_rate * noise_factor
    
    # Simulation de la file
    queue_length = np.zeros(time_steps)
    cumulative_arrivals = np.zeros(time_steps)
    cumulative_served = np.zeros(time_steps)
    queue = 0.0
    total_arrived = 0.0
    total_served = 0.0
    
    for t in range(time_steps):
        # 1. Traiter les jobs (capacité de service)
        served = min(queue, total_capacity)
        queue = max(0, queue - served)
        total_served += served
        
        # 2. Ajouter les nouvelles arrivées
        queue += arrivals[t]
        total_arrived += arrivals[t]
        
        queue_length[t] = queue
        cumulative_arrivals[t] = total_arrived
        cumulative_served[t] = total_served
    
    # Calcul du temps d'attente estimé
    wait_times = np.where(total_capacity > 0, queue_length / total_capacity, 0)
    
    # ============================================================================
    # GRAPHIQUES
    # ============================================================================
    
    # Graphique principal: File + Arrivées vs Capacité
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Évolution de la file d\'attente - {selected_persona.name}',
            'Taux d\'arrivée vs Capacité de traitement'
        ),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # File d'attente
    fig.add_trace(go.Scatter(
        x=times, y=queue_length,
        mode='lines',
        name='Jobs en file',
        line=dict(width=3, color='#667eea'),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.3)'
    ), row=1, col=1)
    
    # Arrivées
    fig.add_trace(go.Scatter(
        x=times, y=arrivals,
        mode='lines',
        name='Arrivées (jobs/min)',
        line=dict(width=2, color='#ff6b6b')
    ), row=2, col=1)
    
    # Capacité (ligne horizontale)
    fig.add_trace(go.Scatter(
        x=times, y=[total_capacity] * time_steps,
        mode='lines',
        name=f'Capacité ({total_capacity:.1f} jobs/min)',
        line=dict(width=3, color='#00ff88', dash='dash')
    ), row=2, col=1)
    
    fig.update_layout(
        height=600,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    fig.update_xaxes(title_text='Temps (minutes)', row=2, col=1)
    fig.update_yaxes(title_text='Jobs en attente', row=1, col=1)
    fig.update_yaxes(title_text='Jobs/min', row=2, col=1)
    
    st.plotly_chart(apply_dark_theme(fig), use_container_width=True)
    
    # ============================================================================
    # MÉTRIQUES
    # ============================================================================
    st.markdown("### 📋 Métriques")
    
    avg_arrival = np.mean(arrivals)
    total_arrivals_sum = np.sum(arrivals)
    utilization = (avg_arrival / total_capacity * 100) if total_capacity > 0 else 0
    max_queue = max(queue_length)
    final_queue = queue_length[-1]
    avg_wait = np.mean(wait_times)
    
    # Temps pour vider la file (pour Prépa)
    time_to_empty = None
    for t in range(time_steps):
        if queue_length[t] < 0.5 and t > 15:  # File presque vide après le burst
            time_to_empty = t
            break
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total arrivées", f"{total_arrivals_sum:.0f} jobs")
    with col2:
        st.metric("Capacité", f"{total_capacity:.1f} j/min")
    with col3:
        st.metric("File max", f"{max_queue:.0f}")
    with col4:
        if time_to_empty:
            st.metric("Temps vidage", f"{time_to_empty} min")
        else:
            st.metric("Attente moy", f"{avg_wait:.1f} min")
    with col5:
        if final_queue < 1:
            st.metric("File finale", f"{final_queue:.0f}", delta="Stable ✅")
        else:
            st.metric("File finale", f"{final_queue:.0f}", delta="Accumulation ⚠️", delta_color="inverse")
    
    # Analyse de stabilité
    st.markdown("---")
    if selected_persona.student_type == StudentType.PREPA:
        if time_to_empty and time_to_empty < time_steps:
            st.success(f"✅ **File vidée en {time_to_empty} minutes** avec {num_servers} serveurs.")
            # Calculer le nombre minimal de serveurs
            min_servers_60min = math.ceil(total_arrivals_sum / (60 * service_rate_per_server))
            st.info(f"💡 Pour vider en ~60 min, il faut minimum {min_servers_60min} serveurs.")
        else:
            st.warning(f"⚠️ La file n'est pas vidée après {simulation_duration}h. Augmentez les serveurs.")
    else:
        if utilization < 100:
            st.success(f"✅ **Système stable** (ρ = {utilization:.0f}% < 100%)")
        else:
            servers_needed = math.ceil(avg_arrival / service_rate_per_server) + 1
            st.error(f"❌ **Système instable** (ρ = {utilization:.0f}% ≥ 100%). Minimum {servers_needed} serveurs requis.")
    
    # ============================================================================
    # ANALYSE COMPARATIVE: FILES SÉPARÉES
    # ============================================================================
    st.divider()
    st.subheader("🔀 Analyse: Files Séparées (Channels)")
    st.markdown("*Chaque population a sa propre file avec serveurs dédiés*")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        servers_prepa = st.number_input("Serveurs Prépa", 1, 50, max(1, num_servers // 2), key="sep_prepa")
    with col2:
        servers_ing = st.number_input("Serveurs Ingénieur", 1, 50, max(1, num_servers // 3), key="sep_ing")
    with col3:
        servers_admin = st.number_input("Serveurs Admin", 1, 50, max(1, num_servers // 6), key="sep_admin")
    
    # Simuler les files séparées
    separate_results = {}
    for persona in personas.values():
        if persona.student_type == StudentType.PREPA:
            servers = servers_prepa
            # Burst pour prépa
            arr = np.zeros(time_steps)
            burst_dur = 15
            arr_per_min = persona.population_size / burst_dur
            for t in range(min(burst_dur, time_steps)):
                arr[t] = arr_per_min
        elif persona.student_type == StudentType.INGENIEUR:
            servers = servers_ing
            arr = np.zeros(time_steps)
            np.random.seed(42)
            for t in range(time_steps):
                current_hour = (start_hour + t // 60) % 24
                base_rate = persona.get_arrival_rate(current_hour) / 60.0
                noise_factor = np.random.uniform(0.6, 1.4)  # Variabilité ±40%
                arr[t] = base_rate * noise_factor
        else:
            servers = servers_admin
            arr = np.zeros(time_steps)
            np.random.seed(42)
            for t in range(time_steps):
                current_hour = (start_hour + t // 60) % 24
                base_rate = persona.get_arrival_rate(current_hour) / 60.0
                noise_factor = np.random.uniform(0.8, 1.2)  # Variabilité ±20%
                arr[t] = base_rate * noise_factor
        
        cap = servers * (1.0 / persona.avg_test_complexity)
        q_len = np.zeros(time_steps)
        queue = 0.0
        
        for t in range(time_steps):
            served = min(queue, cap)
            queue = max(0, queue - served)
            queue += arr[t]
            q_len[t] = queue
        
        separate_results[persona.name] = {
            'queue': q_len,
            'arrivals': arr,
            'capacity': cap,
            'servers': servers
        }
    
    # Graphique files séparées
    fig = go.Figure()
    colors = ['#667eea', '#f093fb', '#00ff88']
    for i, (name, data) in enumerate(separate_results.items()):
        fig.add_trace(go.Scatter(
            x=times, y=data['queue'],
            mode='lines',
            name=f'{name} ({data["servers"]} serv)',
            line=dict(width=2.5, color=colors[i % len(colors)])
        ))
    
    fig.update_layout(
        title='Évolution des files séparées',
        xaxis_title='Temps (minutes)',
        yaxis_title='Jobs en attente',
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(apply_dark_theme(fig), use_container_width=True)
    
    # Tableau métriques files séparées
    sep_data = []
    for name, data in separate_results.items():
        sep_data.append({
            'Population': name,
            'Serveurs': data['servers'],
            'Capacité': f"{data['capacity']:.1f} j/min",
            'File max': f"{max(data['queue']):.0f}",
            'File finale': f"{data['queue'][-1]:.0f}"
        })
    st.dataframe(pd.DataFrame(sep_data), use_container_width=True, hide_index=True)
    
    # ============================================================================
    # ANALYSE COMPARATIVE: FILE AVEC PRIORITÉS
    # ============================================================================
    st.divider()
    st.subheader("⭐ Analyse: File avec Priorités")
    st.markdown("*File unique avec traitement prioritaire*")
    
    priority_order = ["Ingénieur", "Admin/Assistants", "Prépa (SUP/SPE)"]
    st.info(f"**Ordre de priorité:** 1. {priority_order[0]} (priorité max) → 2. {priority_order[1]} → 3. {priority_order[2]} (priorité min)")
    
    total_servers_prio = st.slider("Nombre total de serveurs (priorités)", 1, 50, num_servers, key="prio_servers")
    
    # Générer les arrivées pour chaque persona
    prio_arrivals = {}
    np.random.seed(42)
    for persona in personas.values():
        if persona.student_type == StudentType.PREPA:
            arr = np.zeros(time_steps)
            burst_dur = 15
            arr_per_min = persona.population_size / burst_dur
            for t in range(min(burst_dur, time_steps)):
                arr[t] = arr_per_min
        else:
            arr = np.zeros(time_steps)
            for t in range(time_steps):
                current_hour = (start_hour + t // 60) % 24
                base_rate = persona.get_arrival_rate(current_hour) / 60.0
                # Variabilité selon le type
                if persona.student_type == StudentType.INGENIEUR:
                    noise_factor = np.random.uniform(0.6, 1.4)
                else:
                    noise_factor = np.random.uniform(0.8, 1.2)
                arr[t] = base_rate * noise_factor
        prio_arrivals[persona.name] = arr
    
    # Capacité moyenne pondérée
    total_pop = sum(p.population_size for p in personas.values())
    weighted_service = sum((p.population_size / total_pop) * (1.0 / p.avg_test_complexity) for p in personas.values())
    total_capacity_prio = total_servers_prio * weighted_service
    
    # Simulation avec priorités
    prio_queues = {name: 0.0 for name in prio_arrivals}
    prio_queue_lengths = {name: np.zeros(time_steps) for name in prio_arrivals}
    
    for t in range(time_steps):
        # Traiter avec priorités
        remaining_cap = total_capacity_prio
        for prio_name in priority_order:
            if remaining_cap <= 0:
                break
            if prio_name in prio_queues:
                served = min(prio_queues[prio_name], remaining_cap)
                prio_queues[prio_name] = max(0, prio_queues[prio_name] - served)
                remaining_cap -= served
        
        # Ajouter les arrivées
        for name in prio_arrivals:
            prio_queues[name] += prio_arrivals[name][t]
            prio_queue_lengths[name][t] = prio_queues[name]
    
    # Graphique files avec priorités
    fig = go.Figure()
    for i, name in enumerate(priority_order):
        if name in prio_queue_lengths:
            fig.add_trace(go.Scatter(
                x=times, y=prio_queue_lengths[name],
                mode='lines',
                name=f'#{i+1} {name}',
                line=dict(width=2.5, color=colors[i % len(colors)])
            ))
    
    fig.update_layout(
        title=f'Évolution des files avec priorités ({total_servers_prio} serveurs)',
        xaxis_title='Temps (minutes)',
        yaxis_title='Jobs en attente',
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(apply_dark_theme(fig), use_container_width=True)
    
    # Tableau métriques priorités
    prio_data = []
    for i, name in enumerate(priority_order):
        if name in prio_queue_lengths:
            prio_data.append({
                'Priorité': f"#{i+1}",
                'Population': name,
                'File max': f"{max(prio_queue_lengths[name]):.0f}",
                'File finale': f"{prio_queue_lengths[name][-1]:.0f}"
            })
    st.dataframe(pd.DataFrame(prio_data), use_container_width=True, hide_index=True)


def generate_arrivals(personas, times, start_hour, enable_burst, burst_time, burst_duration, burst_percentage):
    """Génère les arrivées pour chaque persona."""
    arrivals = {}
    
    for student_type, persona in personas.items():
        arrivals[persona.name] = np.zeros(len(times))
        
        for t in times:
            current_hour = (start_hour + t // 60) % 24
            base_rate = persona.get_arrival_rate(current_hour) / 60.0  # jobs/min
            
            # Ajouter le burst pour les Prépa
            if enable_burst and persona.student_type == StudentType.PREPA:
                if burst_time <= t < burst_time + burst_duration:
                    burst_arrivals = (persona.population_size * burst_percentage / 100) / burst_duration
                    arrivals[persona.name][t] = burst_arrivals
                else:
                    arrivals[persona.name][t] = base_rate
            else:
                arrivals[persona.name][t] = base_rate
    
    return arrivals


def simulate_single_queue(personas, arrivals, times, total_servers):
    """Simule une file unique FIFO avec traitement correct."""
    
    # Taux de service: utiliser un taux moyen pondéré par le volume d'arrivées attendu
    total_arrival_rate = sum(np.mean(arrivals[p.name]) for p in personas.values())
    if total_arrival_rate > 0:
        # Pondérer par la contribution de chaque persona aux arrivées
        weighted_service = 0
        for persona in personas.values():
            arrival_contrib = np.mean(arrivals[persona.name]) / total_arrival_rate
            service_rate_persona = 1.0 / persona.avg_test_complexity  # jobs/min/serveur
            weighted_service += arrival_contrib * service_rate_persona
    else:
        weighted_service = 1.0
    
    total_service_rate = total_servers * weighted_service  # jobs/min totale
    
    # File unique
    queue_length = np.zeros(len(times))
    wait_times = np.zeros(len(times))
    in_service = np.zeros(len(times))
    total_arrivals = np.zeros(len(times))
    
    for name in arrivals:
        total_arrivals += arrivals[name]
    
    queue = 0.0  # Jobs en attente (pas en service)
    
    for t in times:
        # 1. D'abord traiter les jobs (servir)
        served = min(queue, total_service_rate)
        queue = max(0, queue - served)
        
        # 2. Ensuite ajouter les nouvelles arrivées
        queue += total_arrivals[t]
        
        queue_length[t] = queue
        # Temps d'attente estimé = jobs en file / capacité de service
        wait_times[t] = queue / total_service_rate if total_service_rate > 0 else 0
        in_service[t] = min(served, total_servers)
    
    return {
        'queue_lengths': {'Total': queue_length},
        'wait_times': {'Total': wait_times},
        'arrivals': {'Total': total_arrivals},
        'in_service': {'Total': in_service},
        'service_rate': total_service_rate,
        'service_rates': {'Total': total_service_rate}
    }


def simulate_separate_queues(personas, arrivals, times, config):
    """Simule des files séparées avec traitement correct."""
    
    queue_lengths = {}
    wait_times = {}
    service_rates = {}
    in_service = {}
    
    for student_type, persona in personas.items():
        # Déterminer le nombre de serveurs
        if persona.student_type == StudentType.PREPA:
            servers = config['servers_prepa']
        elif persona.student_type == StudentType.INGENIEUR:
            servers = config['servers_ing']
        else:
            servers = config['servers_admin']
        
        service_rate_per_server = 1.0 / persona.avg_test_complexity  # jobs/min/serveur
        service_rate = service_rate_per_server * servers  # capacité totale jobs/min
        service_rates[persona.name] = service_rate
        
        queue = 0.0
        q_lengths = np.zeros(len(times))
        w_times = np.zeros(len(times))
        in_serv = np.zeros(len(times))
        
        for t in times:
            # 1. D'abord traiter (servir les jobs)
            served = min(queue, service_rate)
            queue = max(0, queue - served)
            
            # 2. Ensuite ajouter les nouvelles arrivées
            queue += arrivals[persona.name][t]
            
            q_lengths[t] = queue
            w_times[t] = queue / service_rate if service_rate > 0 else 0
            in_serv[t] = served
        
        queue_lengths[persona.name] = q_lengths
        wait_times[persona.name] = w_times
        in_service[persona.name] = in_serv
    
    return {
        'queue_lengths': queue_lengths,
        'wait_times': wait_times,
        'arrivals': arrivals,
        'in_service': in_service,
        'separate': True,
        'service_rates': service_rates
    }


def simulate_priority_queue(personas, arrivals, times, config):
    """Simule une file avec priorités - traitement correct."""
    
    priority_order = config['priority_order']
    total_servers = config['servers']
    
    # Créer un mapping persona -> priorité (0 = plus haute)
    priority_map = {name: i for i, name in enumerate(priority_order)}
    
    # Taux de service par persona (jobs/min/serveur)
    service_rates = {
        persona.name: 1.0 / persona.avg_test_complexity
        for persona in personas.values()
    }
    
    # Capacité totale (on utilise un taux moyen pour simplifier)
    avg_service_rate = np.mean(list(service_rates.values()))
    total_capacity = total_servers * avg_service_rate  # jobs/min
    
    # Files par persona (valeurs flottantes pour cohérence)
    queues = {name: 0.0 for name in arrivals}
    queue_lengths = {name: np.zeros(len(times)) for name in arrivals}
    wait_times = {name: np.zeros(len(times)) for name in arrivals}
    in_service = {name: np.zeros(len(times)) for name in arrivals}
    
    for t in times:
        # 1. D'abord traiter avec priorités
        remaining_capacity = total_capacity
        
        # Traiter par ordre de priorité
        for priority_name in priority_order:
            if remaining_capacity <= 0:
                break
            if priority_name in queues:
                served = min(queues[priority_name], remaining_capacity)
                queues[priority_name] = max(0, queues[priority_name] - served)
                remaining_capacity -= served
                in_service[priority_name][t] = served
        
        # 2. Ensuite ajouter les nouvelles arrivées
        for name in arrivals:
            queues[name] += arrivals[name][t]
        
        # Enregistrer les métriques
        for name in arrivals:
            queue_lengths[name][t] = queues[name]
            # Temps d'attente estimé
            wait_times[name][t] = queues[name] / (total_capacity / len(arrivals)) if total_capacity > 0 else 0
    
    return {
        'queue_lengths': queue_lengths,
        'wait_times': wait_times,
        'arrivals': arrivals,
        'in_service': in_service,
        'priority_order': priority_order,
        'service_rates': {name: total_capacity / len(arrivals) for name in arrivals}
    }
    
    return {
        'queue_lengths': queue_lengths,
        'wait_times': wait_times,
        'arrivals': arrivals,
        'priority_order': priority_order
    }


def display_simulation_results(results, times, personas, strategy_config):
    """Affiche les résultats de simulation avec graphiques améliorés."""
    
    # Graphique principal: Évolution des files avec zone d'arrivée vs capacité
    st.markdown("### 📊 Évolution des files d'attente")
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Jobs en attente dans le temps', 'Arrivées vs Capacité de traitement'),
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4]
    )
    
    # Couleurs hex pour les graphiques
    colors_hex = ['#667eea', '#f093fb', '#00ff88', '#ff6b6b', '#feca57', '#48dbfb']
    colors_rgba = [
        'rgba(102, 126, 234, 0.2)',
        'rgba(240, 147, 251, 0.2)',
        'rgba(0, 255, 136, 0.2)',
        'rgba(255, 107, 107, 0.2)',
        'rgba(254, 202, 87, 0.2)',
        'rgba(72, 219, 251, 0.2)'
    ]
    
    # Graphique du haut: Longueur des files
    for i, (name, lengths) in enumerate(results['queue_lengths'].items()):
        fig.add_trace(go.Scatter(
            x=times, y=lengths,
            mode='lines',
            name=f"📦 {name}",
            line=dict(width=2.5, color=colors_hex[i % len(colors_hex)]),
            fill='tozeroy',
            fillcolor=colors_rgba[i % len(colors_rgba)]
        ), row=1, col=1)
    
    # Graphique du bas: Arrivées vs Capacité
    total_arrivals = np.zeros(len(times))
    for name, arr in results['arrivals'].items():
        total_arrivals += arr
        fig.add_trace(go.Scatter(
            x=times, y=arr,
            mode='lines',
            name=f"→ Arrivées {name}",
            line=dict(width=1.5, dash='dot'),
            showlegend=True
        ), row=2, col=1)
    
    # Ajouter la ligne de capacité
    if 'service_rates' in results:
        total_capacity = sum(results['service_rates'].values())
        fig.add_trace(go.Scatter(
            x=times, y=[total_capacity] * len(times),
            mode='lines',
            name=f"⚡ Capacité totale ({total_capacity:.1f} jobs/min)",
            line=dict(width=3, color='#00ff88', dash='solid')
        ), row=2, col=1)
    elif 'service_rate' in results:
        fig.add_trace(go.Scatter(
            x=times, y=[results['service_rate']] * len(times),
            mode='lines',
            name=f"⚡ Capacité ({results['service_rate']:.1f} jobs/min)",
            line=dict(width=3, color='#00ff88', dash='solid')
        ), row=2, col=1)
    
    # Total des arrivées
    fig.add_trace(go.Scatter(
        x=times, y=total_arrivals,
        mode='lines',
        name='📈 Total arrivées',
        line=dict(width=2.5, color='#ff6b6b')
    ), row=2, col=1)
    
    fig.update_layout(
        height=700,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    fig.update_xaxes(title_text='Temps (minutes)', row=2, col=1)
    fig.update_yaxes(title_text='Jobs en attente', row=1, col=1)
    fig.update_yaxes(title_text='Jobs/min', row=2, col=1)
    
    st.plotly_chart(apply_dark_theme(fig), use_container_width=True)
    
    # Graphique séparé: Temps d'attente
    st.markdown("### ⏱️ Temps d'attente estimé")
    fig = go.Figure()
    for i, (name, waits) in enumerate(results['wait_times'].items()):
        fig.add_trace(go.Scatter(
            x=times, y=waits,
            mode='lines',
            name=name,
            line=dict(width=2.5, color=colors_hex[i % len(colors_hex)])
        ))
    fig.update_layout(
        xaxis_title='Temps (minutes)',
        yaxis_title='Temps d\'attente estimé (min)',
        height=350,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    st.plotly_chart(apply_dark_theme(fig), use_container_width=True)
    
    # Métriques récapitulatives
    st.markdown("### 📋 Métriques Récapitulatives")
    
    metrics_data = []
    for name in results['queue_lengths']:
        max_queue = max(results['queue_lengths'][name])
        avg_queue = np.mean(results['queue_lengths'][name])
        final_queue = results['queue_lengths'][name][-1]
        max_wait = max(results['wait_times'][name])
        avg_wait = np.mean(results['wait_times'][name])
        
        # Calculer taux d'arrivée moyen
        avg_arrival = np.mean(results['arrivals'][name])
        
        # Capacité de traitement (si disponible)
        capacity_info = ""
        utilization = 0
        if 'service_rates' in results and name in results['service_rates']:
            capacity = results['service_rates'][name]
            utilization = (avg_arrival / capacity * 100) if capacity > 0 else 0
            capacity_info = f"{capacity:.1f} jobs/min"
        
        status = "✅ OK" if final_queue < 5 else ("⚠️ File" if final_queue < 50 else "❌ Saturé")
        
        metrics_data.append({
            'Population': name,
            'Arrivée moy': f"{avg_arrival:.2f} j/min",
            'Capacité': capacity_info if capacity_info else 'N/A',
            'ρ (charge)': f"{utilization:.0f}%" if capacity_info else 'N/A',
            'File max': f"{max_queue:.0f}",
            'File finale': f"{final_queue:.0f}",
            'Attente moy': f"{avg_wait:.1f} min",
            'Statut': status
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
    
    # Résumé global
    st.markdown("### 🎯 Résumé Global")
    
    if 'service_rates' in results:
        total_capacity = sum(results['service_rates'].values())
    elif 'service_rate' in results:
        total_capacity = results['service_rate']
    else:
        total_capacity = 0
    
    total_arrival = sum(np.mean(results['arrivals'][name]) for name in results['arrivals'])
    total_max_queue = sum(max(results['queue_lengths'][name]) for name in results['queue_lengths'])
    total_final_queue = sum(results['queue_lengths'][name][-1] for name in results['queue_lengths'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_color = "normal" if total_capacity > total_arrival else "inverse"
        st.metric("Capacité totale", f"{total_capacity:.1f} jobs/min")
    with col2:
        st.metric("Charge moyenne", f"{total_arrival:.1f} jobs/min")
    with col3:
        utilization = (total_arrival / total_capacity * 100) if total_capacity > 0 else 0
        st.metric("Utilisation ρ", f"{utilization:.0f}%")
    with col4:
        if total_final_queue < 5:
            st.metric("File finale", f"{total_final_queue:.0f}", delta="Stable ✅")
        else:
            st.metric("File finale", f"{total_final_queue:.0f}", delta="Accumulation ⚠️", delta_color="inverse")


def run_all_strategies_comparison(personas, total_servers, duration_hours, start_hour,
                                  enable_burst, burst_time, burst_duration, burst_percentage,
                                  separate_config):
    """Compare toutes les stratégies en une seule fois - affichage vertical."""
    
    st.markdown("---")
    st.subheader("🔄 Comparaison de Toutes les Stratégies")
    
    with st.spinner("Simulation en cours pour toutes les stratégies..."):
        # Générer les arrivées une seule fois
        time_steps = duration_hours * 60
        times = np.arange(time_steps)
        arrivals = generate_arrivals(personas, times, start_hour, enable_burst,
                                    burst_time, burst_duration, burst_percentage)
        
        # Configs par défaut si pas de separate_config
        if separate_config is None:
            separate_config = {
                'type': 'separate',
                'servers_prepa': max(1, int(total_servers * 0.5)),
                'servers_ing': max(1, int(total_servers * 0.4)),
                'servers_admin': max(1, int(total_servers * 0.1))
            }
        
        # Simuler chaque stratégie
        results_single = simulate_single_queue(personas, arrivals, times, total_servers)
        results_separate = simulate_separate_queues(personas, arrivals, times, separate_config)
        results_priority = simulate_priority_queue(personas, arrivals, times, {
            'type': 'priority',
            'servers': total_servers,
            'priority_order': ["Admin/Assistants", "Ingénieur", "Prépa (SUP/SPE)"]
        })
        
        # ========================================================================
        # AFFICHAGE VERTICAL - Chaque stratégie l'une après l'autre
        # ========================================================================
        
        # Tableau récapitulatif en haut
        st.markdown("### 📊 Tableau Comparatif Rapide")
        
        comparison_data = []
        
        # File unique
        total_max_queue_single = max(results_single['queue_lengths']['Total'])
        total_final_queue_single = results_single['queue_lengths']['Total'][-1]
        total_avg_wait_single = np.mean(results_single['wait_times']['Total'])
        comparison_data.append({
            'Stratégie': '📦 File Unique',
            'File max': f"{total_max_queue_single:.0f}",
            'File finale': f"{total_final_queue_single:.0f}",
            'Attente moy': f"{total_avg_wait_single:.2f} min",
            'Équité': 'FIFO strict'
        })
        
        # Files séparées
        sep_max = max(max(v) for v in results_separate['queue_lengths'].values())
        sep_final = sum(v[-1] for v in results_separate['queue_lengths'].values())
        sep_avg = np.mean([np.mean(v) for v in results_separate['wait_times'].values()])
        comparison_data.append({
            'Stratégie': '🔀 Files Séparées',
            'File max': f"{sep_max:.0f}",
            'File finale': f"{sep_final:.0f}",
            'Attente moy': f"{sep_avg:.2f} min",
            'Équité': 'Isolation par groupe'
        })
        
        # Priorités
        prio_max = max(max(v) for v in results_priority['queue_lengths'].values())
        prio_final = sum(v[-1] for v in results_priority['queue_lengths'].values())
        prio_avg = np.mean([np.mean(v) for v in results_priority['wait_times'].values()])
        comparison_data.append({
            'Stratégie': '⭐ File avec Priorités',
            'File max': f"{prio_max:.0f}",
            'File finale': f"{prio_final:.0f}",
            'Attente moy': f"{prio_avg:.2f} min",
            'Équité': 'Favorise prioritaires'
        })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # ========================================================================
        # STRATÉGIE 1: File Unique
        # ========================================================================
        st.markdown("## 📦 Stratégie 1: File Unique (FIFO)")
        st.markdown("*Tous les serveurs traitent tous les jobs sans distinction*")
        
        # Graphique évolution de la file
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Évolution de la file', 'Arrivées vs Capacité'),
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4]
        )
        
        fig.add_trace(go.Scatter(
            x=times, y=results_single['queue_lengths']['Total'],
            mode='lines',
            name='Jobs en file',
            line=dict(width=3, color='#667eea'),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.3)'
        ), row=1, col=1)
        
        # Arrivées et capacité
        fig.add_trace(go.Scatter(
            x=times, y=results_single['arrivals']['Total'],
            mode='lines',
            name='Arrivées',
            line=dict(width=2, color='#ff6b6b')
        ), row=2, col=1)
        
        capacity = results_single['service_rate']
        fig.add_trace(go.Scatter(
            x=times, y=[capacity] * len(times),
            mode='lines',
            name=f'Capacité ({capacity:.1f} j/min)',
            line=dict(width=3, color='#00ff88', dash='dash')
        ), row=2, col=1)
        
        fig.update_layout(height=500, hovermode='x unified')
        fig.update_xaxes(title_text='Temps (min)', row=2, col=1)
        fig.update_yaxes(title_text='Jobs', row=1, col=1)
        fig.update_yaxes(title_text='Jobs/min', row=2, col=1)
        st.plotly_chart(apply_dark_theme(fig), use_container_width=True)
        
        # Métriques
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("File max", f"{total_max_queue_single:.0f}")
        col2.metric("File finale", f"{total_final_queue_single:.0f}")
        col3.metric("Attente moy", f"{total_avg_wait_single:.2f} min")
        col4.metric("Capacité", f"{capacity:.1f} j/min")
        
        st.divider()
        
        # ========================================================================
        # STRATÉGIE 2: Files Séparées
        # ========================================================================
        st.markdown("## 🔀 Stratégie 2: Files Séparées (Channels)")
        st.markdown("*Chaque population a sa propre file avec serveurs dédiés*")
        
        # Graphique avec toutes les files
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Évolution des files par population', 'Arrivées par population'),
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4]
        )
        
        colors = ['#667eea', '#f093fb', '#00ff88']
        for i, (name, lengths) in enumerate(results_separate['queue_lengths'].items()):
            fig.add_trace(go.Scatter(
                x=times, y=lengths,
                mode='lines',
                name=f'{name}',
                line=dict(width=2.5, color=colors[i % len(colors)])
            ), row=1, col=1)
        
        for i, (name, arr) in enumerate(results_separate['arrivals'].items()):
            fig.add_trace(go.Scatter(
                x=times, y=arr,
                mode='lines',
                name=f'Arrivées {name}',
                line=dict(width=1.5, color=colors[i % len(colors)], dash='dot'),
                showlegend=False
            ), row=2, col=1)
        
        fig.update_layout(height=500, hovermode='x unified')
        fig.update_xaxes(title_text='Temps (min)', row=2, col=1)
        fig.update_yaxes(title_text='Jobs en file', row=1, col=1)
        fig.update_yaxes(title_text='Jobs/min', row=2, col=1)
        st.plotly_chart(apply_dark_theme(fig), use_container_width=True)
        
        # Info sur allocation
        st.info(f"**Allocation:** Prépa: {separate_config['servers_prepa']} serveurs | "
                f"Ingénieur: {separate_config['servers_ing']} serveurs | "
                f"Admin: {separate_config['servers_admin']} serveurs")
        
        # Tableau détaillé par population
        sep_metrics = []
        for name in results_separate['queue_lengths']:
            max_q = max(results_separate['queue_lengths'][name])
            final_q = results_separate['queue_lengths'][name][-1]
            avg_wait = np.mean(results_separate['wait_times'][name])
            cap = results_separate['service_rates'][name]
            arr = np.mean(results_separate['arrivals'][name])
            rho = (arr / cap * 100) if cap > 0 else 0
            sep_metrics.append({
                'Population': name,
                'Capacité': f"{cap:.1f} j/min",
                'Arrivée moy': f"{arr:.2f} j/min",
                'ρ': f"{rho:.0f}%",
                'File max': f"{max_q:.0f}",
                'File finale': f"{final_q:.0f}",
                'Attente moy': f"{avg_wait:.2f} min"
            })
        st.dataframe(pd.DataFrame(sep_metrics), use_container_width=True, hide_index=True)
        
        st.divider()
        
        # ========================================================================
        # STRATÉGIE 3: File avec Priorités
        # ========================================================================
        st.markdown("## ⭐ Stratégie 3: File avec Priorités")
        st.markdown("*File unique avec traitement prioritaire (Admin > Ingénieur > Prépa)*")
        
        # Graphique
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Évolution des files par population', 'Arrivées par population'),
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4]
        )
        
        for i, (name, lengths) in enumerate(results_priority['queue_lengths'].items()):
            fig.add_trace(go.Scatter(
                x=times, y=lengths,
                mode='lines',
                name=f'{name}',
                line=dict(width=2.5, color=colors[i % len(colors)])
            ), row=1, col=1)
        
        for i, (name, arr) in enumerate(results_priority['arrivals'].items()):
            fig.add_trace(go.Scatter(
                x=times, y=arr,
                mode='lines',
                name=f'Arrivées {name}',
                line=dict(width=1.5, color=colors[i % len(colors)], dash='dot'),
                showlegend=False
            ), row=2, col=1)
        
        fig.update_layout(height=500, hovermode='x unified')
        fig.update_xaxes(title_text='Temps (min)', row=2, col=1)
        fig.update_yaxes(title_text='Jobs en file', row=1, col=1)
        fig.update_yaxes(title_text='Jobs/min', row=2, col=1)
        st.plotly_chart(apply_dark_theme(fig), use_container_width=True)
        
        # Tableau détaillé
        prio_metrics = []
        for name in results_priority['queue_lengths']:
            max_q = max(results_priority['queue_lengths'][name])
            final_q = results_priority['queue_lengths'][name][-1]
            avg_wait = np.mean(results_priority['wait_times'][name])
            priority_rank = results_priority['priority_order'].index(name) + 1 if name in results_priority['priority_order'] else '-'
            prio_metrics.append({
                'Population': name,
                'Priorité': f"#{priority_rank}",
                'File max': f"{max_q:.0f}",
                'File finale': f"{final_q:.0f}",
                'Attente moy': f"{avg_wait:.2f} min"
            })
        st.dataframe(pd.DataFrame(prio_metrics), use_container_width=True, hide_index=True)
        
        st.divider()
        
        # ========================================================================
        # GRAPHIQUE COMPARATIF FINAL
        # ========================================================================
        st.markdown("### 📈 Comparaison des Temps d'Attente par Population")
        
        fig = go.Figure()
        
        strategies = ['File Unique', 'Files Séparées', 'Priorités']
        
        for persona in personas.values():
            name = persona.name
            
            # File unique - moyenne globale pour tous
            wait_single = np.mean(results_single['wait_times']['Total'])
            
            # Files séparées
            wait_sep = np.mean(results_separate['wait_times'][name])
            
            # Priorités
            wait_prio = np.mean(results_priority['wait_times'][name])
            
            fig.add_trace(go.Bar(
                name=name,
                x=strategies,
                y=[wait_single, wait_sep, wait_prio]
            ))
        
        fig.update_layout(
            barmode='group',
            yaxis_title='Temps d\'attente moyen (min)',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        st.plotly_chart(apply_dark_theme(fig), use_container_width=True)
        
        # Recommandation
        st.markdown("### 💡 Recommandation")
        
        if prio_avg < sep_avg and prio_avg < total_avg_wait_single:
            st.success("✅ **File avec Priorités** offre les meilleures performances globales tout en favorisant les populations critiques.")
        elif sep_avg < total_avg_wait_single * 0.9:
            st.success("✅ **Files Séparées** offre une meilleure isolation et prévisibilité pour chaque population.")
        else:
            st.info("ℹ️ **File Unique** est suffisante avec les paramètres actuels, mais moins résiliente aux pics.")


def run_theoretical_analysis(personas, total_servers, strategy_config):
    """Analyse théorique sans simulation (M/M/c)."""
    
    st.markdown("---")
    st.subheader("Analyse Théorique (Modèle M/M/c)")
    
    st.info("Analyse basée sur les taux moyens (14h) sans burst")
    
    if strategy_config['type'] == 'single':
        analyze_single_queue_theoretical(personas, total_servers)
    elif strategy_config['type'] == 'separate':
        analyze_separate_queues_theoretical(personas, strategy_config)
    else:
        st.warning("Analyse théorique non disponible pour les files avec priorités (nécessite simulation)")


def analyze_single_queue_theoretical(personas, total_servers):
    """Analyse théorique file unique."""
    
    total_arrival = sum(p.get_arrival_rate(14) for p in personas.values()) / 60  # jobs/min
    total_pop = sum(p.population_size for p in personas.values())
    weighted_service = sum((p.population_size / total_pop) * (1.0 / p.avg_test_complexity) 
                          for p in personas.values())
    
    rho = total_arrival / (total_servers * weighted_service)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Taux d'arrivée λ", f"{total_arrival:.2f} jobs/min")
        st.metric("Taux de service μ", f"{weighted_service:.2f} jobs/min/serveur")
    
    with col2:
        st.metric("Charge ρ", f"{rho:.2%}")
        st.metric("Serveurs", total_servers)
    
    if rho < 1:
        try:
            queue = GenericQueue(total_arrival, weighted_service, f"M/M/{total_servers}", c=total_servers)
            metrics = queue.compute_theoretical_metrics()
            
            st.success("Système stable")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("L (jobs dans système)", f"{metrics.L:.2f}")
            with col2:
                st.metric("Lq (jobs en attente)", f"{metrics.Lq:.2f}")
            with col3:
                st.metric("W (temps total, min)", f"{metrics.W:.2f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Wq (temps attente, min)", f"{metrics.Wq:.2f}")
            with col2:
                st.metric("P0 (proba vide)", f"{metrics.P0:.2%}")
        except Exception as e:
            st.error(f"Erreur de calcul: {e}")
    else:
        st.error("Système instable (ρ ≥ 1)")


def analyze_separate_queues_theoretical(personas, config):
    """Analyse théorique files séparées."""
    
    results_data = []
    
    for student_type, persona in personas.items():
        if persona.student_type == StudentType.PREPA:
            servers = config['servers_prepa']
        elif persona.student_type == StudentType.INGENIEUR:
            servers = config['servers_ing']
        else:
            servers = config['servers_admin']
        
        arrival_rate = persona.get_arrival_rate(14) / 60  # jobs/min
        service_rate = 1.0 / persona.avg_test_complexity
        
        rho = arrival_rate / (servers * service_rate) if servers > 0 else float('inf')
        
        if rho < 1 and servers > 0:
            try:
                queue = GenericQueue(arrival_rate, service_rate, f"M/M/{servers}", c=servers)
                metrics = queue.compute_theoretical_metrics()
                
                results_data.append({
                    'Population': persona.name,
                    'Serveurs': servers,
                    'λ': f"{arrival_rate:.2f}",
                    'μ': f"{service_rate:.2f}",
                    'ρ': f"{rho:.2%}",
                    'Wq (min)': f"{metrics.Wq:.2f}",
                    'Lq': f"{metrics.Lq:.2f}",
                    'Statut': 'OK'
                })
            except:
                results_data.append({
                    'Population': persona.name,
                    'Serveurs': servers,
                    'λ': f"{arrival_rate:.2f}",
                    'μ': f"{service_rate:.2f}",
                    'ρ': f"{rho:.2%}",
                    'Wq (min)': 'Erreur',
                    'Lq': 'Erreur',
                    'Statut': 'Instable'
                })
        else:
            results_data.append({
                'Population': persona.name,
                'Serveurs': servers,
                'λ': f"{arrival_rate:.2f}",
                'μ': f"{service_rate:.2f}",
                'ρ': '≥100%',
                'Wq (min)': '∞',
                'Lq': '∞',
                'Statut': 'Instable'
            })
    
    df = pd.DataFrame(results_data)
    st.dataframe(df, use_container_width=True)


# ==============================================================================
# OPTIMISATION COÛT / QUALITÉ DE SERVICE
# ==============================================================================

def render_optimization_tab(mu_rate: float, n_servers: int, buffer_size: int):
    """Onglet d'optimisation coût/performance."""
    st.header("Optimisation Coût / Qualité de Service")
    
    st.markdown("""
    <div class="formula-box">
    <strong>Fonction objectif:</strong><br>
    min [ α × E[T] + β × Coût(K, c, μ) ] avec α + β = 1
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    - **E[T]** = Temps moyen de séjour (temps d'attente + service)
    - **Coût** = Coût serveurs (€/h)
    - **α** = Poids accordé à la performance (temps)
    - **β** = Poids accordé au coût
    """)
    
    st.divider()
    
    # Configuration
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Taux d'arrivée")
        
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
        st.subheader("Pondération")
        
        alpha = st.slider("α (poids de performance (temps))", 0.0, 1.0, 0.7, 0.05)
        st.write(f"β (poids de coût) = {1-alpha:.2f}")
        
        st.markdown("---")
        st.subheader("Modèle de coût")
        
        cost_server = st.number_input("Coût serveur (€/h)", 0.1, 10.0, 0.50, 0.1)
        st.info("Le coût total est calculé uniquement à partir du nombre de serveurs actifs.")
    
    st.divider()
    
    # Paramètres de recherche
    st.subheader("Espace de recherche")
    
    col1, col2, col3 = st.columns(3)    
    with col1:
        max_servers = st.slider("Max serveurs", 5, 30, 15)
    with col2:
        mu_min = st.number_input("μ min", 5.0, 20.0, 5.0)
        mu_max = st.number_input("μ max", 10.0, 50.0, 25.0)
    with col3:
        resolution = st.slider("Résolution", 10, 40, 20)
    
    if st.button("Lancer l'optimisation", type="primary"):
        run_optimization(lambda_rate, alpha, cost_server,
                        max_servers, mu_min, mu_max, resolution)


def run_optimization(lambda_rate, alpha, cost_server,
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
                        
                        # Coût horaire (uniquement serveurs)
                        total_cost = c * cost_server
                        
                        Z_cost[i, j] = total_cost
                        Z_time[i, j] = metrics.W
                        
                        # Score normalisé (alpha * coût + (1-alpha) * temps normalisé)
                        # On normalise le temps pour le mettre à l'échelle avec le coût
                        time_normalized = metrics.W * 10  # Facteur d'échelle arbitraire
                        Z_score[i, j] = alpha * total_cost + (1 - alpha) * time_normalized
                        
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
            st.markdown("### Heatmap Coût Total (€/h)")
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
            st.markdown("### Heatmap Temps de Séjour (min)")
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
        st.markdown("### Heatmap Score Combiné (α×Coût + β×Temps)")
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
        ### Configuration Optimale
        
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
    st.header("Stratégies d'Auto-Scaling")
    
    st.markdown("""
    <div class="scenario-box">
    <strong>Objectif:</strong> Adapter dynamiquement le nombre de serveurs à la charge
    </div>
    """, unsafe_allow_html=True)
    
    # Types de scaling
    scaling_type = st.radio(
        "Stratégie de scaling",
        ["Fixe", "Programmé", "Réactif", "Prédictif"],
        horizontal=True
    )
    
    st.divider()
    
    if scaling_type == "Fixe":
        st.markdown("""
        **Scaling Fixe:** Nombre constant de serveurs
        
        - Simple à gérer  
        - Pas d'adaptation à la charge
        """)
        
        st.metric("Serveurs fixes", n_servers)
    
    elif scaling_type == "Programmé":
        st.markdown("""
        **Scaling Programmé:** Nombre de serveurs selon l'heure
        
        - Adapté aux patterns connus  
        - Ne gère pas les pics imprévus
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
    
    elif scaling_type == "Réactif":
        st.markdown("""
        **Scaling Réactif:** Ajustement basé sur la charge actuelle
        
        - Réagit aux pics  
        - Temps de réaction (cooldown)
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
    
    elif scaling_type == "Prédictif":
        st.markdown("""
        **Scaling Prédictif:** Anticipation basée sur les données historiques
        
        - Évite les temps de réaction  
        - Nécessite des données historiques
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
    st.subheader("Comparaison des stratégies")
    
    comparison_data = {
        'Stratégie': ['Fixe', 'Programmé', 'Réactif', 'Prédictif'],
        'Réactivité': ['❌ Aucune', '⚠️ Limitée', '✅ Excellente', '✅ Bonne'],
        'Complexité': ['✅ Simple', '⚠️ Moyenne', '⚠️ Moyenne', '❌ Complexe'],
        'Coût': ['✅ Stable', '✅ Bon si bien planifié', '✅ Optimisé', '✅ Optimisé'],
        'Pics imprévus': ['❌ Non géré', '❌ Non géré', '✅ Géré', '⚠️ Non géré si mal entraîné']
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
            
            st.subheader("Comparaison théorique")
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
    st.subheader("Simulation Monte Carlo Comparative")
    
    col_s1, col_s2 = st.columns([1, 3])
    
    with col_s1:
        sim_n = st.number_input("Clients", 100, 10000, 2000, key="b_sim_n")
        sim_runs = st.number_input("Runs", 1, 20, 5, key="b_sim_runs")
        run_benchmark = st.button("Lancer benchmark", type="primary")
    
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
