"""
Configuration globale des serveurs de la moulinette.

Ce fichier centralise toutes les variables de configuration
pour les serveurs/runners de la moulinette EPITA.

Pour modifier la configuration par defaut, editez les valeurs
dans la classe ServerConfigDefaults ou creez une instance de
ServerConfig avec vos propres valeurs.

Auteurs: ERO2 Team Markov Moulinette Configurators - EPITA
"""

from dataclasses import dataclass, field
from typing import Optional


class ServerConfigDefaults:
    """
    Valeurs par defaut pour la configuration serveur.
    
    Modifiez ces valeurs pour changer les defaults globaux.
    """
    
    # ══════════════════════════════════════════════════════════════
    # INFRASTRUCTURE - Nombre de serveurs
    # ══════════════════════════════════════════════════════════════
    N_SERVERS: int = 4                    # Nombre de runners actifs
    MIN_SERVERS: int = 1                  # Minimum de serveurs
    MAX_SERVERS: int = 50                 # Maximum de serveurs
    
    # ══════════════════════════════════════════════════════════════
    # PERFORMANCE - Taux de service
    # ══════════════════════════════════════════════════════════════
    SERVICE_RATE: float = 6.0             # Corrections/heure/runner (mu)
    SERVICE_VARIANCE: float = 0.3         # Variance relative du temps de service
    AVG_TEST_DURATION_MINUTES: float = 10.0  # Duree moyenne d'un test
    TIMEOUT_MINUTES: float = 30.0         # Timeout maximum par test
    
    # ══════════════════════════════════════════════════════════════
    # CAPACITE - Buffer et limites
    # ══════════════════════════════════════════════════════════════
    BUFFER_SIZE: int = 100                # Taille max de la file d'attente (K)
    MAX_CONCURRENT_TESTS: int = 1         # Tests simultanes par runner
    
    # ══════════════════════════════════════════════════════════════
    # COUTS - Facturation
    # ══════════════════════════════════════════════════════════════
    COST_PER_SERVER_HOUR: float = 0.50    # EUR/heure/serveur
    COST_PER_REJECTION: float = 5.0       # EUR par soumission rejetee (cout indirect)
    COST_PER_WAITING_MINUTE: float = 0.10 # EUR par minute d'attente (cout indirect)
    STARTUP_COST: float = 0.05            # EUR par demarrage de serveur
    
    # ══════════════════════════════════════════════════════════════
    # TEMPS - Latences et delais
    # ══════════════════════════════════════════════════════════════
    STARTUP_TIME_SECONDS: float = 30.0    # Temps de demarrage d'un runner
    SHUTDOWN_TIME_SECONDS: float = 10.0   # Temps d'arret d'un runner
    COOLDOWN_SECONDS: float = 300.0       # Delai avant scale-down (5 min)
    
    # ══════════════════════════════════════════════════════════════
    # AUTO-SCALING - Seuils
    # ══════════════════════════════════════════════════════════════
    SCALE_UP_THRESHOLD: float = 0.8       # Seuil d'utilisation pour scale-up
    SCALE_DOWN_THRESHOLD: float = 0.3     # Seuil d'utilisation pour scale-down
    TARGET_UTILIZATION: float = 0.7       # Utilisation cible
    MAX_QUEUE_LENGTH_TRIGGER: int = 20    # Longueur de file declenchant scale-up
    
    # ══════════════════════════════════════════════════════════════
    # QUALITE DE SERVICE (QoS)
    # ══════════════════════════════════════════════════════════════
    TARGET_WAITING_TIME_MINUTES: float = 5.0   # Temps d'attente cible
    MAX_WAITING_TIME_MINUTES: float = 15.0     # Temps d'attente max acceptable
    TARGET_REJECTION_RATE: float = 0.01        # Taux de rejet cible (1%)
    MAX_REJECTION_RATE: float = 0.05           # Taux de rejet max (5%)
    
    # ══════════════════════════════════════════════════════════════
    # RESSOURCES - Specs techniques
    # ══════════════════════════════════════════════════════════════
    CPU_PER_RUNNER: int = 2               # vCPU par runner
    MEMORY_PER_RUNNER_GB: float = 4.0     # RAM par runner (GB)
    DISK_PER_RUNNER_GB: float = 20.0      # Disque par runner (GB)


@dataclass
class ServerConfig:
    """
    Configuration complete des serveurs de la moulinette.
    
    Cette classe contient tous les parametres configurables
    pour les runners qui executent les corrections.
    
    Exemple d'utilisation:
        # Configuration par defaut
        config = ServerConfig()
        
        # Configuration personnalisee
        config = ServerConfig(
            n_servers=8,
            service_rate=10.0,
            buffer_size=200,
            cost_per_server_hour=0.75
        )
    """
    
    # ══════════════════════════════════════════════════════════════
    # INFRASTRUCTURE
    # ══════════════════════════════════════════════════════════════
    n_servers: int = ServerConfigDefaults.N_SERVERS
    min_servers: int = ServerConfigDefaults.MIN_SERVERS
    max_servers: int = ServerConfigDefaults.MAX_SERVERS
    
    # ══════════════════════════════════════════════════════════════
    # PERFORMANCE
    # ══════════════════════════════════════════════════════════════
    service_rate: float = ServerConfigDefaults.SERVICE_RATE
    service_variance: float = ServerConfigDefaults.SERVICE_VARIANCE
    avg_test_duration_minutes: float = ServerConfigDefaults.AVG_TEST_DURATION_MINUTES
    timeout_minutes: float = ServerConfigDefaults.TIMEOUT_MINUTES
    
    # ══════════════════════════════════════════════════════════════
    # CAPACITE
    # ══════════════════════════════════════════════════════════════
    buffer_size: int = ServerConfigDefaults.BUFFER_SIZE
    max_concurrent_tests: int = ServerConfigDefaults.MAX_CONCURRENT_TESTS
    
    # ══════════════════════════════════════════════════════════════
    # COUTS
    # ══════════════════════════════════════════════════════════════
    cost_per_server_hour: float = ServerConfigDefaults.COST_PER_SERVER_HOUR
    cost_per_rejection: float = ServerConfigDefaults.COST_PER_REJECTION
    cost_per_waiting_minute: float = ServerConfigDefaults.COST_PER_WAITING_MINUTE
    startup_cost: float = ServerConfigDefaults.STARTUP_COST
    
    # ══════════════════════════════════════════════════════════════
    # TEMPS
    # ══════════════════════════════════════════════════════════════
    startup_time_seconds: float = ServerConfigDefaults.STARTUP_TIME_SECONDS
    shutdown_time_seconds: float = ServerConfigDefaults.SHUTDOWN_TIME_SECONDS
    cooldown_seconds: float = ServerConfigDefaults.COOLDOWN_SECONDS
    
    # ══════════════════════════════════════════════════════════════
    # AUTO-SCALING
    # ══════════════════════════════════════════════════════════════
    scale_up_threshold: float = ServerConfigDefaults.SCALE_UP_THRESHOLD
    scale_down_threshold: float = ServerConfigDefaults.SCALE_DOWN_THRESHOLD
    target_utilization: float = ServerConfigDefaults.TARGET_UTILIZATION
    max_queue_length_trigger: int = ServerConfigDefaults.MAX_QUEUE_LENGTH_TRIGGER
    
    # ══════════════════════════════════════════════════════════════
    # QUALITE DE SERVICE
    # ══════════════════════════════════════════════════════════════
    target_waiting_time_minutes: float = ServerConfigDefaults.TARGET_WAITING_TIME_MINUTES
    max_waiting_time_minutes: float = ServerConfigDefaults.MAX_WAITING_TIME_MINUTES
    target_rejection_rate: float = ServerConfigDefaults.TARGET_REJECTION_RATE
    max_rejection_rate: float = ServerConfigDefaults.MAX_REJECTION_RATE
    
    # ══════════════════════════════════════════════════════════════
    # RESSOURCES
    # ══════════════════════════════════════════════════════════════
    cpu_per_runner: int = ServerConfigDefaults.CPU_PER_RUNNER
    memory_per_runner_gb: float = ServerConfigDefaults.MEMORY_PER_RUNNER_GB
    disk_per_runner_gb: float = ServerConfigDefaults.DISK_PER_RUNNER_GB
    
    # ══════════════════════════════════════════════════════════════
    # PROPRIETES CALCULEES
    # ══════════════════════════════════════════════════════════════
    
    @property
    def total_capacity(self) -> float:
        """Capacite totale en corrections/heure."""
        return self.n_servers * self.service_rate
    
    @property
    def total_cpu(self) -> int:
        """Nombre total de vCPU."""
        return self.n_servers * self.cpu_per_runner
    
    @property
    def total_memory_gb(self) -> float:
        """Memoire totale en GB."""
        return self.n_servers * self.memory_per_runner_gb
    
    @property
    def total_disk_gb(self) -> float:
        """Disque total en GB."""
        return self.n_servers * self.disk_per_runner_gb
    
    def get_hourly_cost(self) -> float:
        """Cout horaire total des serveurs."""
        return self.n_servers * self.cost_per_server_hour
    
    def get_daily_cost(self) -> float:
        """Cout journalier total des serveurs (24h)."""
        return self.get_hourly_cost() * 24
    
    def get_monthly_cost(self) -> float:
        """Cout mensuel total des serveurs (30 jours)."""
        return self.get_daily_cost() * 30
    
    def validate(self) -> list[str]:
        """
        Valide la configuration et retourne les erreurs.
        
        Returns:
            Liste des messages d'erreur (vide si valide)
        """
        errors = []
        
        if self.n_servers < self.min_servers:
            errors.append(f"n_servers ({self.n_servers}) < min_servers ({self.min_servers})")
        if self.n_servers > self.max_servers:
            errors.append(f"n_servers ({self.n_servers}) > max_servers ({self.max_servers})")
        if self.service_rate <= 0:
            errors.append(f"service_rate doit etre > 0 (actuel: {self.service_rate})")
        if self.buffer_size <= 0:
            errors.append(f"buffer_size doit etre > 0 (actuel: {self.buffer_size})")
        if not 0 <= self.target_utilization <= 1:
            errors.append(f"target_utilization doit etre entre 0 et 1 (actuel: {self.target_utilization})")
        if self.scale_down_threshold >= self.scale_up_threshold:
            errors.append(f"scale_down_threshold ({self.scale_down_threshold}) >= scale_up_threshold ({self.scale_up_threshold})")
            
        return errors
    
    def to_dict(self) -> dict:
        """Convertit la config en dictionnaire."""
        return {
            # Infrastructure
            'n_servers': self.n_servers,
            'min_servers': self.min_servers,
            'max_servers': self.max_servers,
            # Performance
            'service_rate': self.service_rate,
            'service_variance': self.service_variance,
            'avg_test_duration_minutes': self.avg_test_duration_minutes,
            'timeout_minutes': self.timeout_minutes,
            # Capacite
            'buffer_size': self.buffer_size,
            'max_concurrent_tests': self.max_concurrent_tests,
            # Couts
            'cost_per_server_hour': self.cost_per_server_hour,
            'cost_per_rejection': self.cost_per_rejection,
            'cost_per_waiting_minute': self.cost_per_waiting_minute,
            'startup_cost': self.startup_cost,
            # Temps
            'startup_time_seconds': self.startup_time_seconds,
            'shutdown_time_seconds': self.shutdown_time_seconds,
            'cooldown_seconds': self.cooldown_seconds,
            # Auto-scaling
            'scale_up_threshold': self.scale_up_threshold,
            'scale_down_threshold': self.scale_down_threshold,
            'target_utilization': self.target_utilization,
            'max_queue_length_trigger': self.max_queue_length_trigger,
            # QoS
            'target_waiting_time_minutes': self.target_waiting_time_minutes,
            'max_waiting_time_minutes': self.max_waiting_time_minutes,
            'target_rejection_rate': self.target_rejection_rate,
            'max_rejection_rate': self.max_rejection_rate,
            # Ressources
            'cpu_per_runner': self.cpu_per_runner,
            'memory_per_runner_gb': self.memory_per_runner_gb,
            'disk_per_runner_gb': self.disk_per_runner_gb,
            # Calculees
            'total_capacity': self.total_capacity,
            'total_cpu': self.total_cpu,
            'total_memory_gb': self.total_memory_gb,
            'hourly_cost': self.get_hourly_cost(),
        }
    
    def summary(self) -> str:
        """Resume textuel de la configuration."""
        return f"""
Configuration Serveur Moulinette
================================
Infrastructure: {self.n_servers} runners (min={self.min_servers}, max={self.max_servers})
Performance:    {self.service_rate} corrections/h/runner = {self.total_capacity} total/h
Buffer:         {self.buffer_size} soumissions max en file
Cout:           {self.get_hourly_cost():.2f} EUR/h = {self.get_monthly_cost():.2f} EUR/mois
Ressources:     {self.total_cpu} vCPU, {self.total_memory_gb:.1f} GB RAM
"""


# Instance globale par defaut (peut etre modifiee)
DEFAULT_SERVER_CONFIG = ServerConfig()
