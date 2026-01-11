from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np

from app.models import MM1Queue, MMcQueue, MMcKQueue, MD1Queue, MG1Queue
from app.personas import PersonaFactory
from app.simulation import RushSimulator, MoulinetteSystem, SimulationConfig, ServerConfig
from app.optimization import CostOptimizer, ScalingAdvisor, CostModel

app = FastAPI(
    title="Moulinette Backend API - EPITA",
    description="API de simulation / optimisation file d'attente pour la moulinette",
    version="1.0.0"
)

# ------------------------------------------------------------
# CORS
# ------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change avec ton frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Schemas
# ------------------------------------------------------------

class QueueRequest(BaseModel):
    model: str  # "MM1" | "MMc" | "MMcK" | "MD1" | "MG1"
    lambda_rate: float
    mu_rate: float
    servers: int = 1
    buffer: int = 100
    cv2: float = 1.0


class SimulationRequest(BaseModel):
    model: str
    lambda_rate: float
    mu_rate: float
    servers: int = 1
    buffer: int = 100
    cv2: float = 1.0
    duration: int


class RushRequest(BaseModel):
    duration_hours: int
    base_load: float
    mu_rate: float
    buffer_size: int
    build_servers: int
    seed: int = 42


class OptimizationRequest(BaseModel):
    lambda_rate: float
    mu_rate: float
    cost_per_server: float
    fixed_cost: float
    penalty_wait: float
    alpha: float
    max_servers: int
    buffer: int


class ScalingRequest(BaseModel):
    lambda_rate: float
    mu_rate: float
    max_servers: int
    buffer: int


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def build_queue(req: QueueRequest):
    if req.model == "MM1":
        return MM1Queue(req.lambda_rate, req.mu_rate)
    if req.model == "MMc":
        return MMcQueue(req.lambda_rate, req.mu_rate, req.servers)
    if req.model == "MMcK":
        return MMcKQueue(req.lambda_rate, req.mu_rate, req.servers, req.buffer)
    if req.model == "MD1":
        return MD1Queue(req.lambda_rate, req.mu_rate)
    if req.model == "MG1":
        return MG1Queue(req.lambda_rate, req.mu_rate, req.cv2)

    raise HTTPException(400, "Unknown model")


# ------------------------------------------------------------
# ROOT
# ------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "OK", "name": "Moulinette Backend API"}


# ------------------------------------------------------------
# THEORETICAL METRICS
# ------------------------------------------------------------

@app.post("/queue/metrics")
def compute_metrics(req: QueueRequest):
    queue = build_queue(req)
    metrics = queue.compute_theoretical_metrics()

    return {
        "L": metrics.L,
        "Lq": metrics.Lq,
        "W": metrics.W,
        "Wq": metrics.Wq,
        "rho": metrics.rho,
        "P_block": getattr(metrics, "Pk", 0.0)
    }


# ------------------------------------------------------------
# MONTE CARLO SIMULATION
# ------------------------------------------------------------

@app.post("/queue/simulate")
def simulate_model(req: SimulationRequest):
    queue = build_queue(req)

    result = queue.simulate(req.duration)

    return {
        "served": result.n_served,
        "avg_system_time": float(np.mean(result.system_times)) if result.system_times else 0,
        "avg_waiting_time": float(np.mean(result.waiting_times)) if result.waiting_times else 0,
        "max_queue": int(np.max(result.queue_length_trace)) if result.queue_length_trace else 0,
        "time_trace": list(result.time_trace),
        "queue_trace": list(result.queue_length_trace)
    }


# ------------------------------------------------------------
# PERSONAS
# ------------------------------------------------------------

@app.get("/personas")
def get_personas():
    persons = PersonaFactory.create_all_personas()

    res = []
    for p in persons.values():
        res.append({
            "name": p.name,
            "base_rate": p.base_submission_rate,
            "cv2": p.variance_coefficient,
            "procrastination": p.procrastination_level,
            "peak_hours": p.peak_hours
        })
    return res


# ------------------------------------------------------------
# RUSH SIMULATION
# ------------------------------------------------------------

@app.post("/rush/simulate")
def rush_simulation(req: RushRequest):

    cfg = SimulationConfig(
        duration_hours=req.duration_hours,
        server_config=ServerConfig(
            n_servers=req.build_servers,
            service_rate=req.mu_rate,
            buffer_size=req.buffer_size
        ),
        seed=req.seed
    )

    sim = RushSimulator(cfg)
    report = sim.run()

    return {
        "throughput": report.throughput,
        "avg_system_time": report.avg_system_time,
        "avg_waiting_time": report.avg_waiting_time,
        "utilization": report.utilization,
        "max_queue": report.max_queue_length,
        "rejection_rate": report.rejection_rate
    }


# ------------------------------------------------------------
# COST OPTIMIZATION
# ------------------------------------------------------------

@app.post("/optimize")
def optimize(req: OptimizationRequest):

    cost = CostModel(
        cost_per_server_hour=req.cost_per_server,
        fixed_infrastructure_cost=req.fixed_cost,
        cost_per_waiting_minute=req.penalty_wait
    )

    opt = CostOptimizer(req.lambda_rate, req.mu_rate, cost)

    result = opt.optimize(
        alpha=req.alpha,
        c_range=(1, req.max_servers),
        K_range=(req.buffer, req.buffer)
    )

    return {
        "optimal_servers": result.optimal_servers,
        "optimal_buffer": result.optimal_buffer,
        "objective": result.objective_value,
        "utilization": result.utilization,
        "rejection_rate": result.rejection_rate,
        "waiting_time": result.avg_waiting_time
    }


# ------------------------------------------------------------
# AUTO-SCALING ADVISOR
# ------------------------------------------------------------

@app.post("/scaling")
def scaling(req: ScalingRequest):

    advisor = ScalingAdvisor(
        lambda_rate=req.lambda_rate,
        mu_rate=req.mu_rate,
        buffer_size=req.buffer
    )

    policy = advisor.recommend_policy(req.max_servers)

    return {
        "recommended_servers": policy.recommended_servers,
        "target_rho": policy.target_utilization,
        "expected_waiting": policy.expected_waiting_time
    }