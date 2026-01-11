import axios from "axios";
import { type SimulationConfig, type SimulationReport } from "../types/simulation";

const api = axios.create({
  baseURL: "http://localhost:8000", // ← change selon votre backend
});

export async function runSimulation(
  config: SimulationConfig
): Promise<SimulationReport> {
  const res = await api.post("/simulate", config);
  return res.data;
}