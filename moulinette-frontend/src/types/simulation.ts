export interface SimulationConfig {
  duration_hours: number;
  time_step_minutes: number;
  include_weekend: boolean;
  start_hour: number;
  start_day: number;
  deadline_at_hour?: number | null;
  n_simulation_runs: number;
}

export interface SimulationReport {
  avg_waiting_time: number;
  std_waiting_time: number;
  avg_system_time: number;
  max_queue_length: number;
  avg_queue_length: number;
  rejection_rate: number;
  throughput: number;
  utilization: number;
  peak_hours: number[];
  peak_load: number;
  optimal_servers: number;
  estimated_cost: number;
  recommendations: string[];
  time_series?: Record<string, number[]>;
}