import { useState } from "react";
import { type SimulationConfig } from "../types/simulation";

interface Props {
  onRun: (config: SimulationConfig) => void;
}

export default function ConfigForm({ onRun }: Props) {
  const [config, setConfig] = useState<SimulationConfig>({
    duration_hours: 24,
    time_step_minutes: 15,
    include_weekend: false,
    start_hour: 0,
    start_day: 0,
    n_simulation_runs: 10,
    deadline_at_hour: null
  });

  const update = (k: keyof SimulationConfig, v: any) =>
    setConfig({ ...config, [k]: v });

  return (
    <div style={{ padding: 16 }}>
      <h2>Configuration Simulation</h2>

      <div>
        <label>Durée (heures)</label>
        <input
          type="number"
          value={config.duration_hours}
          onChange={e => update("duration_hours", Number(e.target.value))}
        />
      </div>

      <div>
        <label>Time Step (min)</label>
        <input
          type="number"
          value={config.time_step_minutes}
          onChange={e => update("time_step_minutes", Number(e.target.value))}
        />
      </div>

      <div>
        <label>Deadline Hour</label>
        <input
          type="number"
          value={config.deadline_at_hour ?? ""}
          onChange={e =>
            update("deadline_at_hour", e.target.value ? Number(e.target.value) : null)
          }
        />
      </div>

      <button onClick={() => onRun(config)}>
        Lancer Simulation
      </button>
    </div>
  );
}