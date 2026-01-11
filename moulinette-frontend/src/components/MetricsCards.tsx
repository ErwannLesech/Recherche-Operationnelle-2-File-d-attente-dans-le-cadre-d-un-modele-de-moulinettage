import { type SimulationReport } from "../types/simulation";

export default function MetricsCards({ report }: { report: SimulationReport }) {
  const items = [
    ["Temps d’attente moyen (h)", report.avg_waiting_time.toFixed(3)],
    ["Std attente", report.std_waiting_time.toFixed(3)],
    ["Temps système", report.avg_system_time.toFixed(3)],
    ["Longueur max file", report.max_queue_length],
    ["Rejet (%)", (report.rejection_rate * 100).toFixed(2)],
    ["Utilisation", (report.utilization * 100).toFixed(2)],
    ["Throughput", report.throughput.toFixed(2)],
  ];

  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12 }}>
      {items.map(([k, v]) => (
        <div key={k} style={{ padding: 16, background: "#222", color: "white", borderRadius: 8 }}>
          <strong>{k}</strong>
          <div>{v}</div>
        </div>
      ))}
    </div>
  );
}