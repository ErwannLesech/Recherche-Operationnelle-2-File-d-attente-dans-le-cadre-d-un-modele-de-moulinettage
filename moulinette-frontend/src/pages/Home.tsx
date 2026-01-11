import { useState } from "react";
import ConfigForm from "../components/ConfigForm";
import MetricsCards from "../components/MetricsCards";
import TimeSeriesChart from "../components/TimeSeriesChart";
import { runSimulation } from "../api/simulationApi";
import { type SimulationReport } from "../types/simulation";

export default function Home() {
  const [report, setReport] = useState<SimulationReport | null>(null);
  const [loading, setLoading] = useState(false);

  const handleRun = async (config: any) => {
    setLoading(true);
    const res = await runSimulation(config);
    setReport(res);
    setLoading(false);
  };

  return (
    <div>
      <h1>Moulinette Simulator</h1>

      <ConfigForm onRun={handleRun} />

      {loading && <p>Simulation en cours…</p>}

      {report && (
        <>
          <MetricsCards report={report}/>
          {report.time_series?.queue_length && (
            <TimeSeriesChart data={report.time_series.queue_length}/>
          )}

          <h3>Recommandations</h3>
          <ul>
            {report.recommendations.map((r,i)=> <li key={i}>{r}</li>)}
          </ul>
        </>
      )}
    </div>
  );
}