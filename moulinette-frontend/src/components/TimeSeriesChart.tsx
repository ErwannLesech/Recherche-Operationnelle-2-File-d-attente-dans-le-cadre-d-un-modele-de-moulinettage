import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";

export default function TimeSeriesChart({ data }: { data: number[] }) {
  const formatted = data.map((v, i) => ({ t: i, value: v }));

  return (
    <LineChart width={700} height={300} data={formatted}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="t" label="Time step" />
      <YAxis />
      <Tooltip />
      <Line type="monotone" dataKey="value" stroke="#4ADE80"/>
    </LineChart>
  );
}