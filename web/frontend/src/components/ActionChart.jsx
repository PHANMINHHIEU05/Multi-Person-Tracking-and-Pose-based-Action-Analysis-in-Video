import { useEffect, useRef } from "react";
import {
  Chart,
  ArcElement,
  Tooltip,
  Legend,
  DoughnutController,
} from "chart.js";

Chart.register(ArcElement, Tooltip, Legend, DoughnutController);

const COLORS = [
  "#3b82f6", // blue  - Walking
  "#ef4444", // red   - Fall
  "#10b981", // green - Standing
  "#f59e0b", // amber - Sitting
  "#8b5cf6", // purple - other
];

/**
 * Real-time doughnut chart of action distribution.
 * Props: counts — { ActionLabel: count, ... }
 */
export default function ActionChart({ counts }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    chartRef.current = new Chart(canvasRef.current, {
      type: "doughnut",
      data: {
        labels: [],
        datasets: [{ data: [], backgroundColor: COLORS, borderWidth: 0 }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: { color: "#94a3b8", font: { size: 12 }, padding: 12 },
          },
          tooltip: { callbacks: { label: (c) => ` ${c.label}: ${c.raw}` } },
        },
      },
    });
    return () => {
      chartRef.current?.destroy();
      chartRef.current = null;
    };
  }, []);

  // Update chart data reactively
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    const labels = Object.keys(counts);
    const values = Object.values(counts);
    chart.data.labels = labels;
    chart.data.datasets[0].data = values;
    chart.data.datasets[0].backgroundColor = labels.map(
      (_, i) => COLORS[i % COLORS.length],
    );
    chart.update("none");
  }, [counts]);

  const total = Object.values(counts).reduce((a, b) => a + b, 0);

  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-sm font-semibold text-slate-300">
        Action Distribution
      </h3>
      <div className="relative" style={{ height: 200 }}>
        <canvas ref={canvasRef} />
        {total === 0 && (
          <div className="absolute inset-0 flex items-center justify-center text-slate-500 text-xs">
            No data yet
          </div>
        )}
      </div>
    </div>
  );
}
