import React, { useMemo, useRef, useState } from "react";
import Panel from "./Panel.jsx";
import "./EnergyCurve.css";

const W = 900;
const H = 240;
const PAD = { top: 16, right: 16, bottom: 28, left: 40 };

// Energy per window, drawn as a line against the 50% decision boundary.
// A single series, so no legend: the heading names it.
export default function EnergyCurve({ energy, duration }) {
  const svgRef = useRef(null);
  const [hover, setHover] = useState(null);
  const curve = energy.curve;

  const { path, area, x, y } = useMemo(() => {
    const plotW = W - PAD.left - PAD.right;
    const plotH = H - PAD.top - PAD.bottom;
    const xs = (i) => PAD.left + (curve.length < 2 ? 0 : (i / (curve.length - 1)) * plotW);
    const ys = (v) => PAD.top + (1 - v) * plotH;
    const d = curve.map((v, i) => `${i === 0 ? "M" : "L"}${xs(i).toFixed(2)},${ys(v).toFixed(2)}`).join("");
    return {
      path: d,
      area: `${d}L${xs(curve.length - 1).toFixed(2)},${ys(0)}L${xs(0).toFixed(2)},${ys(0)}Z`,
      x: xs,
      y: ys,
    };
  }, [curve]);

  const move = (event) => {
    const svg = svgRef.current;
    if (!svg) return;
    const box = svg.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * W;
    const plotW = W - PAD.left - PAD.right;
    const ratio = Math.min(1, Math.max(0, (px - PAD.left) / plotW));
    setHover(Math.round(ratio * (curve.length - 1)));
  };

  const seconds = (i) => ((i / Math.max(1, curve.length - 1)) * duration).toFixed(1);

  return (
    <Panel className="curve" aria-labelledby="curve-heading">
      <h2 id="curve-heading">Energy through the track</h2>
      <p className="note">
        One reading per 0.96 second window. The verdict above is the mean of
        this line, so a build, a drop or a quiet opening is visible here rather
        than averaged away.
      </p>

      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="curve-svg"
        role="img"
        aria-label={`Energy over time. Mean ${Math.round(energy.score * 100)} percent, peak ${Math.round(energy.peak * 100)}, trough ${Math.round(energy.trough * 100)}.`}
        onMouseMove={move}
        onMouseLeave={() => setHover(null)}
      >
        <defs>
          <linearGradient id="curve-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--gold)" stopOpacity="0.28" />
            <stop offset="100%" stopColor="var(--gold)" stopOpacity="0" />
          </linearGradient>
        </defs>

        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line x1={PAD.left} x2={W - PAD.right} y1={y(v)} y2={y(v)}
              stroke="var(--rule-soft)" strokeWidth="1" />
            <text x={PAD.left - 8} y={y(v) + 4} textAnchor="end" className="curve-tick">
              {v * 100}
            </text>
          </g>
        ))}

        {/* The decision boundary. Everything above it is "high energy". */}
        <line x1={PAD.left} x2={W - PAD.right} y1={y(0.5)} y2={y(0.5)}
          stroke="var(--sindoor)" strokeWidth="1" strokeDasharray="4 4" opacity="0.7" />
        <text x={W - PAD.right} y={y(0.5) - 6} textAnchor="end" className="curve-boundary">
          decision boundary
        </text>

        <path d={area} fill="url(#curve-fill)" />
        <path d={path} fill="none" stroke="var(--gold)" strokeWidth="2"
          strokeLinejoin="round" strokeLinecap="round" />

        {hover !== null && (
          <g className="curve-hover">
            <line x1={x(hover)} x2={x(hover)} y1={PAD.top} y2={H - PAD.bottom}
              stroke="var(--gold-bright)" strokeWidth="1" opacity="0.6" />
            <circle cx={x(hover)} cy={y(curve[hover])} r="4.5"
              fill="var(--gold-bright)" stroke="var(--ink)" strokeWidth="2" />
          </g>
        )}

        <text x={PAD.left} y={H - 8} className="curve-tick">0s</text>
        <text x={W - PAD.right} y={H - 8} textAnchor="end" className="curve-tick">
          {Math.round(duration)}s
        </text>
      </svg>

      <p className="curve-readout" aria-live="polite">
        {hover !== null ? (
          <>
            At <b>{seconds(hover)}s</b> the network reads{" "}
            <b>{Math.round(curve[hover] * 100)}%</b> energy.
          </>
        ) : (
          <>Hover the line to read any window.</>
        )}
      </p>
    </Panel>
  );
}
