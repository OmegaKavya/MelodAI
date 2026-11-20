import React from "react";
import Panel from "./Panel.jsx";
import "./Verdict.css";

const pct = (v) => `${Math.round(v * 100)}%`;
// One decimal for the headline average only: 0.995 shown as "100%" next to a
// trough of 43% reads as a contradiction rather than a rounding.
const mean = (v) => `${(v * 100).toFixed(v > 0.99 || v < 0.01 ? 1 : 0)}%`;

// The headline is a number, not a chart, so it is set as one.
export default function Verdict({ energy, windows, duration }) {
  const high = energy.label === "High energy";
  return (
    <Panel className={`verdict${high ? " high" : ""}`} aria-labelledby="verdict-heading">
      <p className="eyebrow">The verdict</p>
      <h2 id="verdict-heading">
        <span className="foil">{energy.label}</span>
      </h2>

      <p className="verdict-line">
        Averaged to <b>{mean(energy.score)}</b> across <b>{windows}</b> windows of
        0.96 seconds, over {Math.round(duration)} seconds of audio.
      </p>

      <dl className="verdict-stats">
        <div><dt>Peak</dt><dd>{pct(energy.peak)}</dd></div>
        <div><dt>Trough</dt><dd>{pct(energy.trough)}</dd></div>
        <div><dt>Swing</dt><dd>{energy.swing.toFixed(2)}</dd></div>
      </dl>

      <p className="verdict-note">
        {energy.swing > 0.5
          ? "A wide swing: the track moves a long way between its quietest and loudest passage, so the average hides a build."
          : "A narrow swing: the track holds roughly one level throughout, so the average describes it well."}
      </p>
    </Panel>
  );
}
