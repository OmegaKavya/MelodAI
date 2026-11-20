import React from "react";
import Panel from "./Panel.jsx";
import useCountUp from "../useCountUp";
import "./Verdict.css";

function Stat({ label, value, suffix = "%", decimals = 0 }) {
  const shown = useCountUp(value);
  return (
    <div>
      <dt>{label}</dt>
      <dd>
        {shown.toFixed(decimals)}
        <span>{suffix}</span>
      </dd>
    </div>
  );
}

// A ring rather than a bar: the headline is one proportion, and a ring reads
// as a gauge at a glance without needing an axis.
function Dial({ score, high }) {
  const shown = useCountUp(score, 1100);
  const r = 78;
  const c = 2 * Math.PI * r;
  return (
    <div className="dial">
      <svg viewBox="0 0 200 200" aria-hidden="true">
        <circle cx="100" cy="100" r={r} className="dial-track" />
        <circle
          cx="100" cy="100" r={r}
          className={`dial-value${high ? " high" : ""}`}
          strokeDasharray={`${c}`}
          strokeDashoffset={c * (1 - shown)}
        />
        {/* The decision boundary, drawn where it actually falls. */}
        <line x1="100" y1="12" x2="100" y2="30"
          className="dial-boundary" transform="rotate(180 100 100)" />
      </svg>
      <div className="dial-centre">
        <strong>{(shown * 100).toFixed(shown > 0.99 || shown < 0.01 ? 1 : 0)}</strong>
        <span>%</span>
      </div>
    </div>
  );
}

export default function Verdict({ energy, windows, duration, familiar }) {
  const high = energy.label === "High energy";
  const unfamiliar = familiar && !familiar.in_distribution;

  return (
    <Panel className={`verdict${high ? " high" : ""}`} aria-labelledby="verdict-heading">
      <div className="verdict-grid">
        <Dial score={energy.score} high={high} />

        <div className="verdict-body">
          <p className="eyebrow">The verdict</p>
          <h2 id="verdict-heading">
            <span className="foil">{energy.label}</span>
          </h2>
          <p className="verdict-line">
            Mean of <b>{windows}</b> windows of 0.96 seconds, across{" "}
            {Math.round(duration)} seconds of audio. The boundary is 50%.
          </p>

          <dl className="verdict-stats">
            <Stat label="Peak" value={energy.peak * 100} />
            <Stat label="Trough" value={energy.trough * 100} />
            <Stat label="Swing" value={energy.swing} suffix="" decimals={2} />
          </dl>

          <p className="verdict-note">
            {energy.swing > 0.5
              ? "A wide swing: the track travels a long way between its quietest and loudest passage, so the mean hides a build."
              : "A narrow swing: the track holds roughly one level throughout, so the mean describes it well."}
          </p>
        </div>
      </div>

      {unfamiliar && (
        <p className="verdict-warn" role="note">
          <b>Outside the corpus.</b> The closest song in the training set is{" "}
          {familiar.closest.toFixed(2)} cosine away, below the {familiar.floor} that
          Hindi film tracks reach. The verdict is still shown, but it is an
          extrapolation rather than a reading.
        </p>
      )}
    </Panel>
  );
}
