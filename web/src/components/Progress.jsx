import React, { useEffect, useState } from "react";
import "./Progress.css";

// What the server is doing, in order.
//
// There is no progress channel from the server, so this does not claim a
// percentage. It names the stages the request actually passes through and
// highlights them in turn at roughly the pace they take, which tells a
// waiting visitor what is happening without inventing a number.
const STAGES = [
  { label: "Decoding audio", detail: "resampled to 16 kHz mono" },
  { label: "Running YAMNet", detail: "1024 numbers per 0.96s window" },
  { label: "Reading energy", detail: "four matrix multiplies per window" },
  { label: "Drawing the spectrogram", detail: "percussive, harmonic, mel" },
];

export default function Progress() {
  const [stage, setStage] = useState(0);

  useEffect(() => {
    const id = setInterval(
      () => setStage((s) => Math.min(STAGES.length - 1, s + 1)),
      2600
    );
    return () => clearInterval(id);
  }, []);

  return (
    <div className="progress-stages" role="status" aria-label="Analysing">
      <ol>
        {STAGES.map((item, i) => (
          <li
            key={item.label}
            className={i < stage ? "done" : i === stage ? "active" : ""}
          >
            <span className="pip" aria-hidden="true" />
            <b>{item.label}</b>
            <em>{item.detail}</em>
          </li>
        ))}
      </ol>
      <p className="progress-foot">
        First run after a quiet spell also waits for the server to wake.
      </p>
    </div>
  );
}
