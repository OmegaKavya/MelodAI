import React from "react";
import Panel from "./Panel.jsx";
import "./Finding.css";

// Why the page claims one thing.
//
// The long version lives in the case study. This is the short one, because
// the hero shows a failure figure and a reader who never clicks through
// deserves to know how it resolved.

// Two measurements and the rate you get for guessing. Drawn rather than
// listed, so the gap is the thing you see first.
const PROBE = [
  { label: "Loud from quiet", value: 85, tone: "win" },
  { label: "Happy from sad", value: 45, tone: "fail" },
];
const CHANCE = 36;

export default function Finding() {
  return (
    <Panel className="finding" aria-labelledby="finding-heading">
      <p className="eyebrow">Why it only reports energy</p>
      <h2 id="finding-heading">A third of the model never worked</h2>

      <div className="finding-grid">
        <div className="finding-text">
          <p>
            The first version sorted songs into three moods and reported 66%
            accuracy. Scored per song rather than per spectrogram image, it
            found <b>none of the nine peaceful tracks</b>. Two classes had been
            carrying a third that never worked at all.
          </p>
          <p>
            That is a property of the audio, not the architecture. Five
            architectures failed the same way, including on their own training
            data. Probing the embeddings directly says why: how loud a song is
            sits in a different region of the space, and whether it is happy or
            sad does not separate at all.
          </p>
          <p>
            So the page claims the one thing that holds, and shows you the
            evidence behind it rather than asking you to take the number on
            trust.
          </p>
        </div>

        <figure className="finding-chart">
          <figcaption>Nearest-neighbour agreement, 131 songs</figcaption>
          <ul>
            {PROBE.map((row) => (
              <li key={row.label}>
                <span className="fc-label">{row.label}</span>
                <span className="fc-track">
                  <span className={`fc-bar ${row.tone}`} style={{ width: `${row.value}%` }} />
                  <span className="fc-chance" style={{ left: `${CHANCE}%` }} />
                </span>
                <span className="fc-value">{row.value}%</span>
              </li>
            ))}
          </ul>
          <p className="fc-key">
            <span className="fc-chance-key" aria-hidden="true" />
            Guessing the most common answer scores {CHANCE}%
          </p>
        </figure>
      </div>
    </Panel>
  );
}
