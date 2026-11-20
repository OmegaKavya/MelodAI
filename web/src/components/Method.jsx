import React from "react";
import Panel from "./Panel.jsx";
import "./Method.css";

// The section that makes the page worth more than its demo. It says what was
// tried, what failed, and what the honest limit is.
const STEPS = [
  {
    n: "01",
    title: "The deployed model was not the trained model",
    body: "Training built a three channel image: percussive, harmonic plus chroma, mel plus contrast. The app built a plain mel spectrogram at a different hop length, copied to three identical channels and padded into a corner, leaving 43% of the frame black. The reported 66% was measured correctly and never described what users got.",
  },
  {
    n: "02",
    title: "Song level evaluation replaced image level",
    body: "Splitting on images put segments of the same song on both sides of the split. Re-evaluated per song, the original model scored 0 of 9 on peaceful while still reporting 64% overall.",
  },
  {
    n: "03",
    title: "Five songs carried contradictory labels",
    body: "Found by hashing file contents, not by listening: the same recording sat in two mood folders with two different labels. Confirmed in both the audio and the spectrograms.",
  },
  {
    n: "04",
    title: "The taxonomy was tested, not assumed",
    body: "Merging calm with drive, and romantic with sad, was measured against alternatives rather than argued. The as shipped grouping won at 77%; my own proposed regrouping lost at 68%.",
  },
  {
    n: "05",
    title: "The embedding map settled it",
    body: "A k-nearest-neighbour probe on YAMNet embeddings reaches 85.2% on arousal, a 4.13 times lift over chance, and 44.7% on valence, a 1.25 times lift. Silhouette for valence is 0.073, which is no structure at all. The model is not underfitting valence; this representation does not carry it.",
  },
  {
    n: "06",
    title: "So the app claims only what holds",
    body: "One bit, energy, 26 of 26 on held out songs where guessing the majority gets 21 and catches none of the five high energy tracks. Everything else on this page is evidence rather than prediction.",
  },
];

export default function Method() {
  return (
    <Panel className="method" aria-labelledby="method-heading">
      <p className="eyebrow">How it got here</p>
      <h2 id="method-heading">Six things that were wrong, and how each was found</h2>
      <p className="note">
        A model that reports a number is easy. Most of this project was
        establishing that the number meant anything, and most of that work was
        negative.
      </p>

      <ol className="method-list">
        {STEPS.map((step) => (
          <li key={step.n}>
            <span className="method-n">{step.n}</span>
            <div>
              <h3>{step.title}</h3>
              <p>{step.body}</p>
            </div>
          </li>
        ))}
      </ol>

      <div className="method-limit">
        <h3>What this cannot do</h3>
        <p>
          It cannot tell a sad song from a peaceful one. That was measured, not
          assumed, and no amount of training fixes it, because the separation is
          absent from the representation rather than missed by the classifier.
          Serving runs without TensorFlow: YAMNet from a vendored TFLite build
          and the head as four matrix multiplies in NumPy, checked against the
          original to 1e-7.
        </p>
      </div>
    </Panel>
  );
}
