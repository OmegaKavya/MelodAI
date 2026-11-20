import React from "react";
import Panel from "./Panel.jsx";
import "./Neighbours.css";

const MOOD_CLASS = { peaceful: "peaceful", energetic: "energetic", emotional: "emotional" };

// Nearest songs by cosine distance in the same 1024 dimensional space the
// verdict is made from. Cosine rather than euclidean because embedding
// magnitude tracks loudness, and two masterings of one arrangement should
// still count as close.
export default function Neighbours({ neighbours }) {
  return (
    <Panel className="neighbours" aria-labelledby="neighbours-heading">
      <h2 id="neighbours-heading">Closest songs in the corpus</h2>
      <p className="note">
        Measured in the same embedding space the verdict comes from, so this is
        the evidence rather than a second opinion.
      </p>

      <ol className="neighbours-list">
        {neighbours.map((song) => (
          <li key={song.title}>
            <span className={`dot ${MOOD_CLASS[song.mood] || ""}`} aria-hidden="true" />
            <span className="neighbours-title">{song.title}</span>
            <span className="neighbours-mood">{song.mood}</span>
            <span className="neighbours-score">{song.similarity.toFixed(3)}</span>
          </li>
        ))}
      </ol>
    </Panel>
  );
}
