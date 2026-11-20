import React from "react";
import Panel from "./Panel.jsx";
import "./Hears.css";

// YAMNet's AudioSet layer, which the classifier throws away. Top of tree
// labels like "Music" are filtered server side: every song scores high on
// them, so they carry no information about this song.
export default function Hears({ hears }) {
  const max = Math.max(...hears.map((h) => h.score), 0.001);
  return (
    <Panel className="hears" aria-labelledby="hears-heading">
      <h2 id="hears-heading">What the network hears</h2>
      <p className="note">
        AudioSet classes, averaged over every window. This is the layer
        underneath the verdict, before anything is reduced to one bit.
      </p>

      <ul className="hears-list">
        {hears.map((item) => (
          <li key={item.name}>
            <span className="hears-name">{item.name}</span>
            <span className="hears-track">
              <span className="hears-bar" style={{ width: `${(item.score / max) * 100}%` }} />
            </span>
            <span className="hears-value">{(item.score * 100).toFixed(1)}</span>
          </li>
        ))}
      </ul>
    </Panel>
  );
}
