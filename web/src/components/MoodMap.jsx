import React, { useMemo, useRef, useState } from "react";
import Panel from "./Panel.jsx";
import "./MoodMap.css";

const W = 900;
const H = 520;
const PAD = 40;

const MOODS = [
  { key: "peaceful", label: "Peaceful", css: "var(--mood-peaceful)" },
  { key: "energetic", label: "Energetic", css: "var(--mood-energetic)" },
  { key: "emotional", label: "Emotional", css: "var(--mood-emotional)" },
];

// The corpus projected to two dimensions, and the finding the project rests
// on. Energetic separates; peaceful and emotional sit on top of each other.
// That is not a rendering artefact, it is the result.
export default function MoodMap({ corpus, position }) {
  const [hover, setHover] = useState(null);
  const wrapRef = useRef(null);

  const { points, project } = useMemo(() => {
    if (!corpus.length) return { points: [], project: () => [0, 0] };
    const xs = corpus.map((s) => s.x);
    const ys = corpus.map((s) => s.y);
    const all = position ? { x: xs.concat(position[0]), y: ys.concat(position[1]) } : { x: xs, y: ys };
    const minX = Math.min(...all.x), maxX = Math.max(...all.x);
    const minY = Math.min(...all.y), maxY = Math.max(...all.y);
    const p = (x, y) => [
      PAD + ((x - minX) / (maxX - minX || 1)) * (W - PAD * 2),
      H - PAD - ((y - minY) / (maxY - minY || 1)) * (H - PAD * 2),
    ];
    return { points: corpus.map((s) => ({ ...s, xy: p(s.x, s.y) })), project: p };
  }, [corpus, position]);

  if (!corpus.length) return null;
  const me = position ? project(position[0], position[1]) : null;

  return (
    <Panel className="map" aria-labelledby="map-heading">
      <h2 id="map-heading">Every song in the collection, placed by sound</h2>
      <p className="note">
        Each dot is one of the 131 Hindi film songs the model was fitted on.
        Songs that sound alike to the network sit near each other; the axes
        themselves carry no units, which is normal for a projection of this
        kind. Hover any dot to see the song.
      </p>
      <p className="map-look">
        <b>What to look for.</b> The gold cluster pulls away from the rest:
        high-energy songs really do sound different. Teal and rose sit on top
        of one another, and no amount of training separates them. That overlap
        is the reason this page reports energy and nothing else.
      </p>

      <ul className="map-legend">
        {MOODS.map((m) => (
          <li key={m.key}>
            <span className="dot" style={{ background: m.css }} aria-hidden="true" />
            {m.label}
          </li>
        ))}
        {me && (
          <li>
            <svg width="13" height="13" viewBox="-6 -6 12 12" aria-hidden="true">
              <path d={star(5.5, 2.4)} fill="var(--text)" />
            </svg>
            Your track
          </li>
        )}
      </ul>

      <div className="map-wrap" ref={wrapRef}>
        <svg viewBox={`0 0 ${W} ${H}`} className="map-svg" role="img"
          aria-label="Scatter plot of the song corpus coloured by mood, with energetic songs separating from peaceful and emotional, which overlap.">
          {points.map((s, i) => (
            <circle
              key={`${s.title}-${i}`}
              cx={s.xy[0]} cy={s.xy[1]} r="5.5"
              fill={`var(--mood-${s.mood})`}
              stroke="var(--ink)" strokeWidth="2"
              opacity={hover && hover.title !== s.title ? 0.35 : 0.92}
              onMouseEnter={() => setHover(s)}
              onMouseLeave={() => setHover(null)}
            />
          ))}

          {me && (
            <g transform={`translate(${me[0]},${me[1]})`} className="map-me">
              <path d={star(11, 4.6)} fill="var(--text)" stroke="var(--ink)" strokeWidth="2" />
            </g>
          )}
        </svg>

        {hover && (
          <div className="map-tip" style={tipStyle(hover.xy)}>
            <b>{hover.title}</b>
            <span>{hover.mood}</span>
          </div>
        )}
      </div>

      <p className="map-foot">
        An uploaded track is placed between its five nearest corpus songs.
        That lands within about 13% of the map's width of where re-running the
        projection would put it, so read the star as a neighbourhood rather
        than an exact coordinate.
      </p>
    </Panel>
  );
}

function tipStyle([x, y]) {
  return { left: `${(x / W) * 100}%`, top: `${(y / H) * 100}%` };
}

// A five pointed star, as a path, so the marker is not an emoji.
function star(outer, inner) {
  const pts = [];
  for (let i = 0; i < 10; i += 1) {
    const r = i % 2 === 0 ? outer : inner;
    const a = (Math.PI / 5) * i - Math.PI / 2;
    pts.push(`${(Math.cos(a) * r).toFixed(2)},${(Math.sin(a) * r).toFixed(2)}`);
  }
  return `M${pts.join("L")}Z`;
}
