import React, { useEffect, useMemo, useRef, useState } from "react";
import Panel from "./Panel.jsx";
import { decodePixels } from "../api";
import followPlayhead from "../followPlayhead";
import "./Timeline.css";

const W = 960;
const OVERVIEW_H = 62;
const DETAIL_H = 300;
// Zero, deliberately. The spectrogram canvas spans the panel edge to edge,
// so the curve drawn over it has to use the identical range or the two are
// off by the padding and the shared axis is a lie.
const PAD = { left: 0, right: 0 };

// Spectrogram and energy on one axis.
//
// They cannot share the whole track: the spectrogram is computed over the
// opening window only, because harmonic/percussive separation over a full
// song costs more memory than the host has. So the overview strip carries the
// whole track and marks the window, and the detail view below shows that
// window with the curve drawn over the image it was derived from.
export default function Timeline({ energy, duration, spectrogram, window: win, audioUrl }) {
  const canvasRef = useRef(null);
  const detailRef = useRef(null);
  const pixelsRef = useRef(null);
  const [at, setAt] = useState(null);
  const audioRef = useRef(null);
  const overviewRef = useRef(null);
  // Seconds into the track while the audio plays, else null.
  const [playhead, setPlayhead] = useState(null);

  const curve = energy.curve;
  const windowSeconds = Math.min(win || duration, duration);
  const shown = Math.max(2, Math.round((windowSeconds / duration) * curve.length));
  const detail = useMemo(() => curve.slice(0, shown), [curve, shown]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { width, height, rgb } = spectrogram;
    const bytes = decodePixels(rgb);
    pixelsRef.current = { bytes, width, height };
    const image = new ImageData(width, height);
    for (let i = 0, j = 0; i < width * height; i += 1) {
      image.data[j++] = bytes[i * 3];
      image.data[j++] = bytes[i * 3 + 1];
      image.data[j++] = bytes[i * 3 + 2];
      image.data[j++] = 255;
    }
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d").putImageData(image, 0, 0);
  }, [spectrogram]);

  const plotW = W - PAD.left - PAD.right;
  const xs = (i, n) => PAD.left + (n < 2 ? 0 : (i / (n - 1)) * plotW);
  const ys = (v, h) => 6 + (1 - v) * (h - 12);

  const line = (values, h) =>
    values
      .map((v, i) => `${i === 0 ? "M" : "L"}${xs(i, values.length).toFixed(1)},${ys(v, h).toFixed(1)}`)
      .join("");

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return undefined;
    return followPlayhead(audio, setPlayhead);
  }, [audioUrl]);

  useEffect(() => { setPlayhead(null); }, [audioUrl]);

  // Clicking either view moves the audio there, when there is audio.
  const seek = (seconds) => {
    const audio = audioRef.current;
    if (!audio || !Number.isFinite(seconds)) return;
    audio.currentTime = Math.max(0, seconds);
  };
  const seekOverview = (event) => {
    const box = overviewRef.current?.getBoundingClientRect();
    if (box) seek(((event.clientX - box.left) / box.width) * duration);
  };
  const seekDetail = (event) => {
    const box = detailRef.current?.getBoundingClientRect();
    if (box) seek(((event.clientX - box.left) / box.width) * windowSeconds);
  };

  // Where the playhead falls on each view, or null when it is off that view.
  const headInTrack = playhead !== null && playhead <= duration ? playhead / duration : null;
  const headInDetail = playhead !== null && playhead <= windowSeconds ? playhead / windowSeconds : null;
  const pastAnalysis = playhead !== null && playhead > duration;

  const scrub = (event) => {
    const node = detailRef.current;
    if (!node) return;
    const box = node.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width));
    setAt(ratio);
  };

  // Hovering reads a moment; otherwise the readout follows the playhead.
  const reading = at !== null ? at : headInDetail;

  const probe = useMemo(() => {
    if (reading === null) return null;
    const at = reading;
    const store = pixelsRef.current;
    const i = Math.min(detail.length - 1, Math.round(at * (detail.length - 1)));
    const out = {
      seconds: (at * windowSeconds).toFixed(1),
      energy: detail[i],
    };
    if (store) {
      const x = Math.min(store.width - 1, Math.floor(at * store.width));
      const col = [];
      for (let band = 0; band < 3; band += 1) {
        let total = 0;
        for (let y = 0; y < store.height; y += 1) total += store.bytes[(y * store.width + x) * 3 + band];
        col.push(Math.round(total / store.height));
      }
      [out.r, out.g, out.b] = col;
    }
    return out;
  }, [reading, detail, windowSeconds]);

  return (
    <Panel className="timeline" aria-labelledby="timeline-heading">
      <h2 id="timeline-heading">The track, second by second</h2>
      <p className="note">
        The image is what the model is shown; the line over it is what the
        model returns for each 0.96 second window. Same axis, so a drop in the
        line sits directly under the moment that caused it.
      </p>

      {audioUrl && (
        <div className="tl-player">
          <audio ref={audioRef} src={audioUrl} controls preload="metadata" />
          <span className="tl-hint">
            Play it and watch the cursor. Click the image or the strip to jump.
            {pastAnalysis && ` Past ${Math.round(duration)}s: the model was not shown this part.`}
          </span>
        </div>
      )}

      <div className="tl-overview">
        <span className="tl-label">Whole track, {Math.round(duration)}s</span>
        <svg viewBox={`0 0 ${W} ${OVERVIEW_H}`} className={`tl-over-svg${audioUrl ? " is-seekable" : ""}`} role="img"
          ref={overviewRef} onClick={audioUrl ? seekOverview : undefined}
          aria-label={`Energy across the whole track, mean ${Math.round(energy.score * 100)} percent.`}>
          <line x1={PAD.left} x2={W - PAD.right} y1={ys(0.5, OVERVIEW_H)} y2={ys(0.5, OVERVIEW_H)}
            stroke="var(--sindoor)" strokeDasharray="3 3" strokeWidth="1" opacity="0.55" />
          <rect x={PAD.left} y="4" width={(shown / curve.length) * plotW} height={OVERVIEW_H - 8}
            fill="var(--gold)" opacity="0.13" />
          <path d={line(curve, OVERVIEW_H)} fill="none" stroke="var(--gold)" strokeWidth="1.5" opacity="0.85" />
          {headInTrack !== null && (
            <line x1={PAD.left + headInTrack * plotW} x2={PAD.left + headInTrack * plotW}
              y1="0" y2={OVERVIEW_H} stroke="var(--text)" strokeWidth="1.5" opacity="0.9" />
          )}
        </svg>
        <span className="tl-hint">Shaded: the {Math.round(windowSeconds)}s drawn below</span>
      </div>

      <div
        className="tl-detail"
        ref={detailRef}
        onMouseMove={scrub}
        onMouseLeave={() => setAt(null)}
        onClick={audioUrl ? seekDetail : undefined}
      >
        <canvas ref={canvasRef} className="tl-canvas" aria-hidden="true" />

        <svg viewBox={`0 0 ${W} ${DETAIL_H}`} className="tl-overlay" preserveAspectRatio="none" aria-hidden="true">
          <defs>
            <linearGradient id="tl-fill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--gold-bright)" stopOpacity="0.3" />
              <stop offset="100%" stopColor="var(--gold-bright)" stopOpacity="0" />
            </linearGradient>
          </defs>
          <path d={`${line(detail, DETAIL_H)}L${W - PAD.right},${DETAIL_H}L${PAD.left},${DETAIL_H}Z`} fill="url(#tl-fill)" />
          <path d={line(detail, DETAIL_H)} fill="none" stroke="var(--gold-bright)" strokeWidth="2.5"
            vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
          <line x1={PAD.left} x2={W - PAD.right} y1={ys(0.5, DETAIL_H)} y2={ys(0.5, DETAIL_H)}
            stroke="var(--sindoor)" strokeDasharray="5 5" strokeWidth="1.5" vectorEffect="non-scaling-stroke" opacity="0.8" />
          {headInDetail !== null && (
            <line
              x1={PAD.left + headInDetail * plotW} x2={PAD.left + headInDetail * plotW} y1="0" y2={DETAIL_H}
              stroke="var(--gold-bright)" strokeWidth="2" vectorEffect="non-scaling-stroke"
            />
          )}
          {at !== null && (
            <line
              x1={PAD.left + at * plotW} x2={PAD.left + at * plotW} y1="0" y2={DETAIL_H}
              stroke="#fff" strokeWidth="1.5" vectorEffect="non-scaling-stroke" opacity="0.8"
            />
          )}
        </svg>

        <div className="tl-axis" aria-hidden="true">
          <span>0s</span>
          <span>{Math.round(windowSeconds / 2)}s</span>
          <span>{Math.round(windowSeconds)}s</span>
        </div>
      </div>

      <div className="tl-readout" aria-live="polite">
        {probe ? (
          <>
            <span><em>{probe.seconds}s</em> into the track</span>
            <span>energy <em>{Math.round(probe.energy * 100)}%</em></span>
            {probe.r !== undefined && (
              <>
                <span className="tl-ch r">percussive <em>{probe.r}</em></span>
                <span className="tl-ch g">harmonic <em>{probe.g}</em></span>
                <span className="tl-ch b">mel <em>{probe.b}</em></span>
              </>
            )}
          </>
        ) : (
          <span className="tl-idle">Move across the image to read any moment.</span>
        )}
      </div>
    </Panel>
  );
}
