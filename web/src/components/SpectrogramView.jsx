import React, { useEffect, useRef, useState } from "react";
import Panel from "./Panel.jsx";
import { decodePixels } from "../api";
import "./SpectrogramView.css";

const CHANNELS = [
  { key: "r", label: "Red", meaning: "Percussive energy", css: "var(--sindoor)" },
  { key: "g", label: "Green", meaning: "Harmonic content and chroma", css: "#3fa96b" },
  { key: "b", label: "Blue", meaning: "Mel bands and spectral contrast", css: "var(--teal)" },
];

// The picture the model was trained on, drawn from the raw array the server
// sent. Drawing it here rather than receiving a PNG keeps an image encoder
// out of the server and lets the reader interrogate it with a cursor.
export default function SpectrogramView({ spectrogram, duration, full }) {
  const canvasRef = useRef(null);
  const [probe, setProbe] = useState(null);
  const pixelsRef = useRef(null);

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

  const move = (event) => {
    const canvas = canvasRef.current;
    const store = pixelsRef.current;
    if (!canvas || !store) return;
    const box = canvas.getBoundingClientRect();
    const x = Math.floor(((event.clientX - box.left) / box.width) * store.width);
    const y = Math.floor(((event.clientY - box.top) / box.height) * store.height);
    if (x < 0 || y < 0 || x >= store.width || y >= store.height) return setProbe(null);
    const i = (y * store.width + x) * 3;
    setProbe({
      seconds: ((x / store.width) * duration).toFixed(1),
      // Row 0 is the top of the canvas and the top of the spectrum.
      share: 1 - y / store.height,
      r: store.bytes[i], g: store.bytes[i + 1], b: store.bytes[i + 2],
    });
  };

  return (
    <Panel className="spectro" aria-labelledby="spectro-heading">
      <h2 id="spectro-heading">Your song, as the model sees it</h2>
      <p className="note">
        Not a picture of the waveform. Three separated representations stacked
        into one image, which is the form the convolutional model was trained
        on. Time runs left to right, pitch bottom to top.
        {full > duration + 1 && (
          <>
            {" "}Showing the first {Math.round(duration)} seconds, the segment
            length the model was trained on.
          </>
        )}
      </p>

      <canvas
        ref={canvasRef}
        className="spectro-canvas"
        onMouseMove={move}
        onMouseLeave={() => setProbe(null)}
        aria-label="Three channel spectrogram of the uploaded track."
        role="img"
      />

      <ul className="spectro-key">
        {CHANNELS.map((c) => (
          <li key={c.key}>
            <span className="swatch" style={{ background: c.css }} aria-hidden="true" />
            <b>{c.label}</b>
            <span>{c.meaning}</span>
            {probe && <em>{probe[c.key]}</em>}
          </li>
        ))}
      </ul>

      <p className="spectro-readout" aria-live="polite">
        {probe
          ? `At ${probe.seconds}s, ${Math.round(probe.share * 100)}% of the way up the spectrum.`
          : "Hover the image to read a point."}
      </p>
    </Panel>
  );
}
