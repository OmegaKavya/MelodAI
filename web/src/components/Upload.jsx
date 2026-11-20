import React, { useCallback, useRef, useState } from "react";
import Panel from "./Panel.jsx";
import Progress from "./Progress.jsx";
import "./Upload.css";

const ACCEPT = ".mp3,.wav,.flac,.m4a,audio/*";

// Styled as a cinema ticket because that is the one object in this visual
// language that means "put this in and something happens".
export default function Upload({ onFile, status, error, filename }) {
  const inputRef = useRef(null);
  const [over, setOver] = useState(false);
  const working = status === "working";

  const take = useCallback(
    (file) => {
      if (file) onFile(file);
    },
    [onFile]
  );

  const drop = useCallback(
    (event) => {
      event.preventDefault();
      setOver(false);
      if (!working) take(event.dataTransfer.files?.[0]);
    },
    [take, working]
  );

  return (
    <Panel className="upload" aria-labelledby="upload-heading">
      <h2 id="upload-heading" className="visually-hidden">Upload a song</h2>

      <div
        className={`ticket${over ? " over" : ""}${working ? " working" : ""}`}
        onDragOver={(e) => { e.preventDefault(); if (!working) setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={drop}
      >
        <div className="ticket-stub" aria-hidden="true">
          <span className="deva">सं</span>
        </div>

        <div className="ticket-body">
          <p className="eyebrow">{working ? "Listening" : "Admit one track"}</p>
          <p className="ticket-line">
            {working
              ? "Running YAMNet over every 0.96 second window."
              : filename && status === "done"
              ? filename
              : "Drop an MP3, WAV or FLAC, or choose a file."}
          </p>

          <button
            type="button"
            className="ticket-button"
            onClick={() => inputRef.current?.click()}
            disabled={working}
          >
            {working ? "Working" : filename ? "Choose another" : "Choose a file"}
          </button>

          <input
            ref={inputRef}
            type="file"
            accept={ACCEPT}
            className="visually-hidden"
            onChange={(e) => { take(e.target.files?.[0]); e.target.value = ""; }}
          />
        </div>
      </div>

      {working && <Progress />}

      {status === "failed" && (
        <p className="upload-error" role="alert">{error}</p>
      )}

      <p className="upload-small">
        Audio is analysed and discarded. Nothing is stored, and the first ninety
        seconds are all that is read.
      </p>
    </Panel>
  );
}
