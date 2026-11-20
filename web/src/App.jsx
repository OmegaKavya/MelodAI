import React, { useCallback, useEffect, useState } from "react";
import { analyse, fetchCorpus } from "./api";
import Hero from "./components/Hero.jsx";
import Upload from "./components/Upload.jsx";
import Verdict from "./components/Verdict.jsx";
import EnergyCurve from "./components/EnergyCurve.jsx";
import Hears from "./components/Hears.jsx";
import Neighbours from "./components/Neighbours.jsx";
import MoodMap from "./components/MoodMap.jsx";
import SpectrogramView from "./components/SpectrogramView.jsx";
import Method from "./components/Method.jsx";
import "./App.css";

export default function App() {
  const [corpus, setCorpus] = useState([]);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

  useEffect(() => {
    // The map is worth drawing before anyone uploads anything: it is the
    // finding, not a frame for the upload.
    fetchCorpus()
      .then((data) => setCorpus(data.songs))
      .catch(() => setCorpus([]));
  }, []);

  const run = useCallback(async (file) => {
    setStatus("working");
    setError("");
    try {
      const data = await analyse(file);
      setResult(data);
      setStatus("done");
    } catch (failure) {
      setError(failure.message);
      setStatus("failed");
    }
  }, []);

  return (
    <>
      <Hero />
      <main className="shell">
        <Upload onFile={run} status={status} error={error} filename={result?.title} />

        {result && (
          <div className="results">
            <Verdict energy={result.energy} windows={result.windows} duration={result.duration} />
            <EnergyCurve energy={result.energy} duration={result.duration} />
            <div className="two-up">
              <Hears hears={result.hears} />
              <Neighbours neighbours={result.neighbours} />
            </div>
            <SpectrogramView
              spectrogram={result.spectrogram}
              duration={result.spectrogram_seconds ?? result.duration}
              full={result.duration}
            />
            <MoodMap corpus={corpus} position={result.position} />
          </div>
        )}

        {!result && corpus.length > 0 && <MoodMap corpus={corpus} position={null} />}

        <Method />
      </main>
      <footer className="footer">
        <hr className="rule" />
        <p>
          Built by Kavya Aggarwal. <a href="https://github.com/OmegaKavya/MelodAI">Source and method</a>
        </p>
      </footer>
    </>
  );
}
