import React, { useCallback, useState } from "react";
import { analyse, corpus as bakedCorpus } from "./api";
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
  // Baked into the bundle at build time by scripts/bake_corpus.py, so the
  // map draws on first paint with no request at all.
  const corpus = bakedCorpus;
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

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
