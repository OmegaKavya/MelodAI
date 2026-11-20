import React from "react";
import ThemeToggle from "./ThemeToggle.jsx";
import "./Hero.css";

// The title card. Poster conventions on purpose: the Devanagari word for
// music set above a gold title, ornament rules either side, and a one line
// claim underneath that says what the thing actually does.
export default function Hero() {
  return (
    <header className="hero">
      <ThemeToggle />
      <div className="hero-inner shell">
        <p className="hero-deva deva" aria-hidden="true">संगीत</p>

        <div className="hero-rule" aria-hidden="true">
          <span /><i /><span />
        </div>

        <h1>
          <span className="foil">MelodAI</span>
        </h1>

        <p className="hero-tag">
          A neural network listens to a Hindi film song and tells you what it
          hears. It is right about energy and wrong about feeling, and this
          page shows you both.
        </p>

        <dl className="hero-stats">
          <div>
            <dt>Energy, held out</dt>
            <dd>26<span>/26</span></dd>
          </div>
          <div>
            <dt>Mood, as first built</dt>
            <dd>0<span>/9</span></dd>
          </div>
          <div>
            <dt>Arousal vs valence, k-NN</dt>
            <dd>85<span>% / 45%</span></dd>
          </div>
        </dl>
      </div>
    </header>
  );
}
