import React, { useEffect, useState } from "react";
import "./ThemeToggle.css";

const KEY = "melodai-theme";

// Follows the system by default and only pins an explicit choice once the
// reader makes one, so arriving in light mode is not a surprise.
export default function ThemeToggle() {
  const [theme, setTheme] = useState(null);

  useEffect(() => {
    let stored = null;
    try {
      stored = localStorage.getItem(KEY);
    } catch {
      // Private browsing refuses storage; following the system is a fine
      // outcome, so there is nothing to recover from.
    }
    if (stored === "light" || stored === "dark") setTheme(stored);
  }, []);

  useEffect(() => {
    if (!theme) {
      document.documentElement.removeAttribute("data-theme");
      return;
    }
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem(KEY, theme);
    } catch {
      // The attribute above is what actually themes the page; persisting is
      // a convenience.
    }
  }, [theme]);

  const dark =
    theme === "dark" ||
    (theme === null &&
      typeof window !== "undefined" &&
      !window.matchMedia("(prefers-color-scheme: light)").matches);

  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={() => setTheme(dark ? "light" : "dark")}
      aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
    >
      <span aria-hidden="true">{dark ? "☾" : "☀"}</span>
      {dark ? "Paper" : "Ink"}
    </button>
  );
}
