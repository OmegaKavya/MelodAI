// Follow the audio at frame rate rather than on `timeupdate`, which fires
// about four times a second and makes the cursor jump.
//
// Kept out of the component so the loop can be run in Node without a browser:
// `frames` stands in for the window's requestAnimationFrame pair.
export default function followPlayhead(audio, onTime, frames = globalThis) {
  let frame = 0;
  const tick = () => {
    onTime(audio.currentTime);
    if (!audio.paused && !audio.ended) frame = frames.requestAnimationFrame(tick);
  };
  const start = () => { frames.cancelAnimationFrame(frame); frame = frames.requestAnimationFrame(tick); };
  const stop = () => { frames.cancelAnimationFrame(frame); onTime(audio.currentTime || null); };
  // A seek during playback lands while a frame is already pending. Ticking
  // straight from here would schedule a second one, and every later seek
  // another, each loop redrawing the cursor on every frame.
  const seeked = () => { frames.cancelAnimationFrame(frame); tick(); };
  audio.addEventListener("play", start);
  audio.addEventListener("pause", stop);
  audio.addEventListener("ended", stop);
  audio.addEventListener("seeked", seeked);
  return () => {
    frames.cancelAnimationFrame(frame);
    audio.removeEventListener("play", start);
    audio.removeEventListener("pause", stop);
    audio.removeEventListener("ended", stop);
    audio.removeEventListener("seeked", seeked);
  };
}
