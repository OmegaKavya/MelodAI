// Run with `npm test`. Plain node:test, so the web app gains no dependency.
import test from "node:test";
import assert from "node:assert/strict";
import followPlayhead from "./followPlayhead.js";

// A frame clock that runs only when told to, and an audio element that only
// knows the events the loop listens for.
function fakeFrames() {
  let pending = new Map();
  let id = 0;
  return {
    requestAnimationFrame: (callback) => { pending.set(++id, callback); return id; },
    cancelAnimationFrame: (handle) => { pending.delete(handle); },
    pending: () => pending.size,
    runFrame() {
      const due = pending;
      pending = new Map();
      for (const callback of due.values()) callback();
    },
  };
}

function fakeAudio() {
  const listeners = {};
  return {
    currentTime: 0,
    paused: true,
    ended: false,
    addEventListener(type, listener) { listeners[type] = listener; },
    removeEventListener(type) { delete listeners[type]; },
    emit(type) { listeners[type]?.(); },
    listening: () => Object.keys(listeners).length,
  };
}

test("seeking during playback keeps exactly one frame loop", () => {
  const frames = fakeFrames();
  const audio = fakeAudio();
  let calls = 0;
  followPlayhead(audio, () => { calls += 1; }, frames);

  audio.paused = false;
  audio.emit("play");
  frames.runFrame();
  for (let i = 0; i < 5; i += 1) {
    audio.currentTime += 3;
    audio.emit("seeked");
    frames.runFrame();
  }

  calls = 0;
  frames.runFrame();
  assert.equal(frames.pending(), 1);
  assert.equal(calls, 1);
});

test("a seek while paused moves the cursor without starting a loop", () => {
  const frames = fakeFrames();
  const audio = fakeAudio();
  const seen = [];
  followPlayhead(audio, (t) => seen.push(t), frames);

  audio.currentTime = 12;
  audio.emit("seeked");
  assert.deepEqual(seen, [12]);
  assert.equal(frames.pending(), 0);
});

test("cleanup cancels the loop and removes every listener", () => {
  const frames = fakeFrames();
  const audio = fakeAudio();
  const stop = followPlayhead(audio, () => {}, frames);

  audio.paused = false;
  audio.emit("play");
  stop();
  assert.equal(frames.pending(), 0);
  assert.equal(audio.listening(), 0);
});
