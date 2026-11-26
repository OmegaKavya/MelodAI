// Run with `npm test`. Checks the picker against the server's own lists, so a
// format added or dropped on one side fails here rather than on an upload.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { ACCEPT, UPLOAD_PROMPT } from "./formats.js";

const server = readFileSync(new URL("../../server/main.py", import.meta.url), "utf8");

function serverAllowed() {
  const set = server.match(/^ALLOWED = \{([^}]*)\}/m);
  assert.ok(set, "ALLOWED not found in server/main.py");
  return [...set[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]).sort();
}

function serverFormats() {
  const line = server.match(/^FORMATS = "([^"]*)"/m);
  assert.ok(line, "FORMATS not found in server/main.py");
  return line[1].split(/, | and /).sort();
}

test("the picker offers exactly the extensions the server allows", () => {
  assert.deepEqual(ACCEPT.split(",").sort(), serverAllowed());
});

test("the upload prompt names every format the server reads, and no other", () => {
  const named = UPLOAD_PROMPT.match(/\b[A-Z0-9]{2,}\b/g).sort();
  assert.deepEqual(named, serverFormats());
});
