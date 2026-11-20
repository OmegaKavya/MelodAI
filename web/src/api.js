// One place that knows the server exists.
//
// The client is served from Firebase and the API from a separate free tier
// host, so requests carry an absolute base in production. In development Vite
// proxies /api and the base is empty.
const BASE = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");
async function request(path, options) {
  const response = await fetch(`${BASE}${path}`, options);
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (body && body.detail) detail = body.detail;
    } catch {
      // A non-JSON error body means the server fell over rather than
      // refusing politely; the status line is all we have.
    }
    throw new Error(detail);
  }
  return response.json();
}

// The corpus ships inside the bundle. The map is the finding, and it should
// not be blank for thirty seconds while a sleeping free instance wakes up.
export { default as corpus } from "./corpus.json";

// Kept for parity with the server, and used by nothing on the happy path.
export const fetchCorpus = () => request("/api/corpus");

export function analyse(file) {
  const form = new FormData();
  form.append("file", file);
  return request("/api/analyse", { method: "POST", body: form });
}

// The spectrogram arrives as raw RGB bytes rather than a PNG, so the server
// needs no image encoder and the client can draw it at any size.
export function decodePixels(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}
