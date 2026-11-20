// One place that knows the server exists.
async function request(path, options) {
  const response = await fetch(path, options);
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
