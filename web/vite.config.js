import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API runs separately in development. Proxying it here means the client
// calls /api in both environments and never carries a base URL.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://localhost:8711" },
  },
});
