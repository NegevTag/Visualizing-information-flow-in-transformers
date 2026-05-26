import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy `/api?prompt=...` -> `http://127.0.0.1:8000/?prompt=...`
// so the browser doesn't hit CORS and the backend endpoint stays at `/`.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, "/"),
      },
    },
  },
});
