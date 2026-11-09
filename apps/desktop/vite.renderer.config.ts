import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({
  plugins: [react()],
  base: './',
  server: { host: '127.0.0.1', port: 5180, strictPort: true },
  build: { target: 'es2022', outDir: 'dist/renderer', emptyOutDir: true },
});
