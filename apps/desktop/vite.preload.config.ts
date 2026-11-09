import { defineConfig } from 'vite';
export default defineConfig({
  build: {
    target: 'es2022',
    outDir: 'dist/preload',
    emptyOutDir: true,
    minify: false,
    lib: { entry: 'src/preload/index.ts', formats: ['cjs'], fileName: () => 'index.cjs' },
    rollupOptions: { external: ['electron'] },
  },
});
