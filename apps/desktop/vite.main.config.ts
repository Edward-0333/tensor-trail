import { defineConfig } from 'vite';
import { builtinModules } from 'node:module';
export default defineConfig({
  build: {
    target: 'node24',
    outDir: 'dist/main',
    emptyOutDir: true,
    minify: false,
    lib: { entry: 'src/main/index.ts', formats: ['cjs'], fileName: () => 'index.cjs' },
    rollupOptions: {
      external: ['electron', ...builtinModules, ...builtinModules.map((m) => `node:${m}`)],
    },
  },
});
