import { readFileSync } from 'node:fs';
import type { Plugin } from 'vite';

export function fontLicense(): Plugin {
  return {
    name: 'tensor-trail-font-license',
    generateBundle() {
      this.emitFile({
        type: 'asset',
        fileName: 'assets/Figtree-OFL.txt',
        source: readFileSync(new URL('./src/assets/fonts/OFL.txt', import.meta.url), 'utf8'),
      });
    },
  };
}
