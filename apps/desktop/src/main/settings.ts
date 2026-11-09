import * as fs from 'node:fs/promises';
import path from 'node:path';
import type { Theme } from '@tensor-trail/platform-api';
import { replaceFile } from './files';
import { createHash } from 'node:crypto';

export class Settings {
  theme: Theme = 'dark';
  private file: string;
  private queue = Promise.resolve();
  constructor(readonly directory: string) {
    this.file = path.join(directory, 'settings.json');
  }
  async load() {
    await fs.mkdir(this.directory, { recursive: true });
    try {
      const bytes = await fs.readFile(this.file);
      if (bytes.length < 4096) {
        const value = JSON.parse(bytes.toString('utf8'));
        if (value.theme === 'light' || value.theme === 'dark') this.theme = value.theme;
      }
    } catch {
      /* A missing or invalid preference file uses the default theme. */
    }
  }
  setTheme(theme: Theme) {
    this.theme = theme;
    this.queue = this.queue
      .catch(() => {})
      .then(async () => {
        const old = await fs.readFile(this.file).catch((error) => {
          if (error.code === 'ENOENT') return null;
          throw error;
        });
        const expected = old ? createHash('sha256').update(old).digest('hex') : null;
        await replaceFile(
          this.file,
          JSON.stringify({ version: 1, theme }, null, 2) + '\n',
          expected,
        );
      });
    return this.queue;
  }
  settled() {
    return this.queue;
  }
}
