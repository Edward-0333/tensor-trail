// Run with Electron: electron scripts/export-desktop-icons.cjs.
// Uses Chromium's SVG renderer; no new image-processing dependency is needed.
const { app, BrowserWindow } = require('electron');
const fs = require('node:fs/promises');
const path = require('node:path');
const directory = path.resolve(__dirname, '../apps/desktop/resources');
app.setPath('userData', path.resolve(__dirname, '../test-results/icon-export-profile'));
const timeout = setTimeout(() => app.exit(1), 20000);
async function run() {
  await app.whenReady();
  const window = new BrowserWindow({
    show: false,
    webPreferences: { sandbox: true, contextIsolation: true, nodeIntegration: false },
  });
  await window.loadURL('data:text/html,<meta charset="utf-8">');
  const svg = await fs.readFile(path.join(directory, 'tensor-trail.svg'), 'utf8');
  const sizes = [16, 24, 32, 48, 64, 128, 256];
  const images = [];
  for (const size of sizes) {
    const encoded = await window.webContents.executeJavaScript(
      `(async()=>{const image=new Image();image.src=${JSON.stringify('data:image/svg+xml;base64,' + Buffer.from(svg).toString('base64'))};await image.decode();const canvas=document.createElement('canvas');canvas.width=canvas.height=${size};canvas.getContext('2d').drawImage(image,0,0,${size},${size});return canvas.toDataURL('image/png').split(',')[1];})()`,
    );
    images.push(Buffer.from(encoded, 'base64'));
  }
  const header = Buffer.alloc(6 + sizes.length * 16);
  header.writeUInt16LE(1, 2);
  header.writeUInt16LE(sizes.length, 4);
  let offset = header.length;
  sizes.forEach((size, i) => {
    const entry = 6 + i * 16;
    header[entry] = header[entry + 1] = size === 256 ? 0 : size;
    header.writeUInt16LE(1, entry + 4);
    header.writeUInt16LE(32, entry + 6);
    header.writeUInt32LE(images[i].length, entry + 8);
    header.writeUInt32LE(offset, entry + 12);
    offset += images[i].length;
  });
  await fs.writeFile(path.join(directory, 'tensor-trail.ico'), Buffer.concat([header, ...images]));
  await fs.writeFile(path.join(directory, 'tensor-trail.png'), images.at(-1));
  console.log(`Exported Tensor Trail PNG and ICO (${sizes.join(', ')} px).`);
  clearTimeout(timeout);
  window.destroy();
  app.quit();
}
void run().catch((error) => {
  console.error(error);
  clearTimeout(timeout);
  app.exit(1);
});
