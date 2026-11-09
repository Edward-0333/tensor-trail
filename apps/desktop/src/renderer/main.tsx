import React from 'react';
import ReactDOM from 'react-dom/client';
import { App, configureEditor } from '@tensor-trail/editor-ui';
import { desktopPlatform } from './platform';

async function main() {
  const platform = await desktopPlatform();
  const dispose = configureEditor(platform);
  if (import.meta.hot) import.meta.hot.dispose(dispose);
  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  );
}
void main().catch((error) => {
  document.getElementById('root')!.textContent = `桌面界面启动失败：${error.message}`;
});
