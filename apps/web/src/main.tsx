import React from 'react';
import ReactDOM from 'react-dom/client';
import { App, configureEditor } from '@tensor-trail/editor-ui';
import { createBrowserPlatform } from './platform/browser';

const dispose = configureEditor(createBrowserPlatform());
if (import.meta.hot) import.meta.hot.dispose(dispose);
ReactDOM.createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>);
