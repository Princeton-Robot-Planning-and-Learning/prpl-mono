import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';

console.log('Application starting...');
console.log('React version:', React.version);

// Check if Plotly is available
import('react-plotly.js').then(() => {
  console.log('react-plotly.js loaded successfully');
}).catch((err) => {
  console.error('Failed to load react-plotly.js:', err);
});

const rootElement = document.getElementById('root');
console.log('Root element:', rootElement);

if (rootElement) {
  const root = ReactDOM.createRoot(rootElement);
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  );
  console.log('Application rendered');
} else {
  console.error('Root element not found!');
}
