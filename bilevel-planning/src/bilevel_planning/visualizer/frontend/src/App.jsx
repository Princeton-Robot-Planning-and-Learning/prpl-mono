import React, { useState } from 'react';
import { GraphViewer3D } from './components/GraphViewer3D';

const DEFAULT_RENDERER_SOURCE = `# Define a callable named render_state(state) that returns an
# HxWx3 uint8 numpy array. Any package in the backend's Python
# environment can be imported.
import numpy as np

def render_state(state):
    return np.full((256, 256, 3), 128, dtype=np.uint8)
`;

function App() {
  const [graphData, setGraphData] = useState(null);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState('unknown');
  const [pickleLoaded, setPickleLoaded] = useState(false);
  const [picklePath, setPicklePath] = useState('');
  const [rendererSource, setRendererSource] = useState(DEFAULT_RENDERER_SOURCE);
  const [rendererSet, setRendererSet] = useState(false);
  const [rendererStatus, setRendererStatus] = useState(null);

  // Check backend health on mount and periodically
  React.useEffect(() => {
    checkBackendHealth();
    // Poll backend health every 30 seconds
    const interval = setInterval(checkBackendHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkBackendHealth = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/health');
      if (response.ok) {
        const data = await response.json();
        setBackendStatus('connected');
        setPickleLoaded(data.graph_loaded);
        setRendererSet(data.renderer_set);
      } else {
        setBackendStatus('error');
        setPickleLoaded(false);
        setRendererSet(false);
      }
    } catch (err) {
      setBackendStatus('disconnected');
      setPickleLoaded(false);
      setRendererSet(false);
    }
  };

  const handleApplyRenderer = async () => {
    setRendererStatus(null);
    try {
      const response = await fetch('http://localhost:5001/api/set_renderer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: rendererSource }),
      });
      const data = await response.json();
      if (!response.ok) {
        setRendererStatus({ ok: false, message: data.error || 'Failed to install renderer' });
        setRendererSet(false);
        return;
      }
      setRendererStatus({ ok: true, message: 'Renderer installed.' });
      setRendererSet(true);
    } catch (err) {
      setRendererStatus({ ok: false, message: `Error: ${err.message}` });
      setRendererSet(false);
    }
  };

  const handleLoadPickleByPath = async () => {
    if (!picklePath.trim()) {
      setError('Please enter a pickle file path');
      return;
    }
    await loadPickleByPath(picklePath);
  };

  const loadPickleByPath = async (path) => {
    try {
      const response = await fetch('http://localhost:5001/api/load_pickle', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ pickle_path: path })
      });

      const data = await response.json();

      if (!response.ok) {
        console.error('Error loading pickle:', data.error);
        let errorMsg = `Failed to load pickle: ${data.error}`;
        if (data.attempted_paths) {
          errorMsg += `\nAttempted paths:\n${data.attempted_paths.join('\n')}`;
        }
        setError(errorMsg);
        setGraphData(null);
        setPickleLoaded(false);
        return;
      }

      // Backend loaded the bundle; fetch the topology for the viewer.
      const graphResp = await fetch('http://localhost:5001/api/graph');
      if (!graphResp.ok) {
        const graphErr = await graphResp.json();
        setError(`Failed to fetch graph: ${graphErr.error}`);
        setGraphData(null);
        setPickleLoaded(false);
        return;
      }
      const graphJson = await graphResp.json();

      if (!graphJson.nodes || !graphJson.edges || !graphJson.config) {
        setError('Backend returned a graph with an unexpected shape');
        setGraphData(null);
        setPickleLoaded(false);
        return;
      }

      console.log(
        `Loaded ${data.num_states} states; graph has ${graphJson.nodes.length} nodes, ${graphJson.edges.length} edges`
      );
      setGraphData(graphJson);
      setPickleLoaded(true);
      setError(null);
      await checkBackendHealth();
    } catch (err) {
      console.error('Error loading pickle:', err);
      setError(`Error: ${err.message}`);
    }
  };


  return (
    <div style={styles.app}>
      <header style={styles.header}>
        <h1 style={styles.title}>Bilevel Planning Graph Viewer</h1>
        <div style={styles.controls}>
          <div style={styles.pathInputContainer}>
            <input
              type="text"
              value={picklePath}
              onChange={(e) => setPicklePath(e.target.value)}
              placeholder="Enter pickle path (e.g., test_o4o4_state_data_0.pkl)"
              style={{
                ...styles.pathInput,
                opacity: backendStatus === 'connected' ? 1 : 0.5,
                cursor: backendStatus === 'connected' ? 'text' : 'not-allowed',
              }}
              disabled={backendStatus !== 'connected'}
              title="Tip: Right-click file in Finder, Copy as Pathname, then paste here"
            />
            <button
              onClick={handleLoadPickleByPath}
              style={{...styles.button, backgroundColor: '#9C27B0'}}
              disabled={backendStatus !== 'connected'}
              title="Load pickle file from path"
            >
              Load Pickle (Backend)
            </button>
          </div>
        </div>
        <details style={styles.rendererDetails}>
          <summary style={styles.rendererSummary}>
            Python renderer {rendererSet ? '(installed)' : '(not installed)'}
          </summary>
          <div style={styles.rendererPane}>
            <textarea
              value={rendererSource}
              onChange={(e) => setRendererSource(e.target.value)}
              style={styles.rendererTextarea}
              spellCheck={false}
              disabled={backendStatus !== 'connected'}
            />
            <div style={styles.rendererControls}>
              <button
                onClick={handleApplyRenderer}
                style={{...styles.button, backgroundColor: '#4CAF50'}}
                disabled={backendStatus !== 'connected'}
              >
                Apply renderer
              </button>
              {rendererStatus && (
                <span
                  style={{
                    ...styles.rendererStatus,
                    color: rendererStatus.ok ? '#4CAF50' : '#f44336',
                  }}
                >
                  {rendererStatus.message}
                </span>
              )}
            </div>
          </div>
        </details>
        {error && (
          <div style={styles.error}>
            {error}
          </div>
        )}
      </header>

      <main style={styles.main}>
        <GraphViewer3D
          graphData={graphData}
          pickleLoaded={pickleLoaded && rendererSet}
        />
      </main>
    </div>
  );
}

const styles = {
  app: {
    width: '100%',
    height: '100vh',
    display: 'flex',
    flexDirection: 'column',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  },
  header: {
    position: 'relative',
    zIndex: 3000, 
    padding: '15px 20px',
    backgroundColor: '#282c34',
    color: 'white',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
  },
  title: {
    margin: '0 0 10px 0',
    fontSize: '24px',
    fontWeight: '600',
  },
  statusBar: {
    marginBottom: '10px',
    fontSize: '14px',
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  statusBadge: {
    padding: '2px 8px',
    borderRadius: '12px',
    fontSize: '12px',
    fontWeight: '500',
    marginLeft: '5px',
  },
  pickleStatus: {
    color: '#4CAF50',
    fontSize: '12px',
    marginLeft: '10px',
  },
  controls: {
    display: 'flex',
    gap: '15px',
    alignItems: 'center',
    flexWrap: 'wrap',
  },
  button: {
    padding: '8px 16px',
    backgroundColor: '#2196F3',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: '500',
  },
  pathInputContainer: {
    display: 'flex',
    gap: '10px',
    alignItems: 'center',
    flex: 1,
    minWidth: '300px',
  },
  pathInput: {
    flex: 1,
    padding: '8px 12px',
    fontSize: '14px',
    borderRadius: '4px',
    border: '1px solid #444',
    backgroundColor: '#1a1d24',
    color: 'white',
  },
  helpText: {
    fontSize: '12px',
    color: '#aaa',
    fontStyle: 'italic',
    marginTop: '5px',
  },
  error: {
    marginTop: '10px',
    padding: '10px',
    backgroundColor: '#f44336',
    color: 'white',
    borderRadius: '4px',
    fontSize: '14px',
  },
  rendererDetails: {
    marginTop: '10px',
    fontSize: '13px',
  },
  rendererSummary: {
    cursor: 'pointer',
    padding: '4px 0',
    userSelect: 'none',
  },
  rendererPane: {
    marginTop: '8px',
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  rendererTextarea: {
    width: '100%',
    minHeight: '140px',
    padding: '10px',
    fontFamily: 'Menlo, Monaco, "Courier New", monospace',
    fontSize: '13px',
    color: 'white',
    backgroundColor: '#1a1d24',
    border: '1px solid #444',
    borderRadius: '4px',
    resize: 'vertical',
  },
  rendererControls: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  rendererStatus: {
    fontSize: '13px',
  },
  main: {
    flex: 1,
    overflow: 'hidden',
  },
};

export default App;
