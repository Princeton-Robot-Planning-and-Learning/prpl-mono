import React, { useState } from 'react';
import { GraphViewer3D } from './components/GraphViewer3D';

function App() {
  const [graphData, setGraphData] = useState(null);
  const [error, setError] = useState(null);

  // The backend is launched with the bundle and renderer already loaded, so
  // we just fetch the graph topology on mount and render it.
  React.useEffect(() => {
    const loadGraph = async () => {
      try {
        const response = await fetch('/api/graph');
        if (!response.ok) {
          const data = await response.json();
          setError(`Failed to fetch graph: ${data.error}`);
          return;
        }
        const graphJson = await response.json();
        if (!graphJson.nodes || !graphJson.edges || !graphJson.config) {
          setError('Backend returned a graph with an unexpected shape');
          return;
        }
        setGraphData(graphJson);
      } catch (err) {
        setError(`Error: ${err.message}. Is the visualizer backend running?`);
      }
    };
    loadGraph();
  }, []);

  return (
    <div style={styles.app}>
      {error && <div style={styles.error}>{error}</div>}
      <main style={styles.main}>
        <GraphViewer3D graphData={graphData} />
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
  error: {
    padding: '10px 20px',
    backgroundColor: '#f44336',
    color: 'white',
    fontSize: '14px',
  },
  main: {
    flex: 1,
    overflow: 'hidden',
  },
};

export default App;
