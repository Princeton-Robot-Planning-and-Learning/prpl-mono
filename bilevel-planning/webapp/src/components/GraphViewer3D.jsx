import React, { useState, useCallback, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { jsonToPlotlyTraces, getPlotlyLayout, getPlotlyConfig } from '../utils/plotlyHelpers';

export function GraphViewer3D({ graphData, pickleLoaded = false }) {
  const [selectedNode, setSelectedNode] = useState(null);
  const [renderError, setRenderError] = useState(null);
  const [stateImage, setStateImage] = useState(null);
  const [loadingImage, setLoadingImage] = useState(false);
  const [cameraState, setCameraState] = useState(null); // Store camera position
  const [currentTime, setCurrentTime] = useState(null); // Timeline slider position
  const [playbackSpeed, setPlaybackSpeed] = useState(200); // ms per step
  const [playDirection, setPlayDirection] = useState(null); // null, 'next', 'prev'

  // Auto-stepping effect
  React.useEffect(() => {
    if (!playDirection) return;

    const step = () => {
      setCurrentTime(prevTime => {
        const current = prevTime ?? (graphData?.config?.min_time ?? 0);
        if (playDirection === 'next') {
           const max = graphData?.config?.max_time ?? 100;
           return Math.min(max, current + 1);
        } else {
           const min = graphData?.config?.min_time ?? 0;
           return Math.max(min, current - 1);
        }
      });
    };

    // Execute immediately so that a quick click triggers at least one step
    step(); 
    
    const id = setInterval(step, playbackSpeed);
    return () => clearInterval(id);
  }, [playDirection, playbackSpeed, graphData]);
  
  // Log prop changes
  React.useEffect(() => {
    console.log('GraphViewer3D props updated:', { 
      hasGraphData: !!graphData, 
      pickleLoaded,
      nodeCount: graphData?.nodes?.length,
      edgeCount: graphData?.edges?.length
    });
  }, [graphData, pickleLoaded]);
  
  // Reset state when graphData changes (e.g., loading a new graph)
  React.useEffect(() => {
    console.log('GraphData changed - resetting viewer state');
    // Don't reset selectedNode - keep the info panel always visible
    setStateImage(null);
    setLoadingImage(false);
    setRenderError(null);
    setCameraState(null);
    // Initialize timeline to min_time if available
    if (graphData && graphData.config && typeof graphData.config.min_time === 'number') {
      setCurrentTime(graphData.config.min_time);
    } else {
      setCurrentTime(null);
    }
  }, [graphData]);
  
  // Hooks must be called unconditionally, before any early returns
  const handleClick = useCallback(async (event) => {
    console.log('handleClick triggered');
    console.log('pickleLoaded:', pickleLoaded);
    
    if (event.points && event.points.length > 0) {
      const point = event.points[0];
      console.log('Clicked point:', point);

      if (point.customdata) {
        const nodeId = point.customdata;
        console.log('Node clicked:', nodeId);
        
        // If it's a concrete node and we have state data, log the state
        if (nodeId.startsWith('x:')) {
          const stateData = graphData?.state_data?.[nodeId] || 'No state data available';
          console.log('Concrete state data:', stateData);
          console.log('Full state object:', stateData);
          
          // Always set selected node for concrete nodes
          setSelectedNode({ id: nodeId, stateData });
          
          // Check if pickle is loaded before requesting visualization
          if (!pickleLoaded) {
            console.warn('Pickle not loaded - showing error message');
            const msg = 'Please load a pickle file first to visualize states.\n\nUse "Load Pickle (Backend)" button above.';
            setRenderError(msg);
            alert(msg); // Force alert to ensure user sees it
            setStateImage(null);
            setLoadingImage(false);
            return;
          }
          
          // Clear previous image and error
          setStateImage(null);
          setRenderError(null);
          
          // Request visualization from backend
          setLoadingImage(true);
          try {
            console.log('Requesting visualization for node:', nodeId);
            const response = await fetch('http://localhost:5001/api/visualize_state', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({ node_id: nodeId }),
            });
            
            console.log('Response status:', response.status, response.statusText);
            
            if (response.ok) {
              const data = await response.json();
              console.log('Visualization received, image length:', data.image?.length);
              setStateImage({
                src: data.image,
                width: data.width,
                height: data.height
              });
              setRenderError(null); // Clear any previous errors
            } else {
              // Try to parse error as JSON
              const contentType = response.headers.get('content-type');
              console.log('Error response content-type:', contentType);
              
              if (contentType && contentType.includes('application/json')) {
                const error = await response.json();
                console.error('Failed to get visualization (JSON error):', error);
                const errorMsg = `Visualization failed: ${error.error || 'Unknown error'}`;
                setRenderError(errorMsg);
                console.error('Full error:', error);
              } else {
                // Not JSON - might be HTML error page or empty
                const text = await response.text();
                console.error('Failed to get visualization (non-JSON response):', text);
                const errorMsg = `Visualization failed: ${response.status} ${response.statusText}`;
                setRenderError(errorMsg);
              }
              setStateImage(null);
            }
          } catch (error) {
            console.error('Error fetching visualization:', error);
            const errorMsg = `Error: ${error.message}. Make sure backend is running on http://localhost:5001`;
            setRenderError(errorMsg);
            setStateImage(null);
          } finally {
            setLoadingImage(false);
          }
        } else if (nodeId.startsWith('s:')) {
          console.log('Abstract state clicked');
          setSelectedNode({ id: nodeId, stateData: null });
          setStateImage(null);
          setRenderError(null);
        } else {
          setSelectedNode({ id: nodeId, stateData: null });
          setStateImage(null);
          setRenderError(null);
        }
      } else {
        console.warn('Clicked point has no customdata:', point);
      }
    }
  }, [graphData, pickleLoaded]);
  
  /**
   * Memoize traces and layouts
   * This prevents expensive re-calculates on every component re-render.
   */
  const { traces, layout, config, error } = useMemo(() => {
    // Early return if no data is available yet
    if (!graphData) {
      return { traces: null, layout: null, config: null, error: null };
    }
    
    try {
      console.log('Rendering graph with data:', graphData);
      
      // Create working copy to apply overlays wo modifying original data
      let graphDataCopy = graphData;
      
      // Apply time-based coloring overlay (if timeline info available)
      if (currentTime !== null) {
        const recoloredNodes = graphDataCopy.nodes.map(node => {
          // Only recolor concrete nodes that have valid time information
          if (node.type !== 'concrete' || typeof node.time_index !== 'number') {
            return node;
          }

          const baseColor = node.color;
          if (!baseColor || typeof baseColor !== 'string') {
            return node;
          }

          // Internal helper to darken an rgb(r, g, b) string by a scaling factor.
          const darkenRgb = (colorStr, factor = 0.6) => {
            const match = colorStr.match(/rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)/i);
            if (!match) return colorStr;
            const r = Math.max(0, Math.min(255, Math.round(parseInt(match[1], 10) * factor)));
            const g = Math.max(0, Math.min(255, Math.round(parseInt(match[2], 10) * factor)));
            const b = Math.max(0, Math.min(255, Math.round(parseInt(match[3], 10) * factor)));
            return `rgb(${r}, ${g}, ${b})`;
          };

          let newColor = baseColor;
          let newAlpha = node.alpha;

          if (node.time_index === currentTime) {
            // Highlighting: The current node becomes bright red.
            newColor = 'rgb(255, 0, 0)';
            newAlpha = 1.0;
          } else if (node.time_index < currentTime) {
            // Past: Visited nodes are darkened to show progress through the graph.
            newColor = darkenRgb(baseColor, 0.6);
          }
          // Note: Nodes in the future (index > currentTime) retain their default colors.

          return {
            ...node,
            color: newColor,
            alpha: newAlpha,
          };
        });

        graphDataCopy = {
          ...graphDataCopy,
          nodes: recoloredNodes,
        };
      }
      
      // Convert our working copy into the specific Trace/Layout data 
      // format expected by the Plotly library.
      const traces = jsonToPlotlyTraces(graphDataCopy);
      console.log('Generated traces:', traces.length);
      
      const layout = getPlotlyLayout(graphData);
      
      // UI Persistence: maintain camera view
      if (cameraState) {
        layout.scene.camera = cameraState;
      }
      
      const config = getPlotlyConfig();
      
      return { traces, layout, config, error: null };
    } catch (err) {
      console.error('Error generating Plotly data:', err);
      return { traces: null, layout: null, config: null, error: err };
    }
  }, [graphData, cameraState, currentTime]);
  
  // Force complete remount when graph changes by using revision in layout
  const [revision, setRevision] = useState(0);
  React.useEffect(() => {
    if (graphData) {
      setRevision(prev => prev + 1);
    }
  }, [graphData]); // Only increment when graphData changes, not on every click
  
  // Now we can do early returns after all hooks are called
  if (!graphData) {
    return (
      <div style={styles.container}>
        <div style={styles.message}>
          <h2>No graph data loaded</h2>
          <p>Please load a graph JSON file using the file input above.</p>
        </div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div style={styles.container}>
        <div style={styles.errorBox}>
          <h2>Error rendering graph</h2>
          <p>{error.message}</p>
          <pre style={{ fontSize: '12px', overflow: 'auto' }}>{error.stack}</pre>
        </div>
      </div>
    );
  }
  
  return (
    <div style={styles.container}>
      {/* Control Panel - Always visible */}
      <div style={{
        position: 'absolute',
        bottom: '20px',
        right: '20px',
        padding: '10px',
        backgroundColor: 'rgba(255,255,255,0.95)',
        border: '1px solid #ccc',
        borderRadius: '4px',
        fontSize: '12px',
        zIndex: 2000,
        boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
        minWidth: '200px',
      }}>
        {/* Timeline slider*/}
        <div style={{ marginBottom: '8px', paddingBottom: '8px', borderBottom: '1px solid #ccc', padding: '5px' }}>
          <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>
            Timeline: {currentTime !== null ? currentTime : 'N/A'} / {graphData?.config?.max_time ?? '?'}
          </div>
          <input
            type="range"
            min={graphData?.config?.min_time ?? 0}
            max={graphData?.config?.max_time ?? 100}
            step={1}
            value={currentTime !== null ? currentTime : (graphData?.config?.min_time ?? 0)}
            onChange={(e) => setCurrentTime(Number(e.target.value))}
            style={{ width: '100%', cursor: 'pointer' }}
          />
          
          {/* Step buttons */}
          <div style={{ display: 'flex', gap: '5px', marginTop: '5px' }}>
            <button 
              onMouseDown={() => setPlayDirection('prev')}
              onMouseUp={() => setPlayDirection(null)}
              onMouseLeave={() => { if (playDirection === 'prev') setPlayDirection(null); }}
              style={{ flex: 1, cursor: 'pointer', padding: '2px', fontSize: '11px' }}
              title="Previous Step"
            >
              &lt; Prev
            </button>
            <button 
              onMouseDown={() => setPlayDirection('next')}
              onMouseUp={() => setPlayDirection(null)}
              onMouseLeave={() => { if (playDirection === 'next') setPlayDirection(null); }}
              style={{ flex: 1, cursor: 'pointer', padding: '2px', fontSize: '11px' }}
              title="Next Step"
            >
              Next &gt;
            </button>
            <button 
              onClick={() => setPlayDirection('next')}
              style={{ flex: 1, cursor: 'pointer', padding: '2px', fontSize: '11px', backgroundColor: playDirection === 'next' ? '#e6fffa' : '' }}
              title="Auto-Play"
            >
              Play
            </button>
            <button 
              onClick={() => setPlayDirection(null)}
              style={{ flex: 1, cursor: 'pointer', padding: '2px', fontSize: '11px' }}
              title="Pause"
            >
              Pause
            </button>
          </div>

          {/* Speed control */}
          <div style={{ marginTop: '5px', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span>Speed (ms):</span>
            <input 
              type="number" 
              min="10" 
              step="10" 
              value={playbackSpeed} 
              onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
              style={{ width: '50px' }}
            />
          </div>

          <div style={{ fontSize: '10px', color: '#666', marginTop: '4px' }}>
            <p>Red = current; darker = visited.</p>
            <p>Can hold prev/next to auto-step.</p>
          </div>
        </div>

        <div>Pickle: {pickleLoaded ? 'Loaded' : 'Not loaded'}</div>
        <div>Nodes: {graphData?.nodes?.length || 0}</div>
      </div>
      
      {traces && layout && config && (
        <Plot
          divId="graph-viewer-3d"
          data={traces}
          layout={{
            ...layout, 
            autosize: true,
            clickmode: 'event', // Ensure click events are enabled
            hovermode: 'closest',
            datarevision: revision, // Use revision to force updates
            uirevision: 'constant', // Preserve camera position across updates
          }}
          revision={revision} // Force update when revision changes
          config={config}
          onClick={(data) => {
            console.log('Plot detected click (wrapper):', data);
            handleClick(data);
          }}
          onInitialized={(figure, graphDiv) => console.log('Plot initialized:', figure, graphDiv)}
          onUpdate={(figure, graphDiv) => console.log('Plot updated:', figure, graphDiv)}
          onRelayout={(event) => {
            // Capture camera changes to preserve them across re-renders
            if (event['scene.camera']) {
              console.log('Camera updated:', event['scene.camera']);
              setCameraState(event['scene.camera']);
            }
          }}
          style={styles.plot}
          useResizeHandler={true}
          onError={(err) => {
            console.error('Plotly error:', err);
            setRenderError(err.message || 'Unknown plotly error');
          }}
        />
      )}
      
      {!traces && graphData && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
          <div style={{ textAlign: 'center', color: '#666' }}>
            <h3>Loading graph visualization...</h3>
          </div>
        </div>
      )}
      
      {/* Info Panel - Always visible */}
      <div style={styles.info}>
        {selectedNode ? (
          <>
            <strong>Selected Node:</strong> {selectedNode.id}
            <br />
            {loadingImage && (
              <div style={{ marginTop: '10px', color: '#666' }}>
                Loading visualization...
              </div>
            )}
            {stateImage && (
              <div style={{ marginTop: '10px' }}>
                <strong>Visualization:</strong>
                <br />
                <img 
                  src={stateImage.src} 
                  alt="State visualization"
                  style={{ 
                    maxWidth: '400px', 
                    maxHeight: '400px',
                    border: '1px solid #ccc',
                    marginTop: '5px',
                    marginBottom: '50px',
                    borderRadius: '4px'
                  }}
                />
              </div>
            )}
            {renderError && (
              <div style={{ marginTop: '10px', padding: '10px', backgroundColor: '#fff3cd', border: '1px solid #ffc107', borderRadius: '4px', color: '#856404' }}>
                <strong>Error:</strong><br />
                {renderError}
              </div>
            )}
          </>
        ) : (
          <div style={{ color: '#666', fontStyle: 'italic' }}>
            <strong>Node Inspector</strong>
            <br /><br />
            Click on a node to view details.
          </div>
        )}
      </div>
    </div>
  );
}

const styles = {
  container: {
    position: 'relative',
    width: '100%',
    height: '100%',
    backgroundColor: '#f5f5f5',
  },
  plot: {
    width: '100%',
    height: '100%',
  },
  message: {
    textAlign: 'center',
    color: '#666',
  },
  info: {
    position: 'absolute',
    bottom: '20px',
    left: '20px',
    padding: '10px 15px',
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    border: '1px solid #ccc',
    borderRadius: '4px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
    zIndex: 2000,
    maxWidth: '450px',
    maxHeight: '80vh',
    overflowY: 'auto',
  },
  errorBox: {
    position: 'absolute',
    bottom: '20px',
    left: '50%',
    transform: 'translateX(-50%)',
    padding: '20px',
    backgroundColor: '#fff3cd',
    border: '2px solid #ffc107',
    borderRadius: '4px',
    color: '#856404',
    maxWidth: '600px',
    boxShadow: '0 4px 8px rgba(0,0,0,0.2)',
    zIndex: 1000,
    whiteSpace: 'pre-wrap',
  }
};
