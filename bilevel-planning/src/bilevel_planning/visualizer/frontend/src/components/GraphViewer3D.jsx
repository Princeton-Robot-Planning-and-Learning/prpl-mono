import React, { useState, useCallback, useMemo, useRef } from 'react';
import Plot from 'react-plotly.js';
import { applyDefaultStyles, jsonToPlotlyTraces, getPlotlyLayout, getPlotlyConfig } from '../utils/plotlyHelpers';

// Starting camera used when the user hasn't moved the view yet. Must match the
// default eye/up in getPlotlyLayout() so the zoom buttons dolly from the same
// pose the scene first renders with.
const DEFAULT_CAMERA = {
  eye: { x: 1.3, y: 1.3, z: 1.5 },
  up: { x: 0, y: 0, z: 1 },
  center: { x: 0, y: 0, z: 0 },
};
const ZOOM_IN_FACTOR = 0.8;
const ZOOM_OUT_FACTOR = 1.25;

export function GraphViewer3D({ graphData }) {
  const [selectedNode, setSelectedNode] = useState(null);
  const [renderError, setRenderError] = useState(null);
  const [stateImage, setStateImage] = useState(null);
  const [loadingImage, setLoadingImage] = useState(false);
  const [cameraState, setCameraState] = useState(null); // Store camera position
  // Bumped on each zoom-button press to force Plotly to re-read the camera.
  // Plotly ignores programmatic camera changes while a scene's uirevision is
  // held constant (so manual scroll/rotate persist); changing this value on a
  // button press is what lets the dolly buttons actually move the camera.
  const [zoomTick, setZoomTick] = useState(0);
  const [currentTime, setCurrentTime] = useState(null); // Timeline slider position
  const [playbackSpeed, setPlaybackSpeed] = useState(200); // ms per step
  const [playDirection, setPlayDirection] = useState(null); // null, 'next', 'prev'
  // Latest timeline position, mirrored in a ref so the playback loop can read it
  // without being torn down and recreated on every step.
  const currentTimeRef = useRef(null);
  // Monotonic counter to drop stale render responses: a slower earlier render
  // completing after a newer one must not overwrite the newer frame.
  const renderSeqRef = useRef(0);

  // The playback loop is defined further down (it depends on the timeline->node
  // map and the render helper). Pacing it by render completion -- rather than a
  // fixed-rate setInterval -- is what keeps fast playback from piling up and
  // appearing to "get stuck".

  // Log prop changes
  React.useEffect(() => {
    console.log('GraphViewer3D props updated:', {
      hasGraphData: !!graphData,
      nodeCount: graphData?.nodes?.length,
      edgeCount: graphData?.edges?.length,
    });
  }, [graphData]);
  
  // Reset state when graphData changes (e.g., loading a new graph)
  React.useEffect(() => {
    console.log('GraphData changed - resetting viewer state');
    // Don't reset selectedNode - keep the info panel always visible
    setStateImage(null);
    setLoadingImage(false);
    setRenderError(null);
    setCameraState(null);
    // Stop any in-progress playback when a new graph loads.
    setPlayDirection(null);
    // Initialize timeline to min_time if available
    if (graphData && graphData.config && typeof graphData.config.min_time === 'number') {
      setCurrentTime(graphData.config.min_time);
      currentTimeRef.current = graphData.config.min_time;
    } else {
      setCurrentTime(null);
      currentTimeRef.current = null;
    }
  }, [graphData]);
  
  // Fetch and display the backend rendering of one concrete state node. The
  // returned promise resolves when this request is done (or aborts), which is
  // what lets the playback loop pace itself by render completion. A monotonic
  // sequence number drops stale responses, and an abort timeout guarantees the
  // promise always settles so a hung backend can never stall playback.
  const doFetch = useCallback(async (nodeId) => {
    const seq = ++renderSeqRef.current;
    // Keep the previous image on screen until the new one arrives, rather than
    // clearing it here. Clearing would unmount the <img>, collapsing and then
    // re-expanding the panel every frame -- the source of the playback jitter.
    setLoadingImage(true);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch('/api/visualize_state', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ node_id: nodeId }),
        signal: controller.signal,
      });
      if (seq !== renderSeqRef.current) return; // a newer frame superseded this one
      if (response.ok) {
        const data = await response.json();
        if (seq !== renderSeqRef.current) return;
        setStateImage({ src: data.image, width: data.width, height: data.height });
        setRenderError(null);
      } else {
        const contentType = response.headers.get('content-type');
        let errorMsg;
        if (contentType && contentType.includes('application/json')) {
          const error = await response.json();
          errorMsg = `Visualization failed: ${error.error || 'Unknown error'}`;
        } else {
          errorMsg = `Visualization failed: ${response.status} ${response.statusText}`;
        }
        setRenderError(errorMsg);
        setStateImage(null);
      }
    } catch (error) {
      if (seq !== renderSeqRef.current) return;
      const msg = error.name === 'AbortError'
        ? 'Visualization timed out.'
        : `Error: ${error.message}. Is the visualizer backend running?`;
      setRenderError(msg);
      setStateImage(null);
    } finally {
      clearTimeout(timeoutId);
      if (seq === renderSeqRef.current) setLoadingImage(false);
    }
  }, []);

  // Select a concrete node and render its state image. Returns the render
  // promise so callers (the playback loop) can await completion.
  const renderNode = useCallback((nodeId) => {
    const stateData = graphData?.state_data?.[nodeId] || 'No state data available';
    setSelectedNode({ id: nodeId, stateData });
    return doFetch(nodeId);
  }, [graphData, doFetch]);

  // Hooks must be called unconditionally, before any early returns
  const handleClick = useCallback(async (event) => {
    console.log('handleClick triggered');

    if (event.points && event.points.length > 0) {
      const point = event.points[0];
      console.log('Clicked point:', point);

      if (point.customdata) {
        // Abstract-action edge marker: customdata is an object, not a node id.
        if (typeof point.customdata === 'object' && point.customdata.kind === 'abstract_action') {
          setSelectedNode({ id: null, abstractActionName: point.customdata.name });
          setStateImage(null);
          setRenderError(null);
          return;
        }
        const nodeId = point.customdata;
        console.log('Node clicked:', nodeId);

        // If it's a concrete node, select it and render its state.
        if (nodeId.startsWith('x:')) {
          renderNode(nodeId);
        } else if (nodeId.startsWith('s:')) {
          console.log('Abstract state clicked');
          const abstractNode = graphData?.nodes?.find(n => n.id === nodeId);
          setSelectedNode({ id: nodeId, stateData: null, atoms: abstractNode?.atoms ?? null });
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
  }, [graphData, renderNode]);

  // Map each timeline step (a concrete node's time_index) to its node. The
  // backend stamps a unique time_index on every concrete node, so this is a
  // 1:1 lookup from the slider/playback position to the node to render.
  const timeIndexToNode = useMemo(() => {
    const m = new Map();
    for (const n of graphData?.nodes ?? []) {
      if (n.type === 'concrete' && typeof n.time_index === 'number') {
        m.set(n.time_index, n);
      }
    }
    return m;
  }, [graphData]);

  // Move the timeline to step ``t``: update the slider/highlight and render the
  // corresponding state. Returns the render promise so the playback loop can
  // wait for the frame before scheduling the next one.
  const renderTimeStep = useCallback((t) => {
    setCurrentTime(t);
    currentTimeRef.current = t;
    const node = timeIndexToNode.get(t);
    return node ? renderNode(node.id) : Promise.resolve();
  }, [timeIndexToNode, renderNode]);

  // Playback loop. Each iteration advances one step, renders it, and only then
  // schedules the next iteration -- after whatever time is left in the playback
  // interval. Pacing by completion (instead of a blind setInterval) keeps a slow
  // matplotlib render from letting frames pile up, which previously made
  // playback stall until the user pressed Play again. Stops at the timeline ends.
  React.useEffect(() => {
    if (!playDirection) return;
    const min = graphData?.config?.min_time ?? 0;
    const max = graphData?.config?.max_time ?? 100;
    let cancelled = false;
    let timer = null;

    const loop = async () => {
      if (cancelled) return;
      const cur = currentTimeRef.current ?? (playDirection === 'next' ? min - 1 : max + 1);
      const next = playDirection === 'next' ? cur + 1 : cur - 1;
      if (next < min || next > max) {
        setPlayDirection(null); // reached an end; stop
        return;
      }
      const started = performance.now();
      await renderTimeStep(next);
      if (cancelled) return;
      const elapsed = performance.now() - started;
      timer = setTimeout(loop, Math.max(0, playbackSpeed - elapsed));
    };
    loop();

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [playDirection, playbackSpeed, graphData, renderTimeStep]);

  // Dolly the camera by scaling its eye vector toward (factor < 1, zoom in) or
  // away from (factor > 1, zoom out) the scene center. This goes past the
  // scroll wheel's built-in zoom cap. Operates on the live camera (cameraState
  // if the user has moved it, otherwise the layout default) and bumps zoomTick
  // so Plotly applies the result despite the constant scene uirevision.
  const handleZoom = useCallback((factor) => {
    setCameraState(prev => {
      const cam = prev || DEFAULT_CAMERA;
      const center = cam.center || DEFAULT_CAMERA.center;
      const eye = cam.eye || DEFAULT_CAMERA.eye;
      let ex = center.x + (eye.x - center.x) * factor;
      let ey = center.y + (eye.y - center.y) * factor;
      let ez = center.z + (eye.z - center.z) * factor;
      // Clamp the eye-to-center distance so zooming in can't cross the center
      // (which flips the view) and zooming out can't lose the scene entirely.
      const dist = Math.hypot(ex - center.x, ey - center.y, ez - center.z) || 1;
      const clamped = Math.max(0.1, Math.min(50, dist));
      const s = clamped / dist;
      ex = center.x + (ex - center.x) * s;
      ey = center.y + (ey - center.y) * s;
      ez = center.z + (ez - center.z) * s;
      return {
        ...cam,
        center,
        up: cam.up || DEFAULT_CAMERA.up,
        eye: { x: ex, y: ey, z: ez },
      };
    });
    setZoomTick(prev => prev + 1);
  }, []);

  // Restore the default camera (also via a uirevision bump so it takes effect).
  const handleResetView = useCallback(() => {
    setCameraState(null);
    setZoomTick(prev => prev + 1);
  }, []);

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
      
      // The backend payload only carries topology and plan membership;
      // assign default colors/sizes/alphas here so overlays below can
      // mutate them without the backend caring.
      let graphDataCopy = applyDefaultStyles(graphData);
      
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
      // Scene-scoped uirevision: constant across manual scroll/rotate (so the
      // camera is preserved) but bumped by the zoom buttons so a programmatic
      // camera change is actually applied. Kept separate from the top-level
      // uirevision so it doesn't reset legend toggles.
      layout.scene.uirevision = `cam-${zoomTick}`;

      const config = getPlotlyConfig();

      return { traces, layout, config, error: null };
    } catch (err) {
      console.error('Error generating Plotly data:', err);
      return { traces: null, layout: null, config: null, error: err };
    }
  }, [graphData, cameraState, currentTime, zoomTick]);
  
  // Plotly's datarevision hint tells the library to re-examine its
  // data arrays. Bump it whenever the trace inputs change (a new graph
  // loads, or the timeline slider moves and recolors concrete nodes);
  // otherwise Plotly skips the update even though a fresh trace array
  // is passed, and the time overlay appears frozen.
  const [revision, setRevision] = useState(0);
  React.useEffect(() => {
    if (graphData) {
      setRevision(prev => prev + 1);
    }
  }, [graphData, currentTime]);
  
  // Now we can do early returns after all hooks are called
  if (!graphData) {
    return (
      <div style={styles.container}>
        <div style={styles.message}>
          <h2>Loading graph…</h2>
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
            onChange={(e) => renderTimeStep(Number(e.target.value))}
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

        {/* Zoom controls: dolly the camera past the scroll wheel's zoom cap. */}
        <div style={{ marginBottom: '8px', paddingBottom: '8px', borderBottom: '1px solid #ccc', padding: '5px' }}>
          <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>Zoom</div>
          <div style={{ display: 'flex', gap: '5px' }}>
            <button
              onClick={() => handleZoom(ZOOM_IN_FACTOR)}
              style={{ flex: 1, cursor: 'pointer', padding: '2px', fontSize: '13px', fontWeight: 'bold' }}
              title="Zoom in"
            >
              +
            </button>
            <button
              onClick={() => handleZoom(ZOOM_OUT_FACTOR)}
              style={{ flex: 1, cursor: 'pointer', padding: '2px', fontSize: '13px', fontWeight: 'bold' }}
              title="Zoom out"
            >
              &minus;
            </button>
            <button
              onClick={handleResetView}
              style={{ flex: 2, cursor: 'pointer', padding: '2px', fontSize: '11px' }}
              title="Reset to the default camera"
            >
              Reset view
            </button>
          </div>
          <div style={{ fontSize: '10px', color: '#666', marginTop: '4px' }}>
            <p>Buttons zoom past the scroll-wheel limit.</p>
          </div>
        </div>

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
          selectedNode.abstractActionName ? (
          <>
            <strong>Abstract action</strong>
            <div style={{ marginTop: '6px', fontFamily: 'monospace', fontSize: '13px' }}>
              {selectedNode.abstractActionName}
            </div>
          </>
          ) : (
          <>
            <strong>Selected Node:</strong> {selectedNode.id}
            <br />
            {selectedNode.atoms && (
              <div style={{ marginTop: '10px' }}>
                <strong>Atoms ({selectedNode.atoms.length}):</strong>
                {selectedNode.atoms.length > 0 ? (
                  <ul style={{ margin: '5px 0 0', paddingLeft: '18px', fontFamily: 'monospace', fontSize: '12px' }}>
                    {selectedNode.atoms.map((atom, i) => (
                      <li key={i}>{atom}</li>
                    ))}
                  </ul>
                ) : (
                  <div style={{ marginTop: '5px', color: '#666', fontStyle: 'italic' }}>
                    (no atoms — empty abstract state)
                  </div>
                )}
              </div>
            )}
            {loadingImage && !stateImage && (
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
          )
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
