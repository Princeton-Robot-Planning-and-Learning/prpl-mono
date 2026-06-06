// Default node/edge styling. The backend exports only topology and plan
// membership; the frontend owns presentation.
const NODE_COLORS = {
  concrete: 'rgb(31, 119, 180)', // blue:   concrete search states (lower plane)
  abstract: 'rgb(148, 103, 189)', // purple: abstract states (upper plane)
};

const NODE_SIZE_PLAN = 12;
const NODE_SIZE_DEFAULT = 10;
// Abstract nodes are drawn noticeably larger than concrete ones so the upper
// (abstract) plane reads as the higher-level summary of the lower plane.
const ABSTRACT_NODE_SIZE_PLAN = 22;
const ABSTRACT_NODE_SIZE_DEFAULT = 20;
const NODE_ALPHA_PLAN = 1.0;
const NODE_ALPHA_DEFAULT = 0.7;

const EDGE_COLORS = {
  action:          'rgb(0, 0, 0)',
  abstractor:      'rgb(128, 128, 128)',
  abstract_action: 'rgb(0, 0, 0)',
};
const EDGE_ALPHA = 0.7;

function defaultNodeStyle(node) {
  const inPlan = Boolean(node.in_plan);
  const isAbstract = node.type === 'abstract';
  const color = isAbstract ? NODE_COLORS.abstract : NODE_COLORS.concrete;
  const planSize = isAbstract ? ABSTRACT_NODE_SIZE_PLAN : NODE_SIZE_PLAN;
  const defaultSize = isAbstract ? ABSTRACT_NODE_SIZE_DEFAULT : NODE_SIZE_DEFAULT;
  return {
    color,
    size: inPlan ? planSize : defaultSize,
    alpha: inPlan ? NODE_ALPHA_PLAN : NODE_ALPHA_DEFAULT,
  };
}

function defaultEdgeStyle(edge) {
  return {
    color: EDGE_COLORS[edge.type] || 'rgb(0, 0, 0)',
    alpha: EDGE_ALPHA,
  };
}

/**
 * Attach default color/size/alpha to every node and edge of ``graphData``.
 *
 * Returns a shallow copy. The backend payload only carries topology
 * and plan membership; all styling decisions live here so they can be
 * changed without a backend release.
 *
 * @param {Object} graphData - topology from the visualizer backend.
 * @returns {Object} a styled copy safe to mutate further (e.g. time overlay).
 */
export function applyDefaultStyles(graphData) {
  const styledNodes = graphData.nodes.map(node => ({
    ...node,
    ...defaultNodeStyle(node),
  }));
  const styledEdges = graphData.edges.map(edge => ({
    ...edge,
    ...defaultEdgeStyle(edge),
  }));
  return { ...graphData, nodes: styledNodes, edges: styledEdges };
}

/**
 * Convert graph JSON data to Plotly traces for 3D visualization.
 *
 * Expects ``graphData`` to have been styled by ``applyDefaultStyles``.
 *
 * @param {Object} graphData - styled graph data.
 * @returns {Array} Array of Plotly trace objects
 */
export function jsonToPlotlyTraces(graphData) {
  console.log('Converting graph to Plotly traces...');
  const traces = [];
  
  // Two node planes: all concrete search states (lower) and all abstract
  // states (upper). The plan path is conveyed by the timeline overlay, not
  // a separate color, so concrete nodes share one trace; per-node color
  // arrays still let the overlay recolor individual nodes.
  const concreteNodes = graphData.nodes.filter(n => n.type === 'concrete');
  const abstractNodes = graphData.nodes.filter(n => n.type === 'abstract');

  console.log(`Node counts:
    Concrete: ${concreteNodes.length}
    Abstract: ${abstractNodes.length}
  `);

  // Concrete state nodes on the z_bottom plane.
  if (concreteNodes.length > 0) {
    traces.push({
      type: 'scatter3d',
      mode: 'markers',
      x: concreteNodes.map(n => n.position[0]),
      y: concreteNodes.map(n => n.position[1]),
      z: concreteNodes.map(n => n.position[2]),
      marker: {
        size: concreteNodes.map(n => n.size),
        color: concreteNodes.map(n => n.color),
        opacity: concreteNodes.map(n => n.alpha),
      },
      customdata: concreteNodes.map(n => n.id),
      name: 'Concrete states',
      hovertemplate: '<b>%{customdata}</b><br>Type: Concrete',
    });
  }

  // Abstract state nodes on the z_top plane.
  if (abstractNodes.length > 0) {
    traces.push({
      type: 'scatter3d',
      mode: 'markers',
      x: abstractNodes.map(n => n.position[0]),
      y: abstractNodes.map(n => n.position[1]),
      z: abstractNodes.map(n => n.position[2]),
      marker: {
        size: abstractNodes.map(n => n.size),
        color: abstractNodes.map(n => n.color),
        opacity: abstractNodes.map(n => n.alpha),
        symbol: 'circle',
      },
      customdata: abstractNodes.map(n => n.id),
      name: 'Abstract states',
      hovertemplate: '<b>%{customdata}</b><br>Type: Abstract',
    });
  }
  
  // Group edges by type
  const edgesByType = {
    action: [],
    abstractor: [],
    abstract_action: [],
  };

  graphData.edges.forEach(edge => {
    if (edgesByType[edge.type]) {
      edgesByType[edge.type].push(edge);
    }
  });
  
  // Helper to create edge trace
  function createEdgeTrace(edges, name) {
    if (edges.length === 0) return null;
    
    const x = [], y = [], z = [];
    
    edges.forEach(edge => {
      const source = graphData.nodes.find(n => n.id === edge.source);
      const target = graphData.nodes.find(n => n.id === edge.target);
      
      if (source && target) {
        x.push(source.position[0], target.position[0], null);
        y.push(source.position[1], target.position[1], null);
        z.push(source.position[2], target.position[2], null);
      }
    });
    
    return {
      type: 'scatter3d',
      mode: 'lines',
      x, y, z,
      line: {
        color: edges[0]?.color || 'black',
        width: 3,
      },
      opacity: edges[0]?.alpha || 0.7,
      name,
      hoverinfo: 'skip',
      showlegend: true,
    };
  }
  
  // Add edge traces. The action and abstract-action edges are kept out of the
  // legend (showlegend false): they're self-evident from the graph and only
  // cluttered the legend. The abstraction edges stay legended.
  const actionTrace = createEdgeTrace(edgesByType.action, 'Actions');
  if (actionTrace) {
    actionTrace.showlegend = false;
    traces.push(actionTrace);
  }
  const abstractorTrace = createEdgeTrace(edgesByType.abstractor, 'Abstraction');
  if (abstractorTrace) traces.push(abstractorTrace);
  const abstractActionTrace = createEdgeTrace(edgesByType.abstract_action, 'Abstract actions');
  if (abstractActionTrace) {
    abstractActionTrace.showlegend = false;
    traces.push(abstractActionTrace);
  }

  // Clickable midpoint markers carrying each abstract action's short name.
  // Plotly doesn't reliably fire click events on 3D lines, so these markers
  // are what make the abstract-action edges inspectable.
  const labeledAbstractActions = edgesByType.abstract_action.filter(e => e.name);
  if (labeledAbstractActions.length > 0) {
    const mx = [], my = [], mz = [], customdata = [], text = [];
    labeledAbstractActions.forEach(edge => {
      const source = graphData.nodes.find(n => n.id === edge.source);
      const target = graphData.nodes.find(n => n.id === edge.target);
      if (source && target) {
        // Seat each label at the edge midpoint. Depth-stamping removes
        // antiparallel edges, so labels no longer collide there.
        mx.push((source.position[0] + target.position[0]) / 2);
        my.push((source.position[1] + target.position[1]) / 2);
        mz.push((source.position[2] + target.position[2]) / 2);
        customdata.push({ kind: 'abstract_action', name: edge.name });
        text.push(edge.name);
      }
    });
    traces.push({
      type: 'scatter3d',
      mode: 'markers',
      x: mx, y: my, z: mz,
      marker: { size: 5, color: 'rgb(0, 0, 0)', symbol: 'square', opacity: 0.7 },
      customdata,
      text,
      name: 'Abstract action labels',
      hovertemplate: '%{text}<extra></extra>',
      showlegend: false,
    });
  }

  // Direction cones for abstract-action edges. Scatter3d lines have no
  // arrowheads, so without these you can't tell which way an edge points.
  if (edgesByType.abstract_action.length > 0) {
    const cx = [], cy = [], cz = [], cu = [], cv = [], cw = [];
    edgesByType.abstract_action.forEach(edge => {
      const source = graphData.nodes.find(n => n.id === edge.source);
      const target = graphData.nodes.find(n => n.id === edge.target);
      if (source && target) {
        const [sx, sy, sz] = source.position;
        const [tx, ty, tz] = target.position;
        let dx = tx - sx, dy = ty - sy, dz = tz - sz;
        const len = Math.hypot(dx, dy, dz) || 1;
        // Seat the cone ~70% of the way toward the target, pointing along the
        // edge (unit vector, so all arrowheads render the same size).
        cx.push(sx + 0.7 * dx);
        cy.push(sy + 0.7 * dy);
        cz.push(sz + 0.7 * dz);
        cu.push(dx / len);
        cv.push(dy / len);
        cw.push(dz / len);
      }
    });
    traces.push({
      type: 'cone',
      x: cx, y: cy, z: cz,
      u: cu, v: cv, w: cw,
      sizemode: 'absolute',
      sizeref: 0.15,
      anchor: 'center',
      colorscale: [[0, 'rgb(0, 0, 0)'], [1, 'rgb(0, 0, 0)']],
      showscale: false,
      hoverinfo: 'skip',
      showlegend: false,
      name: 'Abstract action directions',
    });
  }

  console.log(`Total traces created: ${traces.length}`);
  return traces;
}

/**
 * Get Plotly layout configuration matching render_gif() style
 * 
 * @param {Object} graphData - The graph data from export_graph_for_web()
 * @returns {Object} Plotly layout object
 */
export function getPlotlyLayout(graphData) {
  // Render the scene box with sides proportional to the data extent rather
  // than forcing a cube, so a narrow, deep tree isn't stretched sideways to
  // fill the full width. Floor the in-plane fractions so a very narrow layout
  // stays readable, and give z a fixed fraction so the two planes stay clearly
  // separated.
  const xs = graphData.nodes.map(n => n.position[0]);
  const ys = graphData.nodes.map(n => n.position[1]);
  const xRange = (Math.max(...xs) - Math.min(...xs)) || 1;
  const yRange = (Math.max(...ys) - Math.min(...ys)) || 1;
  const maxRange = Math.max(xRange, yRange);
  const MIN_FRAC = 0.5;
  const xAspect = Math.max(xRange / maxRange, MIN_FRAC);
  const yAspect = Math.max(yRange / maxRange, MIN_FRAC);

  return {
    scene: {
      camera: {
        // Isometric-ish view with z as the screen vertical, so the
        // abstract plane (z=z_top) visibly sits above the concrete
        // plane (z=z_bottom). The ground plane shows the x/y
        // hierarchical layout at an angle, with roots toward the
        // back and leaves toward the viewer.
        eye: { x: 1.3, y: 1.3, z: 1.5 },
        up: { x: 0, y: 0, z: 1 },
      },
      // Axis coordinates are abstract layout positions, so their numeric values
      // are meaningless to the viewer: hide the tick labels, tick marks, and
      // axis titles. The grid/box is kept for depth cues.
      xaxis: {
        visible: true,
        showgrid: true,
        zeroline: false,
        showticklabels: false,
        ticks: '',
        title: { text: '' },
      },
      yaxis: {
        visible: true,
        showgrid: true,
        zeroline: false,
        showticklabels: false,
        ticks: '',
        title: { text: '' },
      },
      zaxis: {
        visible: true,
        showgrid: true,
        zeroline: false,
        showticklabels: false,
        ticks: '',
        title: { text: '' },
        range: [graphData.config.z_bottom, graphData.config.z_top]
      },
      // Proportional box (see above), with z exaggerated to 0.5 so the
      // abstract plane (z=z_top) stays visibly separated from the concrete
      // plane (z=z_bottom) despite the tiny [z_bottom, z_top] data range.
      aspectmode: 'manual',
      aspectratio: { x: xAspect, y: yAspect, z: 0.5 },
      bgcolor: 'white',
    },
    clickmode: 'event+select', // Enable click events and selection
    hovermode: 'closest',      // Hover over closest point
    showlegend: true,
    legend: {
      x: 0.02,
      y: 0.98,
      bgcolor: 'rgba(255, 255, 255, 0.8)',
      bordercolor: 'gray',
      borderwidth: 1,
    },
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: 'white',
    plot_bgcolor: 'white',
  };
}

/**
 * Get Plotly config options
 */
export function getPlotlyConfig() {
  return {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['toImage', 'orbitRotation', 'tableRotation', 'resetCameraDefault3d', 'resetCameraLastSave3d'],
    displaylogo: false,
  };
}
