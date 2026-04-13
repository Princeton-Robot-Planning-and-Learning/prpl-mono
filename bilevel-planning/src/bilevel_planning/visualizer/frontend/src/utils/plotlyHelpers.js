// Default node/edge styling. The backend exports only topology and plan
// membership; the frontend owns presentation.
const NODE_COLORS = {
  planAbstractAssoc: 'rgb(255, 165, 0)', // orange: plan + in an abstract group
  planSimple:        'rgb(44, 160, 44)', // green:  plan + no abstract group
  abstractAssoc:     'rgb(148, 103, 189)', // purple: concrete + in abstract group
  concreteSimple:    'rgb(31, 119, 180)', // blue:   concrete + no abstract group
  abstract:          'rgb(148, 103, 189)', // purple: abstract state nodes
};

const NODE_SIZE_PLAN = 12;
const NODE_SIZE_DEFAULT = 10;
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
  const hasAbstractAssoc = node.abstract_state_id !== undefined;
  let color;
  if (node.type === 'abstract') {
    color = NODE_COLORS.abstract;
  } else if (inPlan && hasAbstractAssoc) {
    color = NODE_COLORS.planAbstractAssoc;
  } else if (inPlan) {
    color = NODE_COLORS.planSimple;
  } else if (hasAbstractAssoc) {
    color = NODE_COLORS.abstractAssoc;
  } else {
    color = NODE_COLORS.concreteSimple;
  }
  return {
    color,
    size: inPlan ? NODE_SIZE_PLAN : NODE_SIZE_DEFAULT,
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
  
  // Categorize concrete and abstract nodes for separate Plotly traces.
  const planAssocNodes = graphData.nodes.filter(n => n.type === 'concrete' && n.in_plan && n.abstract_state_id !== undefined);
  const planSimpleNodes = graphData.nodes.filter(n => n.type === 'concrete' && n.in_plan && n.abstract_state_id === undefined);
  const concreteAssocNodes = graphData.nodes.filter(n => n.type === 'concrete' && !n.in_plan && n.abstract_state_id !== undefined);
  const concreteSimpleNodes = graphData.nodes.filter(n => n.type === 'concrete' && !n.in_plan && n.abstract_state_id === undefined);
  const abstractNodes = graphData.nodes.filter(n => n.type === 'abstract');

  console.log(`Node counts:
    Plan (Assoc): ${planAssocNodes.length}
    Plan (Simple): ${planSimpleNodes.length}
    Concrete (Assoc): ${concreteAssocNodes.length}
    Concrete (Simple): ${concreteSimpleNodes.length}
    Abstract: ${abstractNodes.length}
  `);
  
  // Plan Nodes (Abstract Associated) - Orange
  if (planAssocNodes.length > 0) {
    traces.push({
      type: 'scatter3d',
      mode: 'markers',
      x: planAssocNodes.map(n => n.position[0]),
      y: planAssocNodes.map(n => n.position[1]),
      z: planAssocNodes.map(n => n.position[2]),
      marker: {
        size: planAssocNodes.map(n => n.size),
        color: planAssocNodes.map(n => n.color),
        opacity: 1.0,
        line: { width: 2, color: 'white' },
      },
      customdata: planAssocNodes.map(n => n.id),
      text: planAssocNodes.map(n => `Abstract ID: ${n.abstract_state_id}`),
      name: 'Plan (Abstract Assoc)',
      hovertemplate: '<b>%{customdata}</b><br>Type: Plan (Abstract Assoc)<br>%{text}',
    });
  }

  // Plan Nodes (Simple) - Green
  if (planSimpleNodes.length > 0) {
    traces.push({
      type: 'scatter3d',
      mode: 'markers',
      x: planSimpleNodes.map(n => n.position[0]),
      y: planSimpleNodes.map(n => n.position[1]),
      z: planSimpleNodes.map(n => n.position[2]),
      marker: {
        size: planSimpleNodes.map(n => n.size),
        color: planSimpleNodes.map(n => n.color),
        opacity: 1.0,
        line: { width: 2, color: 'white' },
      },
      customdata: planSimpleNodes.map(n => n.id),
      name: 'Plan',
      hovertemplate: '<b>%{customdata}</b><br>Type: Plan',
    });
  }

  // Concrete Nodes (Abstract Associated) - Purple
  if (concreteAssocNodes.length > 0) {
    traces.push({
      type: 'scatter3d',
      mode: 'markers',
      x: concreteAssocNodes.map(n => n.position[0]),
      y: concreteAssocNodes.map(n => n.position[1]),
      z: concreteAssocNodes.map(n => n.position[2]),
      marker: {
        size: concreteAssocNodes.map(n => n.size),
        color: concreteAssocNodes.map(n => n.color),
        opacity: concreteAssocNodes.map(n => n.alpha),
      },
      customdata: concreteAssocNodes.map(n => n.id),
      text: concreteAssocNodes.map(n => `Abstract ID: ${n.abstract_state_id}`),
      name: 'Concrete (Abstract Assoc)',
      hovertemplate: '<b>%{customdata}</b><br>Type: Concrete (Abstract Assoc)<br>%{text}',
    });
  }

  // Concrete Nodes (Simple) - Blue
  if (concreteSimpleNodes.length > 0) {
    traces.push({
      type: 'scatter3d',
      mode: 'markers',
      x: concreteSimpleNodes.map(n => n.position[0]),
      y: concreteSimpleNodes.map(n => n.position[1]),
      z: concreteSimpleNodes.map(n => n.position[2]),
      marker: {
        size: concreteSimpleNodes.map(n => n.size),
        color: concreteSimpleNodes.map(n => n.color),
        opacity: concreteSimpleNodes.map(n => n.alpha),
      },
      customdata: concreteSimpleNodes.map(n => n.id),
      name: 'Concrete States',
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
        symbol: 'diamond',
      },
      customdata: abstractNodes.map(n => n.id),
      name: 'Abstract States',
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
  
  // Add edge traces
  const actionTrace = createEdgeTrace(edgesByType.action, 'Actions');
  if (actionTrace) traces.push(actionTrace);
  const abstractorTrace = createEdgeTrace(edgesByType.abstractor, 'Abstractor');
  if (abstractorTrace) traces.push(abstractorTrace);
  const abstractActionTrace = createEdgeTrace(edgesByType.abstract_action, 'Abstract actions');
  if (abstractActionTrace) traces.push(abstractActionTrace);

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
  return {
    scene: {
      camera: {
        // Start with a bird's-eye view from above, looking down
        eye: { x: 0, y: 0, z: 2.5 },
        up: { x: 0, y: 1, z: 0 }
      },
      xaxis: { 
        visible: true,
        showgrid: true,
        zeroline: false,
      },
      yaxis: { 
        visible: true,
        showgrid: true,
        zeroline: false,
      },
      zaxis: { 
        visible: true,
        showgrid: true,
        zeroline: false,
        range: [graphData.config.z_bottom, graphData.config.z_top]
      },
      // 'cube' gives each axis equal visual length regardless of data
      // extent, so the abstract plane (z=z_top) is visibly separated from
      // the concrete plane (z=z_bottom) even though the xy range is much
      // wider than [z_bottom, z_top].
      aspectmode: 'cube',
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
    margin: { l: 0, r: 0, t: 30, b: 0 },
    paper_bgcolor: 'white',
    plot_bgcolor: 'white',
    title: {
      text: 'Bilevel Planning Graph',
      font: { size: 16 }
    }
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
