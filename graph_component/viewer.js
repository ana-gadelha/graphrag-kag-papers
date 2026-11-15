export default function(component) {
  const { data = {}, parentElement, setStateValue } = component;
  const root = parentElement.querySelector("[data-graph-root]");
  const canvas = root.querySelector(".graph-canvas");
  const pathControls = root.querySelector(".path-controls");
  const legend = root.querySelector(".graph-legend");
  const summary = root.querySelector(".graph-summary");
  const title = root.querySelector(".detail-title");
  const detail = root.querySelector(".detail-content");
  const resetButton = root.querySelector(".reset-view");
  const graphData = {
    nodes: Array.isArray(data.nodes) ? data.nodes : [],
    edges: Array.isArray(data.edges) ? data.edges : [],
    paths: Array.isArray(data.paths) ? data.paths : [],
    claims: Array.isArray(data.claims) ? data.claims : [],
  };
  const fingerprint = data.fingerprint || "default-graph";
  const cacheRoot = globalThis.__graphragGraphViewCache || (globalThis.__graphragGraphViewCache = {});
  const cache = cacheRoot[fingerprint] || (cacheRoot[fingerprint] = {});
  const initialSelection = data.selection === undefined ? (cache.selection || null) : data.selection;
  cache.selection = initialSelection;
  const typeColors = {
    Paper: "#cfe8ff",
    Author: "#e8d5ff",
    Concept: "#fff1c2",
    Method: "#ffe8b3",
    Dataset: "#d5f5d5",
    Unknown: "#e7eaf0",
  };
  const legendLabels = {
    Paper: "Artigo",
    Author: "Autor",
    Concept: "Conceito",
    Method: "Método",
    Dataset: "Dataset",
    Unknown: "Outra entidade",
  };

  function text(value) {
    if (value === null || value === undefined) return "";
    if (typeof value === "string") return value;
    try {
      return JSON.stringify(value, null, 2);
    } catch (_error) {
      return String(value);
    }
  }

  function appendText(parent, tag, value, className) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    element.textContent = text(value);
    parent.appendChild(element);
    return element;
  }

  function appendSectionTitle(parent, label) {
    appendText(parent, "h4", label, "detail-section-title");
  }

  function appendList(parent, values) {
    if (!values || typeof values !== "object") return;
    const entries = Array.isArray(values)
      ? values.map((value, index) => [String(index + 1), value])
      : Object.entries(values);
    if (entries.length === 0) return;
    const list = document.createElement("dl");
    list.className = "detail-list";
    entries.forEach(([key, value]) => {
      appendText(list, "dt", key);
      appendText(list, "dd", value);
    });
    parent.appendChild(list);
  }

  function sourceNames(ids) {
    return Array.isArray(ids) ? ids.filter(Boolean).join(", ") : "";
  }

  function relatedClaims(record, activePath) {
    const sourceIds = new Set(record && record.source_ids ? record.source_ids : []);
    const pathIds = new Set(record && record.path_ids ? record.path_ids : []);
    if (activePath) pathIds.add(activePath.path_id);
    return graphData.claims.filter((claim) => {
      const claimSources = Array.isArray(claim.source_ids) ? claim.source_ids : [];
      const claimPaths = Array.isArray(claim.path_ids) ? claim.path_ids : [];
      return claimSources.some((id) => sourceIds.has(id))
        || claimPaths.some((id) => pathIds.has(id));
    });
  }

  function appendClaims(parent, claims) {
    if (claims.length === 0) return;
    appendSectionTitle(parent, "Afirmações e citações");
    claims.forEach((claim) => {
      const card = document.createElement("div");
      card.className = "claim-card";
      appendText(card, "p", claim.text || claim.kind || "Afirmação");
      const metadata = [];
      if (claim.kind) metadata.push(claim.kind);
      const sources = sourceNames(claim.source_ids);
      if (sources) metadata.push(`Fontes ${sources}`);
      if (Array.isArray(claim.path_ids) && claim.path_ids.length) {
        metadata.push(`Caminhos ${claim.path_ids.join(", ")}`);
      }
      if (metadata.length) appendText(card, "small", metadata.join(" · "));
      if (Array.isArray(claim.quotes)) {
        claim.quotes.forEach((quote) => {
          if (typeof quote === "string") {
            appendText(card, "small", `Trecho: “${quote}”`);
            return;
          }
          if (!quote || typeof quote !== "object" || typeof quote.quote !== "string") return;
          const quoteRange = [quote.start, quote.end].filter((value) => value !== null && value !== undefined).join("–");
          const provenance = [quote.source_id, quote.field, quoteRange].filter(Boolean).join(" · ");
          appendText(card, "small", `Trecho${provenance ? ` (${provenance})` : ""}: “${quote.quote}”`);
        });
      }
      parent.appendChild(card);
    });
  }

  function renderDetail(selection, activePath) {
    detail.replaceChildren();
    const selected = selection && selection.type === "node"
      ? graphData.nodes.find((node) => node.id === selection.id)
      : selection && selection.type === "edge"
        ? graphData.edges.find((edge) => edge.id === selection.id)
        : null;

    if (!selected) {
      if (activePath) {
        const pathLabel = activePath.kind === "fact" ? "Fato de uma relação" : "Menor caminho entre fontes";
        title.textContent = `${activePath.path_id} · ${activePath.kind === "fact" ? "fato" : "caminho"}`;
        appendText(detail, "p", `${pathLabel} · ${activePath.source || "Origem"} → ${activePath.target || "destino"} · ${activePath.length ?? "?"} saltos`);
      } else {
        title.textContent = "Selecione um nó ou uma relação";
        appendText(detail, "p", "As propriedades e a proveniência aparecem aqui.");
      }
      appendClaims(detail, relatedClaims(null, activePath));
      return;
    }

    if (selection.type === "node") {
      title.textContent = `${selected.type} · ${selected.label}`;
      if (selected.connector) appendText(detail, "p", "Artigo conector; não é uma fonte adicional.");
      appendSectionTitle(detail, "Propriedades");
      appendList(detail, selected.properties);
    } else {
      const source = graphData.nodes.find((node) => node.id === selected.source);
      const target = graphData.nodes.find((node) => node.id === selected.target);
      title.textContent = `${source ? source.label : selected.source} — ${selected.type} → ${target ? target.label : selected.target}`;
      appendSectionTitle(detail, "Relação direcionada");
      appendList(detail, { origem: selected.source, relação: selected.type, destino: selected.target });
      if (Object.keys(selected.properties || {}).length) {
        appendSectionTitle(detail, "Propriedades");
        appendList(detail, selected.properties);
      }
    }

    const provenance = {
      fontes: sourceNames(selected.source_ids),
      caminhos: sourceNames(selected.path_ids),
      ...((selected.provenance && typeof selected.provenance === "object") ? selected.provenance : {}),
    };
    Object.keys(provenance).forEach((key) => {
      if (provenance[key] === "" || provenance[key] === null || provenance[key] === undefined) delete provenance[key];
    });
    if (Object.keys(provenance).length) {
      appendSectionTitle(detail, "Proveniência");
      appendList(detail, provenance);
    }
    appendClaims(detail, relatedClaims(selected, activePath));
  }

  function applyPath(cy, path, controls) {
    cy.elements().removeClass("path-highlight dimmed");
    controls.querySelectorAll("button[data-path]").forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.path === (path ? path.path_id : "")));
    });
    if (!path) return;
    cy.elements().addClass("dimmed");
    (path.node_ids || []).forEach((id) => cy.getElementById(id).removeClass("dimmed").addClass("path-highlight"));
    (path.edge_ids || []).forEach((id) => cy.getElementById(id).removeClass("dimmed").addClass("path-highlight"));
  }

  function cachedActivePath() {
    if (Object.prototype.hasOwnProperty.call(cache, "activePathId")) {
      return graphData.paths.find((path) => path.path_id === cache.activePathId) || null;
    }
    return graphData.paths.find((path) => path.path_id === "P1") || graphData.paths[0] || null;
  }

  if (root.__cy && root.__fingerprint === fingerprint) {
    root.__setStateValue = setStateValue;
    root.__cache = cache;
    root.__selection = initialSelection;
    const activePath = cachedActivePath();
    root.__activePathId = activePath ? activePath.path_id : "";
    root.__cy.elements().removeClass("is-selected");
    if (initialSelection && initialSelection.id) {
      root.__cy.getElementById(initialSelection.id).addClass("is-selected");
    }
    applyPath(root.__cy, activePath, pathControls);
    renderDetail(initialSelection, activePath);
    return;
  }

  if (root.__cy) {
    root.__cy.destroy();
    root.__cy = null;
  }

  pathControls.replaceChildren();
  const paths = graphData.paths;
  paths.forEach((path) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.path = path.path_id;
    button.textContent = path.path_id;
    const pathLabel = path.kind === "fact" ? "Fato de um salto" : "Menor caminho";
    button.title = `${pathLabel}: ${path.source || "Origem"} → ${path.target || "destino"}`;
    pathControls.appendChild(button);
  });
  const clearPathButton = document.createElement("button");
  clearPathButton.type = "button";
  clearPathButton.dataset.path = "";
  clearPathButton.textContent = "Sem destaque";
  pathControls.appendChild(clearPathButton);

  const knownTypes = [...new Set(graphData.nodes.map((node) => node.type || "Unknown"))];
  legend.replaceChildren();
  knownTypes.forEach((type) => {
    const item = document.createElement("span");
    item.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.backgroundColor = typeColors[type] || typeColors.Unknown;
    item.appendChild(swatch);
    appendText(item, "span", legendLabels[type] || type);
    legend.appendChild(item);
  });
  const connectorLegend = document.createElement("span");
  connectorLegend.className = "legend-item";
  const connectorSwatch = document.createElement("span");
  connectorSwatch.className = "legend-swatch connector";
  connectorLegend.appendChild(connectorSwatch);
  appendText(connectorLegend, "span", "Artigo conector");
  legend.appendChild(connectorLegend);
  const pathLegend = document.createElement("span");
  pathLegend.className = "legend-item";
  const pathSwatch = document.createElement("span");
  pathSwatch.className = "legend-line";
  pathLegend.appendChild(pathSwatch);
  appendText(pathLegend, "span", "Caminho em destaque");
  legend.appendChild(pathLegend);
  const directionLegend = document.createElement("span");
  directionLegend.className = "legend-item";
  appendText(directionLegend, "span", "→", "legend-arrow");
  appendText(directionLegend, "span", "Direção da relação");
  legend.appendChild(directionLegend);
  const pathCountLabel = paths.length === 1 ? "1 caminho" : `${paths.length} caminhos`;
  summary.textContent = `${graphData.nodes.length} nós · ${graphData.edges.length} relações · ${pathCountLabel}`;

  let activePath = cachedActivePath();
  if (activePath) cache.activePathId = activePath.path_id;
  const savedPositions = cache.positions || {};
  const elements = [
    ...graphData.nodes.map((node) => ({
      group: "nodes",
      data: {
        id: node.id,
        label: node.label,
        type: node.type,
        color: typeColors[node.type] || typeColors.Unknown,
        connector: node.connector,
        properties: node.properties,
        source_ids: node.source_ids,
        path_ids: node.path_ids,
        provenance: node.provenance,
      },
      classes: node.connector ? "connector" : "",
      ...(savedPositions[node.id] ? { position: savedPositions[node.id] } : {}),
    })),
    ...graphData.edges.map((edge) => ({
      group: "edges",
      data: {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: edge.type,
        properties: edge.properties,
        source_ids: edge.source_ids,
        path_ids: edge.path_ids,
        provenance: edge.provenance,
      },
    })),
  ];

  const cy = globalThis.cytoscape({
    container: canvas,
    elements,
    minZoom: 0.2,
    maxZoom: 3,
    boxSelectionEnabled: false,
    style: [
      {
        selector: "node",
        style: {
          label: "data(label)",
          "text-wrap": "wrap",
          "text-max-width": "132px",
          "font-size": 13,
          color: "#263143",
          "text-valign": "center",
          "text-halign": "center",
          "text-outline-width": 2,
          "text-outline-color": "#ffffff",
          shape: "round-rectangle",
          width: 150,
          height: 62,
          padding: "10px",
          "background-color": "data(color)",
          "border-width": 1,
          "border-color": "#536174",
          "overlay-opacity": 0,
        },
      },
      {
        selector: "node.connector",
        style: {
          "border-width": 3,
          "border-style": "dashed",
          "border-color": "#a16207",
        },
      },
      {
        selector: "edge",
        style: {
          label: "data(type)",
          width: 1.6,
          "line-color": "#9aa6b8",
          "target-arrow-color": "#64748b",
          "target-arrow-shape": "triangle",
          "arrow-scale": 0.85,
          "curve-style": "bezier",
          "font-size": 9,
          color: "#455267",
          "text-rotation": "autorotate",
          "text-background-color": "#ffffff",
          "text-background-opacity": 0.88,
          "text-background-padding": "2px",
          "overlay-opacity": 0,
        },
      },
      {
        selector: ".dimmed",
        style: { opacity: 0.22 },
      },
      {
        selector: "node.path-highlight",
        style: { "border-width": 4, "border-color": "#7c3aed", "z-index": 10 },
      },
      {
        selector: "edge.path-highlight",
        style: {
          width: 4,
          "line-color": "#7c3aed",
          "target-arrow-color": "#7c3aed",
          "z-index": 10,
        },
      },
      {
        selector: ".is-selected",
        style: { "overlay-color": "#2563eb", "overlay-opacity": 0.14, "overlay-padding": 7 },
      },
    ],
    layout: { name: "preset", fit: false },
    userPanningEnabled: true,
    userZoomingEnabled: true,
    autoungrabify: false,
  });
  root.__cy = cy;
  root.__fingerprint = fingerprint;
  root.__cache = cache;
  root.__setStateValue = setStateValue;
  root.__selection = initialSelection;

  pathControls.onclick = (event) => {
    const button = event.target.closest("button[data-path]");
    if (!button) return;
    activePath = paths.find((path) => path.path_id === button.dataset.path) || null;
    cache.activePathId = activePath ? activePath.path_id : "";
    root.__activePathId = cache.activePathId;
    applyPath(cy, activePath, pathControls);
    renderDetail(root.__selection || null, activePath);
  };
  applyPath(cy, activePath, pathControls);
  renderDetail(initialSelection, activePath);

  cy.on("tap", "node, edge", (event) => {
    const element = event.target;
    cy.elements().removeClass("is-selected");
    element.addClass("is-selected");
    const selection = { type: element.isNode() ? "node" : "edge", id: element.id() };
    root.__selection = selection;
    cache.selection = selection;
    root.__setStateValue("selection", selection);
    renderDetail(selection, activePath);
  });
  cy.on("tap", (event) => {
    if (event.target !== cy) return;
    cy.elements().removeClass("is-selected");
    root.__selection = null;
    cache.selection = null;
    root.__setStateValue("selection", null);
    renderDetail(null, activePath);
  });
  if (initialSelection && initialSelection.id) {
    cy.getElementById(initialSelection.id).addClass("is-selected");
  }

  const snapshotPositions = () => Object.fromEntries(
    cy.nodes().map((node) => [node.id(), { ...node.position() }]),
  );
  const snapshotViewport = () => ({ zoom: cy.zoom(), pan: { ...cy.pan() } });
  const saveCurrentView = () => {
    cache.positions = snapshotPositions();
    cache.viewport = snapshotViewport();
  };
  cy.on("pan zoom", saveCurrentView);
  cy.on("free", "node", saveCurrentView);

  resetButton.onclick = () => {
    Object.entries(cache.originalPositions || {}).forEach(([id, position]) => {
      cy.getElementById(id).position(position);
    });
    if (cache.originalViewport) cy.viewport(cache.originalViewport);
    else cy.fit(cy.elements(), 42);
    saveCurrentView();
  };

  const restored = Object.keys(savedPositions).length > 0;
  cy.ready(() => {
    cy.one("layoutstop", () => {
      if (!restored) {
        if (cy.elements().length) cy.fit(cy.elements(), 42);
        cache.originalPositions = snapshotPositions();
        cache.originalViewport = snapshotViewport();
      } else if (cache.viewport) {
        cy.viewport(cache.viewport);
      }
      saveCurrentView();
    });
    cy.layout(restored
      ? { name: "preset", fit: false }
      : { name: "breadthfirst", directed: true, padding: 48, spacingFactor: 1.15, animate: false }
    ).run();
  });
}
