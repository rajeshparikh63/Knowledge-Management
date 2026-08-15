"use client";

/**
 * KnowledgeGraphView — force-directed visualization of a GraphRAG retrieval.
 *
 * Renders the entities + relations that backed the answer:
 *   • Nodes  = unique entities (from `anchors` ∪ triple subjects/objects)
 *   • Edges  = `triples` (predicate is the edge label, thickness ∝ confidence)
 *   • Anchor entities are highlighted (they're what drove the graph-expand hop)
 *   • Hover a node → see the chunks (text snippets) that mention it
 */

import { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";
import { useThemeStore } from "@/lib/stores/themeStore";
import type {
  KnowledgeGraph,
  KnowledgeGraphChunk,
  KnowledgeGraphTriple,
} from "@/types";

interface KnowledgeGraphViewProps {
  graph: KnowledgeGraph;
  onClose: () => void;
}

interface Node extends d3.SimulationNodeDatum {
  id: string; // entity name (canonical)
  type?: string | null;
  isAnchor: boolean;
  chunkCount: number; // mentions across retrieved chunks
}

interface Link extends d3.SimulationLinkDatum<Node> {
  source: string | Node;
  target: string | Node;
  predicate: string;
  confidence: number;
  sourceChunk?: string | null;
}

// Build nodes + links from the GraphRAG payload.
function buildGraphData(graph: KnowledgeGraph): {
  nodes: Node[];
  links: Link[];
} {
  const anchorSet = new Map<string, { type?: string | null; count: number }>();
  graph.anchors.forEach((a) => {
    anchorSet.set(a.name, { type: a.type, count: a.chunk_count });
  });

  // Collect every entity name referenced by any triple, plus all anchors.
  const nodeMap = new Map<string, Node>();
  const upsert = (name: string, type?: string | null) => {
    if (!name) return;
    const existing = nodeMap.get(name);
    const anchor = anchorSet.get(name);
    if (existing) {
      if (!existing.type && type) existing.type = type;
      return;
    }
    nodeMap.set(name, {
      id: name,
      type: type ?? anchor?.type ?? null,
      isAnchor: anchorSet.has(name),
      chunkCount: anchor?.count ?? 0,
    });
  };

  graph.anchors.forEach((a) => upsert(a.name, a.type));
  graph.triples.forEach((t) => {
    upsert(t.subject, t.subject_type);
    upsert(t.object, t.object_type);
  });

  const links: Link[] = graph.triples
    .filter((t) => t.subject && t.object)
    .map((t) => ({
      source: t.subject,
      target: t.object,
      predicate: t.predicate,
      confidence: typeof t.confidence === "number" ? t.confidence : 0.5,
      sourceChunk: t.source_chunk,
    }));

  return { nodes: Array.from(nodeMap.values()), links };
}

// Stable color per entity type
const TYPE_COLOURS: Record<string, string> = {
  PERSON: "#ef4444",
  ORG: "#3b82f6",
  ORGANIZATION: "#3b82f6",
  LOCATION: "#10b981",
  PLACE: "#10b981",
  EVENT: "#f59e0b",
  DOCUMENT: "#8b5cf6",
  CONCEPT: "#06b6d4",
};

function colorForNode(node: Node): string {
  if (node.isAnchor) return "#ef4444"; // anchors always red
  if (node.type && TYPE_COLOURS[node.type.toUpperCase()]) {
    return TYPE_COLOURS[node.type.toUpperCase()];
  }
  return "#6366f1"; // default indigo for un-typed
}

export default function KnowledgeGraphView({
  graph,
  onClose,
}: KnowledgeGraphViewProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const theme = useThemeStore((s) => s.theme);
  const isDark = theme === "dark";

  const [hoveredEntity, setHoveredEntity] = useState<string | null>(null);
  const [selectedTriple, setSelectedTriple] = useState<KnowledgeGraphTriple | null>(
    null
  );

  const { nodes, links } = useMemo(() => buildGraphData(graph), [graph]);

  // Chunks that mention a given entity (case-insensitive substring match on text).
  const chunksForEntity = useMemo(() => {
    const map = new Map<string, KnowledgeGraphChunk[]>();
    nodes.forEach((n) => {
      const lower = n.id.toLowerCase();
      const matches = graph.chunks.filter((c) =>
        c.text.toLowerCase().includes(lower)
      );
      map.set(n.id, matches);
    });
    return map;
  }, [nodes, graph.chunks]);

  // Esc to close
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // d3 force simulation
  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;
    if (nodes.length === 0) return;

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`);

    // Zoom/pan layer
    const root = svg.append("g");
    svg.call(
      d3
        .zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.2, 4])
        .on("zoom", (event) => {
          root.attr("transform", event.transform);
        })
    );

    // Arrow marker for directed edges
    svg
      .append("defs")
      .append("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 22)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", isDark ? "#64748b" : "#94a3b8");

    const sim = d3
      .forceSimulation<Node>(nodes)
      .force(
        "link",
        d3
          .forceLink<Node, Link>(links)
          .id((d) => d.id)
          .distance(120)
          .strength(0.6)
      )
      .force("charge", d3.forceManyBody().strength(-280))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collide", d3.forceCollide<Node>().radius(40));

    const link = root
      .append("g")
      .attr("stroke-linecap", "round")
      .selectAll<SVGLineElement, Link>("line")
      .data(links)
      .enter()
      .append("line")
      .attr("stroke", isDark ? "#475569" : "#cbd5e1")
      .attr("stroke-opacity", (d) => 0.4 + d.confidence * 0.6)
      .attr("stroke-width", (d) => 1 + d.confidence * 2.5)
      .attr("marker-end", "url(#arrow)")
      .style("cursor", "pointer")
      .on("click", (_, d) => {
        setSelectedTriple({
          subject: typeof d.source === "string" ? d.source : d.source.id,
          predicate: d.predicate,
          object: typeof d.target === "string" ? d.target : d.target.id,
          confidence: d.confidence,
          source_chunk: d.sourceChunk,
        });
      });

    const linkLabel = root
      .append("g")
      .selectAll("text")
      .data(links)
      .enter()
      .append("text")
      .text((d) => d.predicate)
      .attr("font-size", 10)
      .attr("fill", isDark ? "#94a3b8" : "#64748b")
      .attr("text-anchor", "middle")
      .style("pointer-events", "none")
      .style("user-select", "none");

    const node = root
      .append("g")
      .selectAll<SVGGElement, Node>("g")
      .data(nodes)
      .enter()
      .append("g")
      .style("cursor", "grab")
      .call(
        d3
          .drag<SVGGElement, Node>()
          .on("start", (event, d) => {
            if (!event.active) sim.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) sim.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      )
      .on("mouseenter", (_, d) => setHoveredEntity(d.id))
      .on("mouseleave", () => setHoveredEntity(null));

    node
      .append("circle")
      .attr("r", (d) => (d.isAnchor ? 14 : 9))
      .attr("fill", colorForNode)
      .attr("stroke", isDark ? "#0f172a" : "#ffffff")
      .attr("stroke-width", (d) => (d.isAnchor ? 3 : 2));

    node
      .append("text")
      .text((d) => d.id)
      .attr("x", 16)
      .attr("y", 4)
      .attr("font-size", 12)
      .attr("font-weight", (d) => (d.isAnchor ? 600 : 400))
      .attr("fill", isDark ? "#e2e8f0" : "#1e293b")
      .style("pointer-events", "none")
      .style("user-select", "none");

    sim.on("tick", () => {
      link
        .attr("x1", (d) => (d.source as Node).x ?? 0)
        .attr("y1", (d) => (d.source as Node).y ?? 0)
        .attr("x2", (d) => (d.target as Node).x ?? 0)
        .attr("y2", (d) => (d.target as Node).y ?? 0);

      linkLabel
        .attr(
          "x",
          (d) =>
            (((d.source as Node).x ?? 0) + ((d.target as Node).x ?? 0)) / 2
        )
        .attr(
          "y",
          (d) =>
            (((d.source as Node).y ?? 0) + ((d.target as Node).y ?? 0)) / 2
        );

      node.attr("transform", (d) => `translate(${d.x ?? 0},${d.y ?? 0})`);
    });

    return () => {
      sim.stop();
    };
  }, [nodes, links, isDark]);

  const hoveredChunks = hoveredEntity
    ? chunksForEntity.get(hoveredEntity) ?? []
    : [];

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-6">
      <div
        className={`relative w-full max-w-7xl h-[85vh] rounded-lg shadow-2xl flex flex-col overflow-hidden ${
          isDark ? "bg-slate-900 text-slate-100" : "bg-white text-slate-900"
        }`}
      >
        {/* Header */}
        <div
          className={`flex items-center justify-between px-5 py-3 border-b ${
            isDark ? "border-slate-700" : "border-slate-200"
          }`}
        >
          <div>
            <h2 className="text-lg font-semibold">Knowledge graph</h2>
            {graph.query && (
              <p
                className={`text-xs mt-0.5 ${
                  isDark ? "text-slate-400" : "text-slate-500"
                }`}
              >
                For: <span className="italic">"{graph.query}"</span>
              </p>
            )}
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3 text-xs">
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded-full bg-red-500" /> Anchor
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded-full bg-indigo-500" /> Entity
              </span>
              <span
                className={isDark ? "text-slate-500" : "text-slate-400"}
              >
                {nodes.length} nodes · {links.length} edges
              </span>
            </div>
            <button
              onClick={onClose}
              className={`p-1.5 rounded transition-colors ${
                isDark ? "hover:bg-slate-800" : "hover:bg-slate-100"
              }`}
              title="Close (Esc)"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="flex flex-1 min-h-0">
          {/* Graph canvas */}
          <div ref={containerRef} className="flex-1 relative">
            {nodes.length === 0 ? (
              <div
                className={`absolute inset-0 flex items-center justify-center text-sm ${
                  isDark ? "text-slate-500" : "text-slate-400"
                }`}
              >
                No graph data — the retrieval returned no entity relations.
              </div>
            ) : (
              <svg ref={svgRef} className="w-full h-full" />
            )}
          </div>

          {/* Side panel: hover details or selected triple */}
          <aside
            className={`w-80 border-l overflow-y-auto p-4 text-sm ${
              isDark
                ? "border-slate-700 bg-slate-950/40"
                : "border-slate-200 bg-slate-50"
            }`}
          >
            {selectedTriple ? (
              <div>
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-semibold">Selected relation</h3>
                  <button
                    onClick={() => setSelectedTriple(null)}
                    className={`text-xs ${
                      isDark
                        ? "text-slate-400 hover:text-slate-200"
                        : "text-slate-500 hover:text-slate-700"
                    }`}
                  >
                    clear
                  </button>
                </div>
                <div
                  className={`p-3 rounded text-xs leading-relaxed ${
                    isDark ? "bg-slate-800" : "bg-white border border-slate-200"
                  }`}
                >
                  <div className="font-medium">{selectedTriple.subject}</div>
                  <div
                    className={`my-1 italic ${
                      isDark ? "text-slate-400" : "text-slate-500"
                    }`}
                  >
                    → {selectedTriple.predicate} →
                  </div>
                  <div className="font-medium">{selectedTriple.object}</div>
                  {selectedTriple.confidence != null && (
                    <div
                      className={`mt-2 ${
                        isDark ? "text-slate-500" : "text-slate-400"
                      }`}
                    >
                      confidence:{" "}
                      {(selectedTriple.confidence * 100).toFixed(0)}%
                    </div>
                  )}
                </div>
              </div>
            ) : hoveredEntity ? (
              <div>
                <h3 className="font-semibold mb-2">{hoveredEntity}</h3>
                <p
                  className={`text-xs mb-3 ${
                    isDark ? "text-slate-400" : "text-slate-500"
                  }`}
                >
                  Mentioned in {hoveredChunks.length} retrieved chunk
                  {hoveredChunks.length === 1 ? "" : "s"}
                </p>
                <div className="space-y-2">
                  {hoveredChunks.slice(0, 6).map((c) => (
                    <div
                      key={c.chunk_id}
                      className={`p-2 rounded text-xs ${
                        isDark
                          ? "bg-slate-800"
                          : "bg-white border border-slate-200"
                      }`}
                    >
                      <div
                        className={`flex items-center gap-2 mb-1 text-[10px] uppercase tracking-wide ${
                          isDark ? "text-slate-500" : "text-slate-400"
                        }`}
                      >
                        <span
                          className={`px-1.5 py-0.5 rounded ${
                            c.via === "graph"
                              ? "bg-amber-500/20 text-amber-500"
                              : "bg-sky-500/20 text-sky-500"
                          }`}
                        >
                          {c.via}
                        </span>
                        {c.score != null && (
                          <span>score {c.score.toFixed(3)}</span>
                        )}
                        {c.shared_entities != null && (
                          <span>shared {c.shared_entities}</span>
                        )}
                      </div>
                      <div className="line-clamp-4">{c.text}</div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div
                className={`text-xs ${
                  isDark ? "text-slate-500" : "text-slate-400"
                }`}
              >
                <p className="mb-2">
                  <strong>Tip:</strong> hover a node to see the chunks that
                  mention it. Click an edge to inspect the relation.
                </p>
                <p>Drag nodes to rearrange. Scroll to zoom.</p>
              </div>
            )}
          </aside>
        </div>
      </div>
    </div>
  );
}
