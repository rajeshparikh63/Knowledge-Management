"use client";

import { useEffect, useRef, useState } from "react";
import { Transformer } from "markmap-lib";
import { Markmap, loadCSS, loadJS } from "markmap-view";
import { MindMapResponse } from "@/lib/api/mindmap";
import { convertToMarkdown } from "@/lib/utils/mindmapConverter";
import { useThemeStore } from "@/lib/stores/themeStore";

// Extend Window type for markmap
declare global {
  interface Window {
    markmap?: any;
  }
}

interface MindMapViewerProps {
  mindMapData: MindMapResponse;
  onClose: () => void;
}

export default function MindMapViewer({ mindMapData, onClose }: MindMapViewerProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const markmapRef = useRef<Markmap | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const theme = useThemeStore((state) => state.theme);
  const isDark = theme === 'dark';

  useEffect(() => {
    if (!svgRef.current) return;

    const initMindMap = async () => {
      try {
        // Convert backend data to markdown
        const markdown = convertToMarkdown(mindMapData.mind_map.nodes, mindMapData.mind_map.edges);

        // Transform markdown to markmap data
        const transformer = new Transformer();
        const { root, features } = transformer.transform(markdown);

        // Load required assets
        const { styles, scripts } = transformer.getUsedAssets(features);
        if (styles) loadCSS(styles);
        if (scripts) loadJS(scripts, { getMarkmap: () => window.markmap });

        // Destroy existing markmap if theme changed or first load
        if (markmapRef.current) {
          markmapRef.current.destroy();
          markmapRef.current = null;
        }

        // Create markmap with theme-appropriate colors
        // Theme-aware colors
        const lightColors = [
          '#2563eb', // blue-600 (root)
          '#1d4ed8', // blue-700
          '#1e40af', // blue-800
          '#1e3a8a', // blue-900
          '#172554', // blue-950
        ];
        const darkColors = [
          '#fbbf24', // amber-400 (root)
          '#f59e0b', // amber-500
          '#d97706', // amber-600
          '#b45309', // amber-700
          '#92400e', // amber-800
        ];

        const options = {
          color: (node: any) => {
            // Color nodes based on depth with theme-appropriate colors
            const colors = isDark ? darkColors : lightColors;
            return colors[Math.min(node.state?.depth || 0, colors.length - 1)];
          },
          paddingX: 20,
          paddingY: 10,
          spacingHorizontal: 150,
          spacingVertical: 15,
          autoFit: true,
          duration: 500,
          nodeMinHeight: 30,
          maxWidth: 300,
          initialExpandLevel: 3, // Show first 3 levels expanded
        };

        markmapRef.current = Markmap.create(svgRef.current!, options, root);

        // Fit the view after a short delay to ensure rendering
        setTimeout(() => {
          if (markmapRef.current) {
            markmapRef.current.fit();
          }
        }, 200);

        setIsLoading(false);
      } catch (error) {
        // Mind map render failed
        setIsLoading(false);
      }
    };

    initMindMap();

    // Cleanup
    return () => {
      if (markmapRef.current) {
        markmapRef.current.destroy();
        markmapRef.current = null;
      }
    };
  }, [mindMapData, isDark]);

  return (
    <div className="fixed inset-0 z-50 bg-slate-200/90 dark:bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="w-full h-full max-w-7xl max-h-[90vh] bg-white dark:bg-slate-950 border border-blue-200 dark:border-amber-400/30 shadow-2xl flex flex-col">
        {/* Header */}
        <div className="border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/50 backdrop-blur-sm p-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <svg className="w-6 h-6 text-blue-600 dark:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
            </svg>
            <div>
              <h2 className="text-lg font-bold text-blue-700 dark:text-amber-400 tracking-wider">MIND MAP</h2>
              <p className="text-[10px] text-slate-500 tracking-wider">
                {mindMapData.node_count} NODES • {mindMapData.edge_count} CONNECTIONS
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="tactical-panel p-2 hover:bg-slate-200 dark:hover:bg-slate-800 transition-colors group"
            aria-label="Close"
          >
            <svg className="w-5 h-5 text-slate-400 group-hover:text-blue-600 dark:group-hover:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 flex overflow-hidden">
          {/* Sidebar - Summary & Key Points */}
          <div className="w-80 border-r border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/30 overflow-y-auto tactical-scrollbar p-4">
            {/* Summary */}
            <div className="mb-6">
              <div className="flex items-center gap-2 mb-3">
                <svg className="w-4 h-4 text-tactical-green" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <h3 className="text-xs font-bold text-tactical-green tracking-wider">SUMMARY</h3>
              </div>
              <p className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed">
                {mindMapData.summary}
              </p>
            </div>

            {/* Key Points */}
            <div>
              <div className="flex items-center gap-2 mb-3">
                <svg className="w-4 h-4 text-blue-600 dark:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                </svg>
                <h3 className="text-xs font-bold text-blue-600 dark:text-amber-400 tracking-wider">KEY POINTS</h3>
              </div>
              <ul className="space-y-2">
                {mindMapData.key_points.map((point, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <span className="text-[10px] text-blue-600 dark:text-amber-400 font-mono mt-0.5">▸</span>
                    <span className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed">{point}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Document Info */}
            <div className="mt-6 tactical-panel p-3 bg-slate-100 dark:bg-slate-900/50">
              <div className="text-[10px] text-slate-500 tracking-wider">
                <div className="flex items-center justify-between mb-1">
                  <span>DOCUMENTS:</span>
                  <span className="text-blue-600 dark:text-amber-400 font-bold">{mindMapData.document_count}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>ID:</span>
                  <span className="text-slate-400 font-mono text-[9px]">
                    {mindMapData.mind_map_id.slice(0, 8)}...
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Mind Map Canvas */}
          <div className="flex-1 relative bg-slate-100 dark:bg-slate-950">
            {isLoading && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <div className="inline-block w-8 h-8 border-2 border-blue-600 dark:border-amber-400 border-t-transparent rounded-full animate-spin mb-2"></div>
                  <p className="text-xs text-slate-600 dark:text-slate-400 tracking-wider">RENDERING MIND MAP...</p>
                </div>
              </div>
            )}
            <svg
              ref={svgRef}
              className="w-full h-full"
              style={{
                background: isDark
                  ? 'radial-gradient(circle at center, #0f172a 0%, #020617 100%)'
                  : 'radial-gradient(circle at center, #f1f5f9 0%, #e2e8f0 100%)',
              }}
            />
            {/* Inject custom CSS for Markmap styling */}
            <style jsx global>{`
              .markmap-node circle {
                fill: ${isDark ? '#1e293b' : '#f1f5f9'} !important;
                stroke-width: 3px !important;
                cursor: pointer !important;
              }

              .markmap-node text {
                fill: ${isDark ? '#e2e8f0' : '#1e293b'} !important;
                font-size: 15px !important;
                font-weight: 600 !important;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
                cursor: pointer !important;
                user-select: none !important;
              }

              .markmap-link {
                stroke-width: 2.5px !important;
                stroke-opacity: 0.6 !important;
                fill: none !important;
              }

              .markmap-node:hover circle {
                stroke-width: 4px !important;
                filter: brightness(${isDark ? '1.2' : '0.95'});
              }

              .markmap-node:hover text {
                fill: ${isDark ? '#fbbf24' : '#2563eb'} !important;
              }

              .markmap g[data-depth="0"] circle {
                r: 15 !important;
              }

              .markmap g[data-depth="1"] circle {
                r: 12 !important;
              }

              .markmap g[data-depth="2"] circle {
                r: 10 !important;
              }

              .markmap g[data-depth="3"] circle {
                r: 8 !important;
              }
            `}</style>
          </div>
        </div>

        {/* Footer - Controls */}
        <div className="border-t border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/50 backdrop-blur-sm p-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4 text-[10px] text-slate-500 tracking-wider">
              <div className="flex items-center gap-1">
                <kbd className="px-1.5 py-0.5 bg-slate-200 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded text-[9px]">SCROLL</kbd>
                <span>Zoom</span>
              </div>
              <div className="flex items-center gap-1">
                <kbd className="px-1.5 py-0.5 bg-slate-200 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded text-[9px]">DRAG</kbd>
                <span>Pan</span>
              </div>
              <div className="flex items-center gap-1">
                <kbd className="px-1.5 py-0.5 bg-slate-200 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded text-[9px]">CLICK</kbd>
                <span>Expand/Collapse Nodes</span>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => markmapRef.current?.fit()}
                className="tactical-panel px-3 py-1.5 hover:bg-slate-200 dark:hover:bg-slate-800 transition-colors group flex items-center gap-2"
                title="Reset View"
              >
                <svg className="w-3 h-3 text-blue-600 dark:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                </svg>
                <span className="text-[9px] text-slate-400 tracking-wider">FIT VIEW</span>
              </button>
              <div className="text-[9px] text-slate-600 tracking-wider font-mono">
                TACTICAL VISUALIZATION SYSTEM v1.0
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
