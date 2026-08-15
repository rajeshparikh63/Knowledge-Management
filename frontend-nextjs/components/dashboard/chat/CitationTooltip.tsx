"use client";

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChatMessage } from "@/types";

interface HoveredCitation {
  index: number;
  messageId: string;
  x: number;
  y: number;
}

interface CitationTooltipProps {
  citation: HoveredCitation | null;
  messages: ChatMessage[];
  onMouseEnter: () => void;
  onMouseLeave: () => void;
}

const CitationTooltip = React.memo(function CitationTooltip({
  citation,
  messages,
  onMouseEnter,
  onMouseLeave,
}: CitationTooltipProps) {
  const message = citation
    ? messages.find((m) => m.id === citation.messageId)
    : null;
  const source = message?.sources?.[citation?.index ?? -1];

  return (
    <AnimatePresence>
      {citation && source && (
        <motion.div
          key={citation.messageId + "-" + citation.index}
          className="fixed z-50 pointer-events-none"
          style={{
            left: `${citation.x}px`,
            top: `${citation.y - 10}px`,
            transform: "translate(-50%, -100%)",
          }}
          initial={{ opacity: 0, y: 8, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 8, scale: 0.95 }}
          transition={{ duration: 0.2, ease: "easeOut" }}
        >
          <div
            className="bg-[#252b3b] border border-tactical-green/30 rounded-lg shadow-2xl p-4 max-w-md pointer-events-auto"
            onMouseEnter={onMouseEnter}
            onMouseLeave={onMouseLeave}
          >
            {/* Filename */}
            <div className="text-sm font-semibold text-tactical-green mb-2 flex items-center gap-2">
              <svg
                className="w-4 h-4 flex-shrink-0"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              <span className="truncate">{source.filename}</span>
            </div>

            {/* Source text preview */}
            <div className="text-xs text-slate-300 dark:text-slate-400 leading-relaxed max-h-64 overflow-y-auto tactical-scrollbar">
              {source.text}
            </div>

            {/* Arrow pointing down */}
            <div className="absolute left-1/2 bottom-0 transform -translate-x-1/2 translate-y-full pointer-events-none">
              <div className="w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] border-r-transparent border-t-[6px] border-t-tactical-green/30"></div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
});

export default CitationTooltip;
