"use client";

import React from "react";
import { motion } from "framer-motion";
import { Document } from "@/types";
import IngestionPipeline from "./IngestionPipeline";

interface DocumentItemProps {
  document: Document;
  isSelected: boolean;
  onToggle: (docId: string) => void;
  onDelete: (docId: string) => void;
  isDeleting: boolean;
  index?: number;
}

const DocumentItem = React.memo(function DocumentItem({
  document: doc,
  isSelected,
  onToggle,
  onDelete,
  isDeleting,
  index = 0,
}: DocumentItemProps) {
  const isFailed = doc.status === "failed";

  return (
    <motion.div
      initial={{ opacity: 0, x: -6 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.02 }}
      className={`group relative rounded-lg transition-colors ${
        isDeleting
          ? "opacity-60"
          : isFailed
            ? "opacity-60"
            : isSelected
              ? "bg-zinc-100 dark:bg-zinc-900"
              : "hover:bg-zinc-50 dark:hover:bg-zinc-900/60"
      }`}
    >
      <div className="flex items-start gap-2 px-2 py-1.5">
        <input
          type="checkbox"
          checked={isSelected}
          onChange={() => onToggle(doc.id)}
          className="tactical-checkbox mt-0.5 flex-shrink-0"
          disabled={doc.status === "processing" || isFailed || isDeleting}
        />
        <div className="flex-1 min-w-0">
          <div className="flex items-start gap-2">
            {/*
              Filename is clickable when we have a presigned `file_url` and
              the doc isn't mid-processing / failed / being deleted. Opens
              in a new tab. We stopPropagation on the click so it doesn't
              toggle the row's selection state.
            */}
            {doc.file_url && !isDeleting && doc.status === "completed" ? (
              <a
                href={doc.file_url}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
                className="text-xs text-zinc-800 dark:text-zinc-200 break-words flex-1 leading-tight cursor-pointer hover:text-zinc-900 dark:hover:text-zinc-50 hover:underline underline-offset-2 decoration-zinc-400 dark:decoration-zinc-600"
                title="Open file in a new tab"
              >
                {doc.file_name}
              </a>
            ) : (
              <div className="text-xs text-zinc-800 dark:text-zinc-200 break-words flex-1 leading-tight">
                {doc.file_name}
              </div>
            )}
            {isDeleting && (
              <div className="w-3 h-3 border-2 border-red-300 border-t-red-600 rounded-full animate-spin flex-shrink-0 mt-0.5" />
            )}
            {!isDeleting && doc.status === "processing" && (
              <div className="w-3 h-3 border-2 border-zinc-300 border-t-zinc-700 dark:border-zinc-700 dark:border-t-zinc-300 rounded-full animate-spin flex-shrink-0 mt-0.5" />
            )}
          </div>
          {isDeleting && (
            <div className="text-[10px] text-red-600 dark:text-red-400 mt-0.5">
              Deleting…
            </div>
          )}
          {!isDeleting && doc.status === "processing" && (
            <IngestionPipeline document={doc} />
          )}
          {!isDeleting && doc.status === "failed" && doc.error && (
            <div className="text-[10px] text-red-500 dark:text-red-400 mt-0.5 truncate">
              {doc.error}
            </div>
          )}
          {!isDeleting && (!doc.status || doc.status === "completed") && doc.created_at && (
            <div className="text-[10px] text-zinc-500 mt-0.5">
              {new Date(doc.created_at).toLocaleDateString("en-US", {
                year: "numeric",
                month: "short",
                day: "numeric",
              })}
            </div>
          )}
        </div>
        {!isDeleting && (
          <button
            onClick={() => onDelete(doc.id)}
            className={`text-zinc-400 hover:text-red-600 dark:hover:text-red-400 transition-all p-0.5 flex-shrink-0 ${
              isFailed ? "opacity-100" : "opacity-0 group-hover:opacity-100"
            }`}
            title="Delete document"
          >
            <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6" />
            </svg>
          </button>
        )}
      </div>
    </motion.div>
  );
});

export default DocumentItem;
