"use client";

import React, { useCallback, useState } from "react";
import { motion } from "framer-motion";
import { Document } from "@/types";
import { documentsApi } from "@/lib/api/documents";
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
  const [opening, setOpening] = useState(false);

  // A downloadable file exists when the doc finished and has a file_key. We do
  // NOT have a URL yet — fetch a fresh presigned one only on click, so the list
  // endpoint never has to mint URLs for every doc on every poll.
  const canOpen = doc.status === "completed" && !!doc.file_key && !isDeleting;

  const handleOpen = useCallback(
    async (e: React.MouseEvent) => {
      e.stopPropagation();
      if (!canOpen || opening) return;
      setOpening(true);
      // Open a blank tab synchronously (inside the click) so the popup blocker
      // doesn't kill it; we set its location once the URL comes back.
      const win = window.open("", "_blank");
      try {
        const fresh = await documentsApi.getDocument(doc.id);
        if (fresh.file_url) {
          if (win) win.location.href = fresh.file_url;
          else window.open(fresh.file_url, "_blank", "noopener,noreferrer");
        } else if (win) {
          win.close();
        }
      } catch {
        if (win) win.close();
      } finally {
        setOpening(false);
      }
    },
    [canOpen, opening, doc.id]
  );

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
              ? "bg-secondary dark:bg-card"
              : "hover:bg-surface-2 dark:hover:bg-accent/60"
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
              Filename is clickable when the doc finished and has a file. We
              fetch a FRESH presigned URL on click (not pre-generated for the
              whole list) and open it in a new tab. stopPropagation so the click
              doesn't toggle the row's selection state.
            */}
            {canOpen ? (
              <button
                type="button"
                onClick={handleOpen}
                disabled={opening}
                className="text-left text-xs text-foreground dark:text-foreground break-words flex-1 leading-tight cursor-pointer hover:text-foreground dark:hover:text-foreground hover:underline underline-offset-2 decoration-muted-foreground dark:decoration-muted-foreground disabled:opacity-60"
                title="Open file in a new tab"
              >
                {doc.file_name}
                {opening && " …"}
              </button>
            ) : (
              <div className="text-xs text-foreground dark:text-foreground break-words flex-1 leading-tight">
                {doc.file_name}
              </div>
            )}
            {isDeleting && (
              <div className="w-3 h-3 border-2 border-red-300 border-t-red-600 rounded-full animate-spin flex-shrink-0 mt-0.5" />
            )}
            {!isDeleting && doc.status === "processing" && (
              <div className="w-3 h-3 border-2 border-border border-t-border dark:border-border dark:border-t-border rounded-full animate-spin flex-shrink-0 mt-0.5" />
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
            <div className="text-[10px] text-muted-foreground mt-0.5">
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
            className={`text-muted-foreground hover:text-red-600 dark:hover:text-red-400 transition-all p-0.5 flex-shrink-0 ${
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
