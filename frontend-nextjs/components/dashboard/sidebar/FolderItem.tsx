"use client";

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Document } from "@/types";
import DocumentItem from "./DocumentItem";

interface FolderItemProps {
  folderName: string;
  documents: Document[];
  selectedDocs: Set<string>;
  expandedFolders: Set<string>;
  onToggleFolder: (folderName: string) => void;
  onToggleDoc: (docId: string) => void;
  onSelectAllFolder: (folderName: string, anySelected: boolean, docIds: string[]) => void;
  onDeleteDoc: (docId: string) => void;
  onDeleteFolder: (folderName: string) => void;
  deletingDocId: string | null;
  isDeletingFolder: boolean;
  animationDelay: number;
}

const FolderItem = React.memo(function FolderItem({
  folderName,
  documents: folderDocs,
  selectedDocs,
  expandedFolders,
  onToggleFolder,
  onToggleDoc,
  onSelectAllFolder,
  onDeleteDoc,
  onDeleteFolder,
  deletingDocId,
  isDeletingFolder,
  animationDelay,
}: FolderItemProps) {
  const isExpanded = expandedFolders.has(folderName);
  const folderDocCount = folderDocs.length;

  if (folderDocCount === 0) return null;

  const folderDocIds = folderDocs.map((d) => d.id);
  const anyFolderDocsSelected = folderDocIds.some((id) => selectedDocs.has(id));

  return (
    <div
      className={`data-load ${isDeletingFolder ? "opacity-60 pointer-events-none" : ""}`}
      style={{ animationDelay: `${animationDelay}ms` }}
    >
      {/* Folder Header */}
      <div className="group rounded-lg transition-colors hover:bg-zinc-50 dark:hover:bg-zinc-900/60">
        <div className="flex items-center gap-2 px-2 py-2">
          <button
            onClick={() => onToggleFolder(folderName)}
            disabled={isDeletingFolder}
            className="text-zinc-500 hover:text-zinc-900 dark:hover:text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0"
          >
            <motion.svg
              className="w-3.5 h-3.5"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              animate={{ rotate: isExpanded ? 90 : 0 }}
              transition={{ duration: 0.15, ease: "easeInOut" }}
            >
              <path d="M9 5l7 7-7 7" />
            </motion.svg>
          </button>

          <input
            type="checkbox"
            checked={anyFolderDocsSelected}
            disabled={isDeletingFolder}
            onChange={() => onSelectAllFolder(folderName, anyFolderDocsSelected, folderDocIds)}
            className="tactical-checkbox flex-shrink-0 disabled:opacity-50 disabled:cursor-not-allowed"
            title={
              anyFolderDocsSelected
                ? "Deselect all in folder"
                : "Select all in folder"
            }
          />

          {isDeletingFolder ? (
            <div className="w-3.5 h-3.5 border-2 border-zinc-300 border-t-zinc-700 dark:border-zinc-700 dark:border-t-zinc-300 rounded-full animate-spin flex-shrink-0" />
          ) : (
            <svg
              className="w-4 h-4 text-zinc-500 dark:text-zinc-400 flex-shrink-0"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M4 20h16a2 2 0 002-2V8a2 2 0 00-2-2h-7.93a2 2 0 01-1.66-.9l-.82-1.2A2 2 0 008.93 3H4a2 2 0 00-2 2v13c0 1.1.9 2 2 2z" />
            </svg>
          )}

          <div className="flex-1 min-w-0">
            <div className="text-sm font-medium text-zinc-800 dark:text-zinc-200 truncate">
              {folderName}
            </div>
          </div>

          <span className="text-[11px] text-zinc-400 dark:text-zinc-500 font-mono flex-shrink-0">
            {folderDocCount}
          </span>

          {!isDeletingFolder && (
            <button
              onClick={() => onDeleteFolder(folderName)}
              className="opacity-0 group-hover:opacity-100 text-zinc-400 hover:text-red-600 dark:hover:text-red-400 transition-all p-0.5 flex-shrink-0"
              title="Delete folder"
            >
              <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6" />
              </svg>
            </button>
          )}
        </div>
      </div>

      {/* Folder Documents */}
      <AnimatePresence initial={false}>
        {isExpanded && folderDocs.length > 0 && (
          <motion.div
            key="folder-docs"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18, ease: "easeInOut" }}
            style={{ overflow: "hidden" }}
          >
            <div className="ml-3 pl-3 border-l border-zinc-200 dark:border-zinc-800 space-y-0.5">
              {folderDocs.map((doc, index) => (
                <DocumentItem
                  key={doc.id}
                  document={doc}
                  isSelected={selectedDocs.has(doc.id)}
                  onToggle={onToggleDoc}
                  onDelete={onDeleteDoc}
                  isDeleting={deletingDocId === doc.id}
                  index={index}
                />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
});

export default FolderItem;
