"use client";

import React, { useMemo } from "react";
import { Document, KnowledgeBase } from "@/types";
import FolderItem from "./FolderItem";

interface FolderTreeProps {
  documents: Document[];
  knowledgeBases: KnowledgeBase[];
  selectedDocs: Set<string>;
  expandedFolders: Set<string>;
  onToggleFolder: (folderName: string) => void;
  onToggleDoc: (docId: string) => void;
  onSelectAllFolder: (folderName: string, anySelected: boolean, docIds: string[]) => void;
  onDeleteDoc: (docId: string) => void;
  onDeleteFolder: (folderName: string) => void;
  deletingDocId: string | null;
  deletingKB: string | null;
  isLoading: boolean;
}

const FolderTree = React.memo(function FolderTree({
  documents,
  knowledgeBases,
  selectedDocs,
  expandedFolders,
  onToggleFolder,
  onToggleDoc,
  onSelectAllFolder,
  onDeleteDoc,
  onDeleteFolder,
  deletingDocId,
  deletingKB,
  isLoading,
}: FolderTreeProps) {
  const documentsByFolder = useMemo(() => {
    return (Array.isArray(documents) ? documents : []).reduce(
      (acc, doc) => {
        const folder = doc.folder_name || "Uncategorized";
        if (!acc[folder]) {
          acc[folder] = [];
        }
        acc[folder].push(doc);
        return acc;
      },
      {} as Record<string, Document[]>
    );
  }, [documents]);

  const folderList = useMemo(() => {
    const allFolders = new Set<string>();
    (Array.isArray(knowledgeBases) ? knowledgeBases : []).forEach((kb) =>
      allFolders.add(kb.name)
    );
    Object.keys(documentsByFolder).forEach((folder) =>
      allFolders.add(folder)
    );
    return Array.from(allFolders).sort();
  }, [knowledgeBases, documentsByFolder]);

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <div className="w-5 h-5 border-2 border-zinc-300 border-t-zinc-700 dark:border-zinc-700 dark:border-t-zinc-300 rounded-full animate-spin mb-3" />
        <div className="text-zinc-500 text-xs">Loading</div>
      </div>
    );
  }

  if (folderList.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center px-4">
        <div className="w-10 h-10 rounded-full bg-zinc-100 dark:bg-zinc-900 flex items-center justify-center mb-3">
          <svg
            className="w-5 h-5 text-zinc-400 dark:text-zinc-600"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
            <path d="M14 2v6h6" />
          </svg>
        </div>
        <div className="text-sm font-medium text-zinc-900 dark:text-zinc-100 mb-1">
          No documents yet
        </div>
        <div className="text-xs text-zinc-500 dark:text-zinc-400 leading-relaxed max-w-[220px]">
          Upload a document to create your first knowledge base.
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-1">
      {folderList.map((folderName, folderIdx) => (
        <FolderItem
          key={folderName}
          folderName={folderName}
          documents={documentsByFolder[folderName] || []}
          selectedDocs={selectedDocs}
          expandedFolders={expandedFolders}
          onToggleFolder={onToggleFolder}
          onToggleDoc={onToggleDoc}
          onSelectAllFolder={onSelectAllFolder}
          onDeleteDoc={onDeleteDoc}
          onDeleteFolder={onDeleteFolder}
          deletingDocId={deletingDocId}
          isDeletingFolder={deletingKB === folderName}
          animationDelay={folderIdx * 30}
        />
      ))}
    </div>
  );
});

export default FolderTree;
