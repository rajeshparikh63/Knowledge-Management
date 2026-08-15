"use client";

import React from "react";

interface SidebarHeaderProps {
  totalDocs: number;
  selectedCount: number;
  onSelectAll: () => void;
  onClearSelection: () => void;
  onUploadClick: () => void;
  uploadStatus: string | null;
}

const SidebarHeader = React.memo(function SidebarHeader({
  totalDocs,
  selectedCount,
  onSelectAll,
  onClearSelection,
  onUploadClick,
  uploadStatus,
}: SidebarHeaderProps) {
  const renderUploadContent = () => {
    if (uploadStatus === "uploading" || uploadStatus === "processing") {
      return (
        <>
          <div className="w-3.5 h-3.5 border-2 border-current/30 border-t-current rounded-full animate-spin" />
          {uploadStatus === "uploading" ? "Uploading…" : "Processing…"}
        </>
      );
    }
    if (uploadStatus === "completed") {
      return (
        <>
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M5 13l4 4L19 7" />
          </svg>
          Completed
        </>
      );
    }
    if (uploadStatus === "failed") {
      return (
        <>
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M6 18L18 6M6 6l12 12" />
          </svg>
          Failed
        </>
      );
    }
    return (
      <>
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 5v14M5 12h14" />
        </svg>
        Upload data
      </>
    );
  };

  return (
    <>
      {/* Title + Upload */}
      <div className="px-4 pt-4 pb-3">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-xs font-semibold tracking-wider uppercase text-zinc-500 dark:text-zinc-400">
            Repository
          </h2>
          <span className="text-[11px] text-zinc-400 dark:text-zinc-500 font-mono">
            {totalDocs}
          </span>
        </div>

        <button
          onClick={onUploadClick}
          disabled={uploadStatus !== null}
          className="w-full inline-flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-zinc-900 text-white text-sm font-medium hover:bg-zinc-800 dark:bg-white dark:text-zinc-900 dark:hover:bg-zinc-200 disabled:opacity-60 disabled:cursor-not-allowed transition-colors"
        >
          {renderUploadContent()}
        </button>
      </div>

      {/* Selection Controls */}
      {totalDocs > 0 && (
        <div className="flex items-center justify-between px-4 pb-3 border-b border-zinc-200 dark:border-zinc-800">
          <div className="flex items-center gap-2 text-xs text-zinc-500 dark:text-zinc-400">
            {selectedCount > 0 ? (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                <span className="font-medium text-zinc-700 dark:text-zinc-300">
                  {selectedCount} selected
                </span>
              </>
            ) : (
              <span>None selected</span>
            )}
          </div>
          <div className="flex items-center gap-3 text-xs">
            <button
              onClick={onSelectAll}
              className="font-medium text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white transition-colors"
            >
              Select all
            </button>
            {selectedCount > 0 && (
              <button
                onClick={onClearSelection}
                className="font-medium text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors"
              >
                Clear
              </button>
            )}
          </div>
        </div>
      )}
    </>
  );
});

export default SidebarHeader;
