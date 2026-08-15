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
          <h2 className="text-xs font-semibold tracking-wider uppercase text-muted-foreground dark:text-muted-foreground">
            Repository
          </h2>
          <span className="text-[11px] text-muted-foreground dark:text-muted-foreground font-mono">
            {totalDocs}
          </span>
        </div>

        <button
          onClick={onUploadClick}
          disabled={uploadStatus !== null}
          className="w-full inline-flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-brand text-brand-foreground text-sm font-medium hover:bg-brand-hover shadow-accent disabled:opacity-60 disabled:shadow-none disabled:cursor-not-allowed transition-all"
        >
          {renderUploadContent()}
        </button>
      </div>

      {/* Selection Controls */}
      {totalDocs > 0 && (
        <div className="flex items-center justify-between px-4 pb-3 border-b border-border dark:border-border">
          <div className="flex items-center gap-2 text-xs text-muted-foreground dark:text-muted-foreground">
            {selectedCount > 0 ? (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                <span className="font-medium text-muted-foreground dark:text-foreground">
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
              className="font-medium text-muted-foreground dark:text-muted-foreground hover:text-foreground dark:hover:text-white transition-colors"
            >
              Select all
            </button>
            {selectedCount > 0 && (
              <button
                onClick={onClearSelection}
                className="font-medium text-muted-foreground hover:text-muted-foreground dark:hover:text-foreground transition-colors"
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
