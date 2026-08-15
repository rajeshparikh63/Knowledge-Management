"use client";

/**
 * DriveImportBanner — live progress for a Google Drive folder import.
 *
 * After the user triggers a folder ingest, the backend's discovery task
 * enumerates the folder and creates one document row per file (status
 * 'processing'), then each file is ingested and flips to 'completed'/'failed'.
 *
 * This banner reads the live document list and reports:
 *   - "Scanning Google Drive…"  while discovery hasn't created rows yet
 *   - "Importing — N/M done"     while files are processing
 *   - "✅ Imported M files"       when everything has settled
 *
 * Completion is detected when at least one Drive doc exists for this import
 * AND none are still processing. That's reliable here because each file takes
 * far longer to ingest than the row-insert cadence, so there's never a false
 * "0 processing" gap mid-discovery.
 */

import { useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Document } from "@/types";
import { useDocumentStore } from "@/lib/stores/documentStore";

interface Props {
  documents: Document[];
}

export default function DriveImportBanner({ documents }: Props) {
  const startedAt = useDocumentStore((s) => s.driveImportStartedAt);
  const clearDriveImport = useDocumentStore((s) => s.clearDriveImport);

  // The set of docs belonging to THIS import: source=google_drive and created
  // at/after the import started (minus a small clock-skew buffer). We key off
  // metadata.source, not folder_name, because Drive docs now land in multiple
  // folders (one per source folder), not a single "Google Drive" bucket.
  const batch = useMemo(() => {
    if (!startedAt) return [] as Document[];
    const cutoff = startedAt - 15_000;
    return (Array.isArray(documents) ? documents : []).filter((d) => {
      if (d.metadata?.source !== "google_drive") return false;
      const t = d.created_at ? new Date(d.created_at).getTime() : 0;
      return t >= cutoff;
    });
  }, [documents, startedAt]);

  const total = batch.length;
  const processing = batch.filter((d) => d.status === "processing").length;
  const failed = batch.filter((d) => d.status === "failed").length;
  const done = total - processing;
  const isComplete = total > 0 && processing === 0;
  const isScanning = total === 0;

  // Auto-dismiss a completed banner after a few seconds. Also a hard timeout
  // so a stuck import doesn't pin the banner forever.
  useEffect(() => {
    if (!startedAt) return;
    if (isComplete) {
      const t = setTimeout(() => clearDriveImport(), 6000);
      return () => clearTimeout(t);
    }
    // Hard stop after 30 min regardless
    const hard = setTimeout(() => clearDriveImport(), 30 * 60 * 1000);
    return () => clearTimeout(hard);
  }, [startedAt, isComplete, clearDriveImport]);

  const pct = total > 0 ? Math.round((done / total) * 100) : 0;

  return (
    <AnimatePresence>
      {startedAt && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          exit={{ opacity: 0, height: 0 }}
          className="overflow-hidden border-b border-border dark:border-border"
        >
          <div className="px-4 py-2.5">
            <div className="flex items-center gap-2">
              {isComplete ? (
                <span className="text-emerald-500 text-sm">✓</span>
              ) : (
                <div className="w-3.5 h-3.5 border-2 border-brand/30 border-t-brand rounded-full animate-spin flex-shrink-0" />
              )}
              <div className="flex-1 min-w-0">
                <div className="text-[12px] font-medium text-foreground dark:text-foreground truncate">
                  {isScanning
                    ? "Scanning Google Drive…"
                    : isComplete
                      ? `Imported ${total} file${total === 1 ? "" : "s"} from Drive`
                      : `Importing from Drive — ${done}/${total}`}
                </div>
                {failed > 0 && (
                  <div className="text-[10px] text-red-500 dark:text-red-400">
                    {failed} failed
                  </div>
                )}
              </div>
              <button
                onClick={clearDriveImport}
                className="text-muted-foreground hover:text-muted-foreground dark:hover:text-foreground text-xs flex-shrink-0"
                title="Dismiss"
              >
                ✕
              </button>
            </div>

            {/* Progress bar */}
            {!isScanning && (
              <div className="mt-2 h-1 rounded-full bg-secondary dark:bg-secondary overflow-hidden">
                <motion.div
                  className={`h-full ${isComplete ? "bg-emerald-500" : "bg-brand"}`}
                  initial={{ width: 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
