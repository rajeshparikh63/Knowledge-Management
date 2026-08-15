"use client";

/**
 * DriveFilePickerModal — our own in-app file picker for Google Drive.
 *
 * Why not Google's Picker?
 *   - Google Picker needs a Google Cloud API key + their proprietary JS.
 *   - We already have a DriveClient + OAuth tokens server-side; building
 *     our own picker gives us full UX control, theming, and zero new
 *     dependencies.
 *
 * UX:
 *   - Debounced search input (Drive-side `name contains` filter).
 *   - Scrollable list with mime-type icons + file size + modified date.
 *   - Checkbox selection across pages — selecting a file then loading
 *     the next page keeps it selected.
 *   - "Load more" button when there's a next page.
 *   - "Ingest N file(s)" sticky footer button.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  googleDriveApi,
  DriveListFile,
  DrivePickedFile,
} from "@/lib/api/googleDrive";
import { useDocumentStore } from "@/lib/stores/documentStore";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onIngestStarted?: (count: number) => void;
  folderName?: string;
}

const PAGE_SIZE = 50;

export default function DriveFilePickerModal({
  isOpen,
  onClose,
  onIngestStarted,
  folderName = "Google Drive",
}: Props) {
  const [files, setFiles] = useState<DriveListFile[]>([]);
  const [nextPageToken, setNextPageToken] = useState<string | null>(null);
  const [selected, setSelected] = useState<Map<string, DriveListFile>>(
    new Map()
  );
  const [search, setSearch] = useState("");

  // "Added" is derived from the LIVE document list (by Drive file id), not
  // localStorage — so a file shows as added only while its document actually
  // exists in the KB. Delete it and the file becomes selectable again.
  const documents = useDocumentStore((s) => s.documents);
  const fetchDocuments = useDocumentStore((s) => s.fetchDocuments);
  const ingested = useMemo(
    () =>
      new Set(
        (documents || [])
          .map((d) => d.metadata?.drive_file_id)
          .filter((id): id is string => !!id)
      ),
    [documents]
  );
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const isMountedRef = useRef(true);

  // --- Debounce search input (300ms) ---
  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(search.trim()), 300);
    return () => clearTimeout(t);
  }, [search]);

  // --- Load first page whenever modal opens or debounced search changes ---
  const loadFirstPage = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await googleDriveApi.listFiles({
        pageSize: PAGE_SIZE,
        search: debouncedSearch || undefined,
      });
      if (!isMountedRef.current) return;
      setFiles(res.files);
      setNextPageToken(res.next_page_token);
    } catch (e: any) {
      if (!isMountedRef.current) return;
      setError(e?.response?.data?.detail || e?.message || "Failed to load files");
      setFiles([]);
      setNextPageToken(null);
    } finally {
      if (isMountedRef.current) setLoading(false);
    }
  }, [debouncedSearch]);

  useEffect(() => {
    isMountedRef.current = true;
    if (isOpen) {
      loadFirstPage();
      // Refresh docs so "added" badges reflect the current KB state.
      fetchDocuments();
    }
    return () => {
      isMountedRef.current = false;
    };
  }, [isOpen, loadFirstPage, fetchDocuments]);

  // Reset state on close so the next open is clean
  useEffect(() => {
    if (!isOpen) {
      setSelected(new Map());
      setSearch("");
      setDebouncedSearch("");
      setNextPageToken(null);
      setError(null);
    }
  }, [isOpen]);

  // --- Pagination ---
  const handleLoadMore = useCallback(async () => {
    if (!nextPageToken || loadingMore) return;
    setLoadingMore(true);
    try {
      const res = await googleDriveApi.listFiles({
        pageToken: nextPageToken,
        pageSize: PAGE_SIZE,
        search: debouncedSearch || undefined,
      });
      if (!isMountedRef.current) return;
      setFiles((prev) => [...prev, ...res.files]);
      setNextPageToken(res.next_page_token);
    } catch (e: any) {
      if (!isMountedRef.current) return;
      setError(e?.response?.data?.detail || e?.message || "Failed to load more");
    } finally {
      if (isMountedRef.current) setLoadingMore(false);
    }
  }, [nextPageToken, loadingMore, debouncedSearch]);

  // --- Selection ---
  const toggleSelect = useCallback(
    (file: DriveListFile) => {
      if (ingested.has(file.id)) return; // already-added files are locked
      setSelected((prev) => {
        const next = new Map(prev);
        if (next.has(file.id)) next.delete(file.id);
        else next.set(file.id, file);
        return next;
      });
    },
    [ingested]
  );

  const selectAllVisible = useCallback(() => {
    setSelected((prev) => {
      const next = new Map(prev);
      files.forEach((f) => {
        if (!ingested.has(f.id)) next.set(f.id, f);
      });
      return next;
    });
  }, [files, ingested]);

  const clearSelection = useCallback(() => setSelected(new Map()), []);

  // --- Submit ---
  const handleIngest = useCallback(async () => {
    if (selected.size === 0 || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const picked: DrivePickedFile[] = Array.from(selected.values()).map(
        (f) => ({
          id: f.id,
          name: f.name,
          mime_type: f.mime_type,
          size: f.size,
        })
      );
      const r = await googleDriveApi.ingest(picked, folderName);
      // No localStorage bookkeeping — once the backend creates the documents,
      // each picked file's drive_file_id appears in the live list and the badge
      // turns "added" on its own (and clears when the document is deleted).
      onIngestStarted?.(r.queued_count);
      onClose();
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || "Ingest failed");
    } finally {
      if (isMountedRef.current) setSubmitting(false);
    }
  }, [selected, submitting, folderName, onIngestStarted, onClose]);

  // Esc to close
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isOpen, onClose]);

  const selectedCount = selected.size;
  const allVisibleSelected = useMemo(
    () => files.length > 0 && files.every((f) => selected.has(f.id)),
    [files, selected]
  );

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm flex items-center justify-center px-4"
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            transition={{ duration: 0.15 }}
            onClick={(e) => e.stopPropagation()}
            className="bg-white dark:bg-background rounded-xl border border-border dark:border-border shadow-2xl max-w-2xl w-full overflow-hidden max-h-[85vh] flex flex-col"
          >
            {/* Header */}
            <div className="px-5 py-3.5 border-b border-border dark:border-border flex items-center justify-between flex-shrink-0">
              <div>
                <h3 className="text-sm font-semibold text-foreground dark:text-foreground">
                  Pick from Google Drive
                </h3>
                <p className="text-[11px] text-muted-foreground dark:text-muted-foreground mt-0.5">
                  Select files to ingest. Folder:{" "}
                  <span className="font-medium">{folderName}</span>
                </p>
              </div>
              <button
                onClick={onClose}
                className="text-muted-foreground hover:text-muted-foreground dark:hover:text-foreground p-1 rounded"
                title="Close (Esc)"
              >
                <svg
                  className="w-4 h-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Search */}
            <div className="px-5 py-3 border-b border-border dark:border-border flex-shrink-0">
              <div className="relative">
                <svg
                  className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <circle cx="11" cy="11" r="8" />
                  <path d="M21 21l-4.35-4.35" />
                </svg>
                <input
                  type="text"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search by file name…"
                  autoFocus
                  className="w-full pl-8 pr-3 py-1.5 text-xs rounded-md border border-border dark:border-border bg-surface-2 dark:bg-card text-foreground dark:text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring dark:focus:ring-ring"
                />
              </div>
            </div>

            {/* File list */}
            <div className="flex-1 overflow-y-auto tactical-scrollbar min-h-0">
              {loading ? (
                <div className="flex items-center justify-center py-12 text-xs text-muted-foreground">
                  <div className="w-3.5 h-3.5 border-2 border-border border-t-border dark:border-border dark:border-t-border rounded-full animate-spin mr-2" />
                  Loading files…
                </div>
              ) : error ? (
                <div className="px-5 py-8 text-xs text-red-600 dark:text-red-400">
                  {error}
                </div>
              ) : files.length === 0 ? (
                <div className="px-5 py-12 text-center text-xs text-muted-foreground">
                  {debouncedSearch
                    ? `No files matching "${debouncedSearch}"`
                    : "No supported files in your Drive"}
                </div>
              ) : (
                <ul className="py-1">
                  {files.map((file) => {
                    const isAdded = ingested.has(file.id);
                    const isSel = isAdded || selected.has(file.id);
                    return (
                      <li
                        key={file.id}
                        className={`px-4 py-2 flex items-center gap-3 transition-colors ${
                          isAdded
                            ? "opacity-60 cursor-default"
                            : "cursor-pointer"
                        } ${
                          isSel && !isAdded
                            ? "bg-secondary dark:bg-card"
                            : !isAdded
                              ? "hover:bg-surface-2 dark:hover:bg-accent/60"
                              : ""
                        }`}
                        onClick={() => toggleSelect(file)}
                      >
                        <input
                          type="checkbox"
                          checked={isSel}
                          disabled={isAdded}
                          readOnly
                          className="tactical-checkbox flex-shrink-0"
                        />
                        <MimeIcon mime={file.mime_type} />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-1.5">
                            <span className="text-xs text-foreground dark:text-foreground truncate">
                              {file.name}
                            </span>
                            {isAdded && (
                              <span className="flex-shrink-0 text-[9px] px-1 py-px rounded bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300">
                                added
                              </span>
                            )}
                          </div>
                          <div className="text-[10px] text-muted-foreground dark:text-muted-foreground mt-0.5">
                            {humanType(file.mime_type)}
                            {file.size > 0 && ` · ${humanSize(file.size)}`}
                            {file.modified_time &&
                              ` · ${formatDate(file.modified_time)}`}
                          </div>
                        </div>
                      </li>
                    );
                  })}
                </ul>
              )}
              {nextPageToken && !loading && (
                <div className="px-4 py-3 border-t border-border dark:border-border">
                  <button
                    onClick={handleLoadMore}
                    disabled={loadingMore}
                    className="w-full text-[11px] text-muted-foreground dark:text-muted-foreground hover:text-foreground dark:hover:text-foreground py-1.5 rounded disabled:opacity-50"
                  >
                    {loadingMore ? "Loading…" : "Load more"}
                  </button>
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="px-5 py-3 border-t border-border dark:border-border bg-surface-2 dark:bg-card/40 flex items-center gap-3 flex-shrink-0">
              <div className="flex items-center gap-2 text-[11px]">
                <button
                  onClick={
                    allVisibleSelected ? clearSelection : selectAllVisible
                  }
                  className="text-muted-foreground dark:text-muted-foreground hover:text-foreground dark:hover:text-foreground"
                  disabled={files.length === 0}
                >
                  {allVisibleSelected
                    ? "Clear visible"
                    : "Select all visible"}
                </button>
                {selectedCount > 0 && (
                  <>
                    <span className="text-foreground dark:text-muted-foreground">·</span>
                    <button
                      onClick={clearSelection}
                      className="text-muted-foreground dark:text-muted-foreground hover:text-foreground dark:hover:text-foreground"
                    >
                      Clear all ({selectedCount})
                    </button>
                  </>
                )}
              </div>
              <div className="flex-1" />
              <button
                onClick={onClose}
                className="text-xs px-3 py-1.5 rounded-md text-muted-foreground dark:text-foreground hover:bg-secondary dark:hover:bg-accent"
              >
                Cancel
              </button>
              <button
                onClick={handleIngest}
                disabled={selectedCount === 0 || submitting}
                className="text-xs px-3 py-1.5 rounded-md bg-brand text-brand-foreground hover:bg-brand-hover shadow-accent disabled:opacity-50 disabled:cursor-not-allowed font-medium"
              >
                {submitting
                  ? "Queuing…"
                  : selectedCount === 0
                    ? "Pick files"
                    : `Ingest ${selectedCount} file${selectedCount === 1 ? "" : "s"}`}
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// --- helpers --------------------------------------------------------------

function humanSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024)
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatDate(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  } catch {
    return "";
  }
}

function humanType(mime: string): string {
  const map: Record<string, string> = {
    "application/pdf": "PDF",
    "application/vnd.google-apps.document": "Google Doc",
    "application/vnd.google-apps.spreadsheet": "Google Sheet",
    "application/vnd.google-apps.presentation": "Google Slides",
    "application/msword": "Word",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
      "Word",
    "application/vnd.ms-excel": "Excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
      "Excel",
    "application/vnd.ms-powerpoint": "PowerPoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation":
      "PowerPoint",
    "application/rtf": "RTF",
    "text/plain": "Text",
    "text/markdown": "Markdown",
    "text/csv": "CSV",
    "text/html": "HTML",
  };
  return map[mime] || mime;
}

function MimeIcon({ mime }: { mime: string }) {
  // Two-letter glyph in a colored chip — cheap but visually distinct
  const colors: Record<string, string> = {
    "application/pdf": "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300",
    "application/vnd.google-apps.document":
      "bg-brand/15 text-brand dark:bg-brand/15 dark:text-brand",
    "application/vnd.google-apps.spreadsheet":
      "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
    "application/vnd.google-apps.presentation":
      "bg-brand text-brand dark:bg-brand dark:text-brand",
    "text/plain":
      "bg-secondary text-muted-foreground dark:bg-secondary dark:text-foreground",
    "text/csv":
      "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
  };
  const labels: Record<string, string> = {
    "application/pdf": "PDF",
    "application/vnd.google-apps.document": "DOC",
    "application/vnd.google-apps.spreadsheet": "SH",
    "application/vnd.google-apps.presentation": "PR",
    "text/plain": "TXT",
    "text/csv": "CSV",
    "text/markdown": "MD",
    "text/html": "HTM",
    "application/msword": "DOC",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
      "DOC",
    "application/vnd.ms-excel": "XLS",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "XLS",
    "application/vnd.ms-powerpoint": "PPT",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation":
      "PPT",
  };
  const cls =
    colors[mime] ||
    "bg-secondary text-muted-foreground dark:bg-secondary dark:text-muted-foreground";
  const label = labels[mime] || "FILE";
  return (
    <div
      className={`flex-shrink-0 w-7 h-7 rounded text-[9px] font-semibold flex items-center justify-center ${cls}`}
    >
      {label}
    </div>
  );
}
