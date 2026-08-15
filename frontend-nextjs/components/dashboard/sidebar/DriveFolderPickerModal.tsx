"use client";

/**
 * DriveFolderPickerModal — pick which Drive folders to ingest.
 *
 * Shown right after connecting (and re-openable via "Choose folders"). Lists
 * every folder in the user's Drive with its full path, lets them multi-select,
 * then ingests all supported files under the chosen folders (recursively,
 * including subfolders) via POST /google-drive/ingest-folders.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { googleDriveApi, DriveFolder } from "@/lib/api/googleDrive";
import { useDocumentStore } from "@/lib/stores/documentStore";

// Mirror the backend's KB-folder sanitization so the localStorage folder key
// matches the actual KB folder name (used when a delete clears the mark).
function sanitizeKbFolder(name: string): string {
  return name.trim().replace(/\//g, "-") || "Google Drive";
}

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onIngestStarted?: (folderCount: number) => void;
  folderName?: string;
}

export default function DriveFolderPickerModal({
  isOpen,
  onClose,
  onIngestStarted,
  folderName = "Google Drive",
}: Props) {
  const [folders, setFolders] = useState<DriveFolder[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [search, setSearch] = useState("");

  // "Added" is derived from the LIVE document list, not localStorage — so a
  // folder shows as added only while it actually has documents in the KB. Delete
  // the KB folder and it becomes selectable again automatically (no stale locks).
  const documents = useDocumentStore((s) => s.documents);
  const fetchDocuments = useDocumentStore((s) => s.fetchDocuments);
  const liveFolderNames = useMemo(
    () =>
      new Set(
        (documents || [])
          .map((d) => d.folder_name)
          .filter((n): n is string => !!n)
      ),
    [documents]
  );
  // Map a Drive folder to "added" by matching its KB folder name (= sanitized
  // folder name) against the folders that currently have documents.
  const ingested = useMemo(() => {
    const ids = new Set<string>();
    for (const f of folders) {
      if (liveFolderNames.has(sanitizeKbFolder(f.name))) ids.add(f.id);
    }
    return ids;
  }, [folders, liveFolderNames]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const mountedRef = useRef(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await googleDriveApi.listFolders();
      if (!mountedRef.current) return;
      setFolders(res.folders);
    } catch (e: any) {
      if (!mountedRef.current) return;
      setError(
        e?.response?.data?.detail || e?.message || "Failed to load folders"
      );
      setFolders([]);
    } finally {
      if (mountedRef.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    if (isOpen) {
      load();
      // Refresh docs so the "added" badges reflect the current KB state.
      fetchDocuments();
    }
    return () => {
      mountedRef.current = false;
    };
  }, [isOpen, load, fetchDocuments]);

  useEffect(() => {
    if (!isOpen) {
      setSelected(new Set());
      setSearch("");
      setError(null);
    }
  }, [isOpen]);

  // Esc to close
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isOpen, onClose]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return folders;
    return folders.filter((f) => f.path.toLowerCase().includes(q));
  }, [folders, search]);

  const toggle = useCallback(
    (id: string) => {
      // Already-ingested folders are locked (can't be toggled).
      if (ingested.has(id)) return;
      setSelected((prev) => {
        const next = new Set(prev);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        return next;
      });
    },
    [ingested]
  );

  const handleIngest = useCallback(async () => {
    if (selected.size === 0 || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      // Each selected folder → its own KB folder named after it.
      const picked = folders.filter((f) => selected.has(f.id));
      const r = await googleDriveApi.ingestFolders(
        picked.map((f) => ({ id: f.id, name: f.name }))
      );
      // No localStorage bookkeeping needed — once the backend creates the
      // documents, this folder's name shows up in the live list and the badge
      // turns "added" on its own (and clears when the folder is deleted).
      onIngestStarted?.(r.folder_count);
      onClose();
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || "Ingest failed");
    } finally {
      if (mountedRef.current) setSubmitting(false);
    }
  }, [selected, submitting, folders, onIngestStarted, onClose]);

  const selectedCount = selected.size;

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-[60] bg-black/50 backdrop-blur-sm flex items-center justify-center px-4"
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            transition={{ duration: 0.15 }}
            onClick={(e) => e.stopPropagation()}
            className="bg-white dark:bg-background rounded-xl border border-border dark:border-border shadow-2xl max-w-xl w-full overflow-hidden max-h-[85vh] flex flex-col"
          >
            {/* Header */}
            <div className="px-5 py-3.5 border-b border-border dark:border-border flex items-center justify-between flex-shrink-0">
              <div>
                <h3 className="text-sm font-semibold text-foreground dark:text-foreground">
                  Choose folders to ingest
                </h3>
                <p className="text-[11px] text-muted-foreground dark:text-muted-foreground mt-0.5">
                  All supported files in the selected folders (and their
                  subfolders) will be ingested.
                </p>
              </div>
              <button
                onClick={onClose}
                className="text-muted-foreground hover:text-muted-foreground dark:hover:text-foreground p-1 rounded"
                title="Close (Esc)"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Search */}
            <div className="px-5 py-3 border-b border-border dark:border-border flex-shrink-0">
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search folders by path…"
                autoFocus
                className="w-full px-3 py-1.5 text-xs rounded-md border border-border dark:border-border bg-surface-2 dark:bg-card text-foreground dark:text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring dark:focus:ring-ring"
              />
            </div>

            {/* Folder list */}
            <div className="flex-1 overflow-y-auto tactical-scrollbar min-h-0">
              {loading ? (
                <div className="flex items-center justify-center py-12 text-xs text-muted-foreground">
                  <div className="w-3.5 h-3.5 border-2 border-border border-t-border dark:border-border dark:border-t-border rounded-full animate-spin mr-2" />
                  Loading folders…
                </div>
              ) : error ? (
                <div className="px-5 py-8 text-xs text-red-600 dark:text-red-400">
                  {error}
                </div>
              ) : filtered.length === 0 ? (
                <div className="px-5 py-12 text-center text-xs text-muted-foreground">
                  {search ? "No folders match" : "No folders found in your Drive"}
                </div>
              ) : (
                <ul className="py-1">
                  {filtered.map((folder) => {
                    const isAdded = ingested.has(folder.id);
                    const isSel = isAdded || selected.has(folder.id);
                    return (
                      <li
                        key={folder.id}
                        onClick={() => toggle(folder.id)}
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
                      >
                        <input
                          type="checkbox"
                          checked={isSel}
                          disabled={isAdded}
                          readOnly
                          className="tactical-checkbox flex-shrink-0"
                        />
                        <svg
                          className="w-4 h-4 flex-shrink-0 text-brand"
                          viewBox="0 0 24 24"
                          fill="currentColor"
                        >
                          <path d="M10 4H4a2 2 0 00-2 2v12a2 2 0 002 2h16a2 2 0 002-2V8a2 2 0 00-2-2h-8l-2-2z" />
                        </svg>
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-1.5">
                            <span className="text-xs text-foreground dark:text-foreground truncate">
                              {folder.name}
                            </span>
                            {folder.shared_drive && (
                              <span
                                className="flex-shrink-0 text-[9px] px-1 py-px rounded bg-brand/15 text-brand dark:bg-brand/15 dark:text-brand"
                                title={`Shared drive: ${folder.shared_drive}`}
                              >
                                shared
                              </span>
                            )}
                            {isAdded && (
                              <span className="flex-shrink-0 text-[9px] px-1 py-px rounded bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300">
                                added
                              </span>
                            )}
                          </div>
                          {folder.path !== folder.name && (
                            <div className="text-[10px] text-muted-foreground truncate">
                              {folder.path}
                            </div>
                          )}
                        </div>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>

            {/* Footer */}
            <div className="px-5 py-3 border-t border-border dark:border-border bg-surface-2 dark:bg-card/40 flex items-center gap-3 flex-shrink-0">
              {selectedCount > 0 && (
                <button
                  onClick={() => setSelected(new Set())}
                  className="text-[11px] text-muted-foreground dark:text-muted-foreground hover:text-foreground dark:hover:text-foreground"
                >
                  Clear ({selectedCount})
                </button>
              )}
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
                    ? "Select folders"
                    : `Ingest ${selectedCount} folder${selectedCount === 1 ? "" : "s"}`}
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
