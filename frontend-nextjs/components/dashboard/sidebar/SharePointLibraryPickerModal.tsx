"use client";

/**
 * SharePointLibraryPickerModal — pick which SharePoint document libraries to
 * ingest. Lists every document library across the user's team sites; each
 * selected library's files (recursively) land in a KB folder named after it.
 *
 * "Added" state is derived from the LIVE document list (by KB folder name), so
 * a library shows as added only while it actually has documents — delete the
 * KB folder and it becomes selectable again (no stale localStorage).
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { sharepointApi, SharePointLibrary } from "@/lib/api/sharepoint";
import { useDocumentStore } from "@/lib/stores/documentStore";

// Mirror the backend's KB-folder sanitization. We key off the library's full
// path (e.g. "kroolo.com/Documents") rather than its bare name — every site has
// a library called "Documents", so the bare name collides across sites.
function sanitizeKbFolder(name: string): string {
  return name.trim().replace(/\//g, "-") || "SharePoint";
}

function folderForLibrary(lib: { path?: string | null; name: string }): string {
  return sanitizeKbFolder(lib.path || lib.name);
}

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onIngestStarted?: (count: number) => void;
}

export default function SharePointLibraryPickerModal({ isOpen, onClose, onIngestStarted }: Props) {
  const [libraries, setLibraries] = useState<SharePointLibrary[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const mountedRef = useRef(true);

  const documents = useDocumentStore((s) => s.documents);
  const fetchDocuments = useDocumentStore((s) => s.fetchDocuments);
  // A library is "added" when a live doc carries its exact drive id. We match on
  // the library's unique drive id (metadata.sharepoint_drive_id), NOT the folder
  // name — many sites share a library named "Documents", so name-matching would
  // light them all up at once.
  const ingested = useMemo(
    () =>
      new Set(
        (documents || [])
          .map((d) => d.metadata?.sharepoint_drive_id)
          .filter((id): id is string => !!id)
      ),
    [documents]
  );

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const libs = await sharepointApi.listLibraries();
      if (!mountedRef.current) return;
      setLibraries(libs);
    } catch (e: any) {
      if (!mountedRef.current) return;
      setError(e?.response?.data?.detail || e?.message || "Failed to load libraries");
      setLibraries([]);
    } finally {
      if (mountedRef.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    if (isOpen) {
      load();
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
    if (!q) return libraries;
    return libraries.filter(
      (l) => (l.path || l.name).toLowerCase().includes(q)
    );
  }, [libraries, search]);

  const toggle = useCallback(
    (id: string) => {
      if (ingested.has(id)) return; // already-added libraries are locked
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
      const picked = libraries.filter((l) => selected.has(l.id));
      const r = await sharepointApi.ingestLibraries(
        // Unique per-library KB folder (site + library) so the many "Documents"
        // libraries don't all merge into one sidebar folder.
        picked.map((l) => ({ id: l.id, name: folderForLibrary(l) }))
      );
      onIngestStarted?.(r.library_count);
      onClose();
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || "Ingest failed");
    } finally {
      if (mountedRef.current) setSubmitting(false);
    }
  }, [selected, submitting, libraries, onIngestStarted, onClose]);

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
            <div className="px-5 py-3.5 border-b border-border dark:border-border flex items-center justify-between flex-shrink-0">
              <div>
                <h3 className="text-sm font-semibold text-foreground dark:text-foreground">
                  Choose SharePoint libraries
                </h3>
                <p className="text-[11px] text-muted-foreground dark:text-muted-foreground mt-0.5">
                  All supported files in the selected libraries (and subfolders) will be ingested.
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

            <div className="px-5 py-3 border-b border-border dark:border-border flex-shrink-0">
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search libraries…"
                autoFocus
                className="w-full px-3 py-1.5 text-xs rounded-md border border-border dark:border-border bg-surface-2 dark:bg-card text-foreground dark:text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring dark:focus:ring-ring"
              />
            </div>

            <div className="flex-1 overflow-y-auto tactical-scrollbar min-h-0">
              {loading ? (
                <div className="flex items-center justify-center py-12 text-xs text-muted-foreground">
                  <div className="w-3.5 h-3.5 border-2 border-border border-t-border dark:border-border dark:border-t-border rounded-full animate-spin mr-2" />
                  Loading libraries…
                </div>
              ) : error ? (
                <div className="px-5 py-8 text-xs text-red-600 dark:text-red-400">{error}</div>
              ) : filtered.length === 0 ? (
                <div className="px-5 py-12 text-center text-xs text-muted-foreground">
                  {search ? "No libraries match" : "No document libraries found"}
                </div>
              ) : (
                <ul className="py-1">
                  {filtered.map((lib) => {
                    const isAdded = ingested.has(lib.id);
                    const isSel = isAdded || selected.has(lib.id);
                    return (
                      <li
                        key={lib.id}
                        onClick={() => toggle(lib.id)}
                        className={`px-4 py-2 flex items-center gap-3 transition-colors ${
                          isAdded
                            ? "opacity-60 cursor-default"
                            : "cursor-pointer hover:bg-surface-2 dark:hover:bg-accent"
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={isSel}
                          readOnly
                          disabled={isAdded}
                          className="tactical-checkbox flex-shrink-0"
                        />
                        <div className="flex-1 min-w-0">
                          <div className="text-xs text-foreground dark:text-foreground truncate flex items-center gap-2">
                            {lib.name}
                            {isAdded && (
                              <span className="text-[9px] px-1 py-0.5 rounded bg-emerald-500/15 text-emerald-600 dark:text-emerald-400">
                                added
                              </span>
                            )}
                          </div>
                          {lib.path && (
                            <div className="text-[10px] text-muted-foreground truncate">{lib.path}</div>
                          )}
                        </div>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>

            <div className="px-5 py-3 border-t border-border dark:border-border bg-surface-2 dark:bg-card/40 flex items-center gap-3 flex-shrink-0">
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
                    ? "Pick libraries"
                    : `Ingest ${selectedCount} librar${selectedCount === 1 ? "y" : "ies"}`}
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
