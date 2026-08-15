"use client";

/**
 * GoogleDriveSection — the Drive UI rendered inside UploadModal's "DRIVE" tab.
 *
 * Flow:
 *   - Disconnected → "Connect Google Drive". Opens OAuth in a popup. The
 *     callback postMessages the result back (success/failure + email), so we
 *     update instantly and surface errors (no more silent failures).
 *   - On successful connect → the folder picker opens automatically. The user
 *     chooses which folders to ingest (we no longer slurp the whole drive).
 *   - Connected → "Choose folders" (re-open picker) + "Pick files" (cherry-pick
 *     individual files) + Disconnect.
 *   - If the stored refresh token died (user revoked us / it expired), the
 *     status reports needs_reconnect and we show a reconnect banner.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { googleDriveApi } from "@/lib/api/googleDrive";
import { useDocumentStore } from "@/lib/stores/documentStore";
import DriveFilePickerModal from "./DriveFilePickerModal";
import DriveFolderPickerModal from "./DriveFolderPickerModal";

interface Props {
  /** Called when the user kicks off an ingest (so parent can close itself). */
  onIngestStarted?: () => void;
}

// The callback HTML page is served from the backend origin, so postMessage
// arrives with e.origin === the API origin. Validate against it when set.
function apiOrigin(): string | null {
  try {
    const base = process.env.NEXT_PUBLIC_API_URL;
    return base ? new URL(base).origin : null;
  } catch {
    return null;
  }
}

export default function GoogleDriveSection({ onIngestStarted }: Props) {
  const [status, setStatus] = useState<{
    connected: boolean;
    email: string | null;
    needsReconnect: boolean;
    loaded: boolean;
  }>({ connected: false, email: null, needsReconnect: false, loaded: false });
  const [busy, setBusy] = useState<null | "connect" | "disconnect">(null);
  const [toast, setToast] = useState<string | null>(null);
  const [filePickerOpen, setFilePickerOpen] = useState(false);
  const [folderPickerOpen, setFolderPickerOpen] = useState(false);
  const fetchDocuments = useDocumentStore((s) => s.fetchDocuments);
  const startDriveImport = useDocumentStore((s) => s.startDriveImport);

  const refreshStatus = useCallback(async () => {
    try {
      const s = await googleDriveApi.status();
      setStatus({
        connected: s.connected,
        email: s.email,
        needsReconnect: !!s.needs_reconnect,
        loaded: true,
      });
    } catch {
      setStatus((p) => ({ ...p, loaded: true }));
    }
  }, []);

  useEffect(() => {
    refreshStatus();
  }, [refreshStatus]);

  // --- Connect (popup + postMessage) -------------------------------------
  const startOAuth = useCallback(async () => {
    setBusy("connect");
    setToast(null);
    try {
      const { auth_url } = await googleDriveApi.connect();
      const popup = window.open(
        auth_url,
        "drive-oauth",
        "width=560,height=720"
      );
      if (!popup) {
        // Popup blocked → full-page redirect; callback will redirect back
        window.location.href = auth_url;
        return;
      }

      const origin = apiOrigin();
      let poll: ReturnType<typeof setInterval> | null = null;

      const cleanup = () => {
        window.removeEventListener("message", onMessage);
        if (poll) clearInterval(poll);
      };

      const onMessage = (e: MessageEvent) => {
        if (origin && e.origin !== origin) return;
        if (e.data?.type !== "drive-oauth-result") return;
        cleanup();
        try {
          popup.close();
        } catch {
          /* ignore */
        }
        if (e.data.success) {
          setStatus({
            connected: true,
            email: e.data.email || null,
            needsReconnect: false,
            loaded: true,
          });
          fetchDocuments();
          setBusy(null);
          // Prompt the user to choose folders right away
          setFolderPickerOpen(true);
        } else {
          setToast(e.data.message || "Connection failed");
          setBusy(null);
        }
      };

      window.addEventListener("message", onMessage);

      // Fallback: if the user closes the popup without finishing (no message),
      // clear the busy state and re-check status.
      poll = setInterval(() => {
        if (popup.closed) {
          cleanup();
          refreshStatus();
          setBusy(null);
        }
      }, 1000);
    } catch (e: any) {
      setToast(`Connect failed: ${e?.message || e}`);
      setBusy(null);
    }
  }, [fetchDocuments, refreshStatus]);

  const handleDisconnect = useCallback(async () => {
    if (!confirm("Disconnect Google Drive? Already-ingested docs will stay."))
      return;
    setBusy("disconnect");
    try {
      await googleDriveApi.disconnect();
      setStatus({
        connected: false,
        email: null,
        needsReconnect: false,
        loaded: true,
      });
    } catch (e: any) {
      setToast(`Disconnect failed: ${e?.message || e}`);
    } finally {
      setBusy(null);
    }
  }, []);

  const handleFilesIngested = useCallback(
    (count: number) => {
      setToast(`Queued ${count} file(s) for ingest`);
      startDriveImport("Google Drive");
      fetchDocuments();
      onIngestStarted?.();
    },
    [fetchDocuments, onIngestStarted, startDriveImport]
  );

  const handleFoldersIngested = useCallback(
    (count: number) => {
      setToast(`Ingesting ${count} folder(s) — files will appear shortly`);
      startDriveImport("Google Drive");
      fetchDocuments();
      onIngestStarted?.();
    },
    [fetchDocuments, onIngestStarted, startDriveImport]
  );

  if (!status.loaded) {
    return (
      <div className="text-xs text-muted-foreground text-center py-4">
        Checking Drive…
      </div>
    );
  }

  return (
    <>
      <div className="space-y-3">
        {/* Reconnect banner — refresh token died */}
        {status.connected && status.needsReconnect && (
          <div className="p-3 bg-brand/10 border border-brand/40 text-[11px] text-brand">
            <div className="font-semibold mb-1">⚠️ Drive connection expired</div>
            <div className="mb-2 text-brand/80">
              Your Google Drive access was revoked or expired. Reconnect to
              keep ingesting files.
            </div>
            <button
              onClick={startOAuth}
              disabled={busy === "connect"}
              className="w-full py-1.5 bg-brand text-foreground font-semibold text-[11px] tracking-wider hover:bg-brand disabled:opacity-60"
            >
              {busy === "connect" ? "RECONNECTING…" : "RECONNECT"}
            </button>
          </div>
        )}

        {!status.connected ? (
          <>
            <div className="text-[11px] text-muted-foreground leading-relaxed">
              Authorize access to Google Drive. After connecting, you choose
              which folders to ingest — only files in those folders are
              added to your knowledge base.
            </div>
            <button
              onClick={startOAuth}
              disabled={busy === "connect"}
              className="w-full flex items-center justify-center gap-2 py-2.5 px-3 bg-brand text-foreground font-semibold text-xs tracking-wider hover:bg-brand disabled:opacity-60 transition-colors"
            >
              <GoogleDriveIcon className="w-4 h-4" />
              {busy === "connect" ? "CONNECTING…" : "CONNECT GOOGLE DRIVE"}
            </button>
          </>
        ) : (
          <>
            {/* Connected pill */}
            <div className="p-3 bg-muted/50 border border-brand/20">
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                <span className="text-xs text-foreground truncate">
                  Connected as <strong>{status.email}</strong>
                </span>
              </div>
            </div>

            {/* Actions */}
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => setFolderPickerOpen(true)}
                disabled={busy !== null}
                className="py-2 px-3 bg-brand text-foreground font-semibold text-xs tracking-wider hover:bg-brand disabled:opacity-60 transition-colors"
              >
                CHOOSE FOLDERS
              </button>
              <button
                onClick={() => setFilePickerOpen(true)}
                disabled={busy !== null}
                className="py-2 px-3 border border-brand/30 text-brand font-semibold text-xs tracking-wider hover:bg-brand/10 disabled:opacity-60 transition-colors"
              >
                PICK FILES
              </button>
            </div>

            <button
              onClick={handleDisconnect}
              disabled={busy !== null}
              className="w-full py-1.5 text-[10px] tracking-widest text-muted-foreground hover:text-red-400 disabled:opacity-60 transition-colors uppercase"
            >
              {busy === "disconnect" ? "Disconnecting…" : "Disconnect Drive"}
            </button>
          </>
        )}

        {toast && (
          <div className="text-[10px] text-brand text-center bg-brand/5 border border-brand/20 py-1.5 px-2">
            {toast}
          </div>
        )}
      </div>

      <DriveFolderPickerModal
        isOpen={folderPickerOpen}
        onClose={() => setFolderPickerOpen(false)}
        onIngestStarted={handleFoldersIngested}
      />
      <DriveFilePickerModal
        isOpen={filePickerOpen}
        onClose={() => setFilePickerOpen(false)}
        onIngestStarted={handleFilesIngested}
      />
    </>
  );
}

function GoogleDriveIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path d="M8 4l8 0 6 10-8 0z" fill="#FFC107" />
      <path d="M2 18l4-6 8 0-4 6z" fill="#1E88E5" />
      <path d="M22 14l-6 0-4 6 6 0z" fill="#4CAF50" />
    </svg>
  );
}
