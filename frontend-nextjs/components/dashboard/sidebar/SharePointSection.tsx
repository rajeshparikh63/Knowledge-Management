"use client";

/**
 * SharePoint connector (Composio-backed). Mirrors GoogleDriveSection:
 *   - Connect via a popup pointed at Composio's hosted OAuth flow.
 *   - /oauth-callback (loaded in the popup) messages us when ACTIVE.
 *   - Once connected: "Choose libraries" opens a picker, "Disconnect" revokes.
 *
 * No tenant/subdomain input here — Composio's hosted page collects it.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { sharepointApi } from "@/lib/api/sharepoint";
import SharePointLibraryPickerModal from "./SharePointLibraryPickerModal";

interface Props {
  onIngestStarted?: () => void;
}

export default function SharePointSection({ onIngestStarted }: Props) {
  const [status, setStatus] = useState<{ connected: boolean; loaded: boolean }>({
    connected: false,
    loaded: false,
  });
  const [busy, setBusy] = useState<null | "connect" | "disconnect">(null);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshStatus = useCallback(async () => {
    try {
      const s = await sharepointApi.status();
      setStatus({ connected: !!s.connected, loaded: true });
    } catch {
      setStatus((p) => ({ ...p, loaded: true }));
    }
  }, []);

  useEffect(() => {
    refreshStatus();
  }, [refreshStatus]);

  const startConnect = useCallback(async () => {
    setBusy("connect");
    setError(null);
    try {
      const callbackUrl = `${window.location.origin}/oauth-callback`;
      const { auth_url } = await sharepointApi.connect(callbackUrl);
      const popup = window.open(auth_url, "sharepoint-oauth", "width=600,height=760");
      if (!popup) {
        // Popup blocked → full-page redirect; /oauth-callback bounces back.
        window.location.href = auth_url;
        return;
      }

      const onMessage = (e: MessageEvent) => {
        if (e.origin !== window.location.origin) return;
        if (e.data?.type !== "sharepoint-oauth-result") return;
        window.removeEventListener("message", onMessage);
        if (poll) clearInterval(poll);
        try {
          popup.close();
        } catch {
          /* ignore */
        }
        refreshStatus();
        setBusy(null);
      };
      window.addEventListener("message", onMessage);

      // Fallback: poll status in case the popup can't postMessage (and detect
      // manual close).
      const poll = setInterval(async () => {
        try {
          const s = await sharepointApi.status();
          if (s.connected) {
            clearInterval(poll);
            window.removeEventListener("message", onMessage);
            try {
              popup.close();
            } catch {
              /* ignore */
            }
            setStatus({ connected: true, loaded: true });
            setBusy(null);
          }
        } catch {
          /* keep polling */
        }
        if (popup.closed) {
          clearInterval(poll);
          window.removeEventListener("message", onMessage);
          refreshStatus();
          setBusy(null);
        }
      }, 1500);
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || "Connect failed");
      setBusy(null);
    }
  }, [refreshStatus]);

  const onDisconnect = useCallback(async () => {
    if (!confirm("Disconnect SharePoint? Already-ingested docs stay.")) return;
    setBusy("disconnect");
    try {
      await sharepointApi.disconnect();
      setStatus({ connected: false, loaded: true });
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || "Disconnect failed");
    } finally {
      setBusy(null);
    }
  }, []);

  const handleLibrariesIngested = useCallback(() => {
    setPickerOpen(false);
    onIngestStarted?.();
  }, [onIngestStarted]);

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <SharePointIcon className="w-5 h-5" />
        <div className="text-xs text-slate-300 font-semibold tracking-wide">
          Microsoft SharePoint
        </div>
      </div>

      {!status.loaded ? (
        <div className="text-[11px] text-slate-500">Checking connection…</div>
      ) : status.connected ? (
        <div className="space-y-2">
          <div className="text-[11px] text-emerald-400">✓ Connected</div>
          <button
            onClick={() => setPickerOpen(true)}
            className="w-full py-2 text-[11px] font-semibold tracking-wider bg-amber-400 text-slate-900 hover:bg-amber-300 transition-colors"
          >
            CHOOSE LIBRARIES
          </button>
          <button
            onClick={onDisconnect}
            disabled={busy === "disconnect"}
            className="w-full py-1.5 text-[10px] text-slate-400 hover:text-slate-200 disabled:opacity-50"
          >
            {busy === "disconnect" ? "Disconnecting…" : "Disconnect"}
          </button>
        </div>
      ) : (
        <button
          onClick={startConnect}
          disabled={busy === "connect"}
          className="w-full py-2 text-[11px] font-semibold tracking-wider bg-amber-400 text-slate-900 hover:bg-amber-300 disabled:opacity-50 transition-colors"
        >
          {busy === "connect" ? "Opening Microsoft…" : "CONNECT SHAREPOINT"}
        </button>
      )}

      {error && <div className="text-[10px] text-red-400">{error}</div>}

      <SharePointLibraryPickerModal
        isOpen={pickerOpen}
        onClose={() => setPickerOpen(false)}
        onIngestStarted={handleLibrariesIngested}
      />
    </div>
  );
}

function SharePointIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="9" cy="7" r="4" fill="#036C70" />
      <circle cx="15.5" cy="11" r="4.5" fill="#1A9BA1" />
      <circle cx="11.5" cy="16.5" r="4" fill="#37C6D0" />
    </svg>
  );
}
