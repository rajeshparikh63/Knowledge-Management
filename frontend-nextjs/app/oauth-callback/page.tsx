"use client";

/**
 * OAuth callback landing page for Composio-managed connectors (SharePoint).
 *
 * Composio runs the hosted OAuth flow in a popup and redirects here only AFTER
 * it finishes — so simply arriving here (without an `error` query param) means
 * the connection succeeded. No status polling needed. We then:
 *   - if opened as a popup, message the opener and close ourselves;
 *   - otherwise, bounce back to the dashboard.
 */

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function CallbackInner() {
  const router = useRouter();
  const params = useSearchParams();
  const [msg, setMsg] = useState("Finishing the SharePoint connection…");

  useEffect(() => {
    const connected = !params.get("error");
    setMsg(connected ? "Connected!" : "Connection was not completed.");

    if (window.opener) {
      try {
        window.opener.postMessage(
          { type: "sharepoint-oauth-result", connected },
          window.location.origin
        );
      } catch {
        /* ignore */
      }
      window.close();
      // Some browsers block close() for non-script-opened windows.
      setMsg("You can close this window.");
      return;
    }

    // Full-page fallback (popup was blocked): bounce back to the dashboard.
    let cancelled = false;
    (async () => {
      await sleep(800);
      if (!cancelled) router.replace("/dashboard");
    })();
    return () => {
      cancelled = true;
    };
  }, [router, params]);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center gap-3 bg-background text-foreground">
      <div className="w-5 h-5 border-2 border-brand/30 border-t-amber-400 rounded-full animate-spin" />
      <div className="text-sm text-muted-foreground">{msg}</div>
    </div>
  );
}

export default function OAuthCallback() {
  return (
    <Suspense fallback={<div className="min-h-screen flex items-center justify-center bg-background text-muted-foreground">Loading…</div>}>
      <CallbackInner />
    </Suspense>
  );
}
