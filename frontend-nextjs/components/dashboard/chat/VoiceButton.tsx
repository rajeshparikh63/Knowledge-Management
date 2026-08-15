"use client";

import React from "react";
import { Mic } from "lucide-react";

import { useDocumentStore } from "@/lib/stores/documentStore";
import { useVoiceStore } from "@/lib/stores/voiceStore";

/**
 * Mic button rendered next to the Send button in ChatInput. Reads the current
 * selected documents from `useDocumentStore` and kicks off a voice session via
 * the voice store. The actual full-screen call UI is mounted separately in
 * <VoiceSession /> at the dashboard root.
 */
const VoiceButton = React.memo(function VoiceButton({ disabled }: { disabled?: boolean }) {
  const selectedDocs = useDocumentStore((s) => s.selectedDocs);
  const documents = useDocumentStore((s) => s.documents);
  const voiceState = useVoiceStore((s) => s.state);
  const start = useVoiceStore((s) => s.start);

  const busy = voiceState === "fetching-token" || voiceState === "connected";

  const handleClick = () => {
    if (busy) return;
    const ids = Array.from(selectedDocs);
    const names = documents
      .filter((d) => selectedDocs.has(d.id))
      .map((d) => d.file_name);
    start({ document_ids: ids, file_names: names });
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={disabled || busy}
      aria-label="Start voice call"
      title={
        selectedDocs.size === 0
          ? "Start voice call (no documents selected)"
          : `Start voice call (${selectedDocs.size} document${selectedDocs.size === 1 ? "" : "s"})`
      }
      className="absolute right-11 bottom-2 w-8 h-8 rounded-lg text-zinc-500 hover:text-zinc-900 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:text-zinc-100 dark:hover:bg-zinc-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
    >
      {busy ? (
        <div className="w-3.5 h-3.5 border-2 border-current/30 border-t-current rounded-full animate-spin" />
      ) : (
        <Mic className="w-4 h-4" />
      )}
    </button>
  );
});

export default VoiceButton;
