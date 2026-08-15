"use client";

import React, { useEffect, useRef } from "react";
import { TAKCredentials } from "@/lib/api/documents";
import VoiceButton from "./VoiceButton";

interface ChatInputProps {
  inputMessage: string;
  isLoading: boolean;
  selectedDocsCount: number;
  takEnabled: boolean;
  takCredentials: TAKCredentials | null;
  onSend: (e: React.FormEvent) => void;
  onChange: (value: string) => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
}

const ChatInput = React.memo(function ChatInput({
  inputMessage,
  isLoading,
  selectedDocsCount,
  takEnabled,
  takCredentials,
  onSend,
  onChange,
  onKeyDown,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [inputMessage]);

  return (
    <div className="border-t border-border dark:border-border bg-white dark:bg-[#0a0a0a] px-4 py-3 flex-shrink-0">
      <div className="max-w-4xl mx-auto">
        {/* Status Bar */}
        <div className="flex items-center justify-between mb-2 text-xs">
          <div className="flex items-center gap-2 text-muted-foreground dark:text-muted-foreground">
            {selectedDocsCount > 0 ? (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                <span>
                  {selectedDocsCount} document{selectedDocsCount !== 1 ? "s" : ""} selected
                </span>
              </>
            ) : (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground dark:bg-secondary" />
                <span>General mode</span>
              </>
            )}
          </div>

          <div className="flex items-center gap-3">
            <span className="text-muted-foreground dark:text-muted-foreground">
              {isLoading ? "Thinking…" : "Ready"}
            </span>
            {takEnabled && takCredentials && (
              <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full border border-emerald-200 dark:border-emerald-900/50 bg-emerald-50 dark:bg-emerald-950/30 text-[11px] font-medium text-emerald-700 dark:text-emerald-400">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                TAK active
              </span>
            )}
          </div>
        </div>

        {/* Input Form */}
        <form onSubmit={onSend} className="relative">
          <div className="relative rounded-xl border border-border bg-white dark:bg-background focus-within:border-brand/60 focus-within:ring-2 focus-within:ring-brand/15 transition-all">
            <textarea
              ref={textareaRef}
              value={inputMessage}
              onChange={(e) => onChange(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder={
                selectedDocsCount === 0
                  ? "Ask a question…"
                  : "Ask about your selected documents…"
              }
              disabled={isLoading}
              className="w-full bg-transparent pl-4 pr-12 py-3 resize-none max-h-40 text-sm text-foreground dark:text-foreground placeholder:text-muted-foreground dark:placeholder:text-muted-foreground focus:outline-none disabled:opacity-50"
              rows={1}
            />
            <VoiceButton disabled={isLoading} />
            <button
              type="submit"
              disabled={isLoading || !inputMessage.trim()}
              className="absolute right-2 bottom-2 w-8 h-8 rounded-lg bg-brand text-brand-foreground hover:bg-brand-hover shadow-accent disabled:opacity-40 disabled:shadow-none disabled:cursor-not-allowed transition-all flex items-center justify-center"
              aria-label="Send"
            >
              {isLoading ? (
                <div className="w-3.5 h-3.5 border-2 border-current/30 border-t-current rounded-full animate-spin" />
              ) : (
                <svg
                  className="w-4 h-4"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M5 12h14M13 5l7 7-7 7" />
                </svg>
              )}
            </button>
          </div>
        </form>

        <div className="mt-2 text-[11px] text-muted-foreground dark:text-muted-foreground text-center">
          <kbd className="font-mono">Enter</kbd> to send ·{" "}
          <kbd className="font-mono">Shift</kbd> + <kbd className="font-mono">Enter</kbd> for new line
        </div>
      </div>
    </div>
  );
});

export default ChatInput;
