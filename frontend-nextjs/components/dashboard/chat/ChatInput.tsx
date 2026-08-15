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
    <div className="border-t border-zinc-200 dark:border-zinc-800 bg-white dark:bg-[#0a0a0a] px-4 py-3 flex-shrink-0">
      <div className="max-w-4xl mx-auto">
        {/* Status Bar */}
        <div className="flex items-center justify-between mb-2 text-xs">
          <div className="flex items-center gap-2 text-zinc-500 dark:text-zinc-400">
            {selectedDocsCount > 0 ? (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                <span>
                  {selectedDocsCount} document{selectedDocsCount !== 1 ? "s" : ""} selected
                </span>
              </>
            ) : (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-zinc-400 dark:bg-zinc-600" />
                <span>General mode</span>
              </>
            )}
          </div>

          <div className="flex items-center gap-3">
            <span className="text-zinc-500 dark:text-zinc-400">
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
          <div className="relative rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 focus-within:border-zinc-400 dark:focus-within:border-zinc-600 focus-within:ring-2 focus-within:ring-zinc-900/5 dark:focus-within:ring-white/5 transition-all">
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
              className="w-full bg-transparent pl-4 pr-12 py-3 resize-none max-h-40 text-sm text-zinc-900 dark:text-zinc-100 placeholder:text-zinc-400 dark:placeholder:text-zinc-600 focus:outline-none disabled:opacity-50"
              rows={1}
            />
            <VoiceButton disabled={isLoading} />
            <button
              type="submit"
              disabled={isLoading || !inputMessage.trim()}
              className="absolute right-2 bottom-2 w-8 h-8 rounded-lg bg-zinc-900 text-white hover:bg-zinc-800 dark:bg-white dark:text-zinc-900 dark:hover:bg-zinc-200 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
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

        <div className="mt-2 text-[11px] text-zinc-400 dark:text-zinc-600 text-center">
          <kbd className="font-mono">Enter</kbd> to send ·{" "}
          <kbd className="font-mono">Shift</kbd> + <kbd className="font-mono">Enter</kbd> for new line
        </div>
      </div>
    </div>
  );
});

export default ChatInput;
