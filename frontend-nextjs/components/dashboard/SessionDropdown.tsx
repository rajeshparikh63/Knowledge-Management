"use client";

import { useState, useEffect, useRef } from "react";
import { chatSessionsApi, ChatSession } from "@/lib/api/documents";
import { useChatStore } from "@/lib/stores/chatStore";
import { useAuthStore } from "@/lib/stores/authStore";
import { useChatSessions } from "@/lib/hooks/useChatSessions";

interface SessionDropdownProps {
  hasUserMessages: boolean;
}

export default function SessionDropdown({ hasUserMessages }: SessionDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const user = useAuthStore((s) => s.user);
  const { data: sessions = [], isLoading, refetch } = useChatSessions(user?.id);
  const { startNewSession, loadSession, setLoadingSession } = useChatStore();

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen]);

  const [loadError, setLoadError] = useState<string | null>(null);

  const handleSessionSelect = async (session: ChatSession) => {
    try {
      setLoadingSession(true);
      setIsOpen(false);
      setLoadError(null);

      const sessionHistory = await chatSessionsApi.getSession(session.session_id);

      const formattedMessages = sessionHistory.messages
        .filter((msg) => msg.role === "user" || msg.role === "assistant")
        .map((msg, idx) => ({
          id: `${msg.created_at}-${idx}`,
          role: msg.role as "user" | "assistant",
          content: msg.content,
          timestamp: new Date(msg.created_at * 1000).toISOString(),
          ...(msg.model && { model: msg.model }),
          ...(msg.sources && { sources: msg.sources }),
        }));

      loadSession(session.session_id, formattedMessages);
    } catch (err: any) {
      setLoadError(err.message || "Failed to load session");
    } finally {
      setLoadingSession(false);
    }
  };

  const handleNewSession = () => {
    startNewSession();
    setIsOpen(false);
    refetch();
  };

  const handleDeleteSession = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();

    if (!confirm("Delete this session?")) return;

    try {
      await chatSessionsApi.deleteSession(sessionId);
      refetch();
    } catch (err: any) {
      alert(err.message || "Failed to delete session");
    }
  };

  const buttonText = hasUserMessages ? "New session" : "Sessions";

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="inline-flex items-center gap-1.5 px-2.5 h-8 rounded-lg border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 hover:bg-zinc-50 dark:hover:bg-zinc-900 transition-colors text-xs font-medium text-zinc-700 dark:text-zinc-300"
      >
        {hasUserMessages ? (
          <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 5v14M5 12h14" />
          </svg>
        ) : (
          <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <path d="M12 6v6l4 2" />
          </svg>
        )}
        <span>{buttonText}</span>
        <svg
          className={`w-3 h-3 text-zinc-500 transition-transform ${isOpen ? "rotate-180" : ""}`}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M6 9l6 6 6-6" />
        </svg>
      </button>

      {isOpen && (
        <div className="absolute top-full mt-1.5 left-0 w-80 rounded-xl bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 shadow-xl z-50 max-h-[420px] overflow-hidden flex flex-col">
          {hasUserMessages && (
            <button
              onClick={handleNewSession}
              className="w-full text-left px-4 py-2.5 hover:bg-zinc-50 dark:hover:bg-zinc-900 transition-colors border-b border-zinc-200 dark:border-zinc-800 flex items-center gap-2"
            >
              <svg className="w-4 h-4 text-zinc-900 dark:text-zinc-100" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 5v14M5 12h14" />
              </svg>
              <span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
                Start a new session
              </span>
            </button>
          )}

          <div className="px-4 py-2 text-[11px] font-medium text-zinc-500 uppercase tracking-wider border-b border-zinc-200 dark:border-zinc-800">
            Recent sessions
          </div>

          <div className="flex-1 overflow-y-auto tactical-scrollbar">
            {isLoading && sessions.length === 0 && (
              <div className="px-4 py-8 text-center">
                <div className="w-5 h-5 border-2 border-zinc-300 border-t-zinc-700 dark:border-zinc-700 dark:border-t-zinc-300 rounded-full animate-spin mx-auto" />
                <p className="text-xs text-zinc-500 mt-2">Loading</p>
              </div>
            )}

            {loadError && (
              <div className="px-4 py-3 text-xs text-red-600 dark:text-red-400">
                {loadError}
              </div>
            )}

            {!isLoading && sessions.length === 0 && !loadError && (
              <div className="px-4 py-8 text-center text-xs text-zinc-500">
                No previous sessions
              </div>
            )}

            {sessions.length > 0 && (
              <div>
                {sessions.map((session) => (
                  <button
                    key={session.session_id}
                    onClick={() => handleSessionSelect(session)}
                    className="w-full text-left px-4 py-2.5 hover:bg-zinc-50 dark:hover:bg-zinc-900 transition-colors group border-b border-zinc-100 dark:border-zinc-900 last:border-0"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium text-zinc-900 dark:text-zinc-100 truncate">
                          {session.name}
                        </div>
                        {session.message_preview && (
                          <div className="text-xs text-zinc-500 dark:text-zinc-500 truncate mt-0.5">
                            {session.message_preview}
                          </div>
                        )}
                        <div className="text-[11px] text-zinc-400 dark:text-zinc-600 mt-1">
                          {new Date(session.updated_at * 1000).toLocaleString()}
                        </div>
                      </div>

                      <button
                        onClick={(e) => handleDeleteSession(session.session_id, e)}
                        className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-red-50 dark:hover:bg-red-950/30 flex-shrink-0"
                        title="Delete session"
                      >
                        <svg className="w-3.5 h-3.5 text-red-600 dark:text-red-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6" />
                        </svg>
                      </button>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
