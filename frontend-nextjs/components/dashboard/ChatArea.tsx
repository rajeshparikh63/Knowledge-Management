"use client";

import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { useChatStore, DocumentSource } from "@/lib/stores/chatStore";
import type { ChatMessage, KnowledgeGraph } from "@/types";
import { useDocumentStore } from "@/lib/stores/documentStore";
import { chatApi, TAKCredentials } from "@/lib/api/documents";
import { getTAKConfig } from "@/lib/api/tak";
import { usePresignedUrls } from "@/lib/hooks/usePresignedUrls";
import VideoClipViewer from "./VideoClipViewer";
import KnowledgeGraphView from "./KnowledgeGraphView";
import EmptyState from "./chat/EmptyState";
import MessageList from "./chat/MessageList";
import ChatInput from "./chat/ChatInput";
import MissionSelector, { Mission } from "./chat/MissionSelector";
import CitationTooltip from "./chat/CitationTooltip";

export default function ChatArea() {
  const {
    messages,
    sessionId,
    isLoading,
    isLoadingSession,
    inputMessage,
    selectedModel,
    takCredentials,
    takEnabled,
    setInputMessage,
    addMessage,
    updateLastMessage,
    setLoading,
    setTAKCredentials,
    setTAKEnabled,
  } = useChatStore();

  const { selectedDocs, documents } = useDocumentStore();

  // Collect all sources from messages for the presigned URL hook
  const allSources = useMemo(() => {
    const sources: DocumentSource[] = [];
    messages.forEach((m) => {
      if (m.sources) sources.push(...(m.sources as DocumentSource[]));
    });
    return sources;
  }, [messages]);

  const { urlMap: sourceUrls } = usePresignedUrls(allSources);

  // Mission management state
  const [missions, setMissions] = useState<Mission[]>([]);
  const [currentMissionId, setCurrentMissionId] = useState<string | null>(null);
  const [showNewMissionModal, setShowNewMissionModal] = useState(false);
  const [newMissionName, setNewMissionName] = useState("");

  // Video viewer state
  const [videoViewer, setVideoViewer] = useState<{
    isOpen: boolean;
    url: string;
    filename: string;
    clipStart?: number;
    clipEnd?: number;
  }>({ isOpen: false, url: "", filename: "" });

  // Knowledge-graph modal state
  const [graphView, setGraphView] = useState<KnowledgeGraph | null>(null);

  const handleOpenGraph = useCallback((message: ChatMessage) => {
    if (message.graph) setGraphView(message.graph);
  }, []);

  // Citation hover state
  const [hoveredCitation, setHoveredCitation] = useState<{
    index: number;
    messageId: string;
    x: number;
    y: number;
  } | null>(null);
  const hoverTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // --- Effects ---

  // Load missions from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem("missions");
      if (saved) setMissions(JSON.parse(saved));
    } catch {
      // ignore parse errors
    }
    const savedCurrent = localStorage.getItem("currentMissionId");
    if (savedCurrent) setCurrentMissionId(savedCurrent);
  }, []);

  // Persist missions
  useEffect(() => {
    if (missions.length > 0) {
      localStorage.setItem("missions", JSON.stringify(missions));
    }
  }, [missions]);

  // Persist current mission
  useEffect(() => {
    if (currentMissionId) {
      localStorage.setItem("currentMissionId", currentMissionId);
    }
  }, [currentMissionId]);

  // Fetch TAK configuration on mount
  useEffect(() => {
    const fetchTAKConfig = async () => {
      try {
        const config = await getTAKConfig();
        if (config && config.tak_enabled) {
          const savedPassword = localStorage.getItem("tak_password") || "";
          const credentials: TAKCredentials = {
            tak_host: config.tak_host,
            tak_port: config.tak_port,
            tak_username: config.tak_username || "",
            tak_password: savedPassword,
            agent_callsign: config.agent_callsign,
          };
          setTAKCredentials(credentials);
          setTAKEnabled(true);
        }
      } catch {
        setTAKCredentials(null);
        setTAKEnabled(false);
      }
    };
    fetchTAKConfig();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- Handlers ---

  const handleSelectMission = useCallback((mission: Mission) => {
    setCurrentMissionId(mission.id);
  }, []);

  const handleCreateMissionOpen = useCallback(() => {
    setShowNewMissionModal(true);
  }, []);

  const handleCreateMission = useCallback(() => {
    if (!newMissionName.trim()) return;
    const newMission: Mission = {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      name: newMissionName.trim(),
      createdAt: new Date().toISOString(),
    };
    setMissions((prev) => [newMission, ...prev]);
    setCurrentMissionId(newMission.id);
    setShowNewMissionModal(false);
    setNewMissionName("");
  }, [newMissionName]);

  const handleInputChange = useCallback(
    (value: string) => setInputMessage(value),
    [setInputMessage]
  );

  const handleCitationHover = useCallback(
    (index: number, messageId: string, x: number, y: number) => {
      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current);
        hoverTimeoutRef.current = null;
      }
      setHoveredCitation({ index, messageId, x, y });
    },
    []
  );

  const handleCitationLeave = useCallback(() => {
    hoverTimeoutRef.current = setTimeout(() => {
      setHoveredCitation(null);
    }, 100);
  }, []);

  const handleTooltipMouseEnter = useCallback(() => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
      hoverTimeoutRef.current = null;
    }
  }, []);

  const handleTooltipMouseLeave = useCallback(() => {
    setHoveredCitation(null);
  }, []);

  const handleCitationClick = useCallback(
    (source: DocumentSource, finalUrl: string | undefined) => {
      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current);
        hoverTimeoutRef.current = null;
      }
      setHoveredCitation(null);

      if (!finalUrl) return;

      const isVideo =
        source.filename &&
        /\.(mp4|webm|ogg|mov|avi|mkv)$/i.test(source.filename);
      const hasClipTimes =
        source.clip_start !== undefined && source.clip_end !== undefined;

      if (isVideo && hasClipTimes) {
        setVideoViewer({
          isOpen: true,
          url: finalUrl,
          filename: source.filename,
          clipStart: source.clip_start,
          clipEnd: source.clip_end,
        });
      } else {
        window.open(finalUrl, "_blank");
      }
    },
    []
  );

  const handleSendMessage = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!inputMessage.trim() || isLoading) return;

      const userMessage = inputMessage.trim();
      setInputMessage("");

      addMessage({
        id: `${Date.now()}-user`,
        role: "user",
        content: userMessage,
        timestamp: new Date().toISOString(),
      });

      addMessage({
        id: `${Date.now()}-assistant`,
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
      });

      setLoading(true);

      try {
        const documentIds = Array.from(selectedDocs);

        // Note: we used to resolve filenames here and pass them along, but
        // the backend doesn't use them anymore (the agent reads doc content
        // from search_knowledge_base via document_ids scoping). Sending only
        // ids keeps the payload small and avoids a pointless N-call lookup
        // for unloaded docs.
        const stream = await chatApi.sendMessage(
          userMessage,
          documentIds,
          sessionId,
          selectedModel,
          takEnabled && takCredentials ? takCredentials : undefined
        );

        const reader = stream.getReader();
        const decoder = new TextDecoder();
        let accumulatedContent = "";
        let accumulatedSources: DocumentSource[] = [];
        let accumulatedGraph: KnowledgeGraph | undefined = undefined;
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          buffer += chunk;

          const events = buffer.split("\n\n");
          buffer = events.pop() || "";

          for (const event of events) {
            if (!event.trim()) continue;

            const lines = event.split("\n");
            let eventType = "message";
            let eventData = "";

            for (const line of lines) {
              if (line.startsWith("event: ")) {
                eventType = line.slice(7).trim();
              } else if (line.startsWith("data: ")) {
                eventData = line.slice(6).trim();
              }
            }

            if (eventData === "[DONE]") continue;
            if (event.startsWith(": keepalive")) continue;

            try {
              const parsed = JSON.parse(eventData);

              switch (eventType) {
                case "run.started":
                  break;

                case "message.delta":
                  if (parsed.data?.content) {
                    accumulatedContent += parsed.data.content;
                    updateLastMessage(
                      accumulatedContent,
                      accumulatedSources.length > 0
                        ? accumulatedSources
                        : undefined,
                      accumulatedGraph
                    );
                  }
                  break;

                case "tool.started":
                  break;

                case "tool.completed":
                  if (
                    parsed.data?.tool_name === "search_knowledge_base" &&
                    parsed.data?.result
                  ) {
                    try {
                      const results = JSON.parse(parsed.data.result);
                      // GraphRAG returns a single-element list:
                      // [{ chunks, anchors, triples, count, query }]
                      const payload = Array.isArray(results)
                        ? results[0]
                        : results;

                      if (payload && Array.isArray(payload.chunks)) {
                        // Build the graph payload
                        accumulatedGraph = {
                          anchors: payload.anchors || [],
                          triples: payload.triples || [],
                          chunks: payload.chunks || [],
                          query: payload.query,
                        };

                        // Map chunks → sources (look up filename from doc store)
                        accumulatedSources = payload.chunks.map(
                          (c: any): DocumentSource => {
                            const doc = documents.find(
                              (d) => d.id === c.document_id
                            );
                            return {
                              document_id: c.document_id,
                              filename: doc?.file_name || "",
                              folder_name: doc?.folder_name || "",
                              text: c.text || "",
                              score:
                                typeof c.score === "number" ? c.score : 0,
                              file_key: (doc as any)?.file_key || "",
                            };
                          }
                        );
                        updateLastMessage(
                          accumulatedContent,
                          accumulatedSources,
                          accumulatedGraph
                        );
                      } else if (Array.isArray(results)) {
                        // Legacy shape fallback (flat list of {text, file_id, metadata})
                        accumulatedSources = results.map((result: any) => ({
                          document_id: result.file_id,
                          filename: result.metadata?.file_name || "",
                          folder_name: result.metadata?.folder_name || "",
                          text: result.text || "",
                          score: result.metadata?.score || 0,
                          file_key: result.metadata?.file_key || "",
                          video_id: result.metadata?.video_id,
                          video_name: result.metadata?.video_name,
                          clip_start: result.metadata?.clip_start,
                          clip_end: result.metadata?.clip_end,
                          scene_id: result.metadata?.scene_id,
                          key_frame_timestamp:
                            result.metadata?.key_frame_timestamp,
                          keyframe_file_key:
                            result.metadata?.keyframe_file_key,
                        }));
                        updateLastMessage(
                          accumulatedContent,
                          accumulatedSources,
                          accumulatedGraph
                        );
                      }
                    } catch {
                      // Failed to parse search results
                    }
                  }
                  break;

                case "message.completed":
                  if (parsed.data?.content) {
                    accumulatedContent = parsed.data.content;
                    updateLastMessage(
                      accumulatedContent,
                      accumulatedSources.length > 0
                        ? accumulatedSources
                        : undefined,
                      accumulatedGraph
                    );
                  }
                  break;

                case "run.completed":
                  break;

                case "error":
                  updateLastMessage(
                    `Error: ${parsed.data?.error || "An error occurred during processing"}`
                  );
                  break;

                default:
                  break;
              }
            } catch {
              // Failed to parse SSE event
            }
          }
        }
      } catch (error: any) {
        updateLastMessage(
          `Sorry, an error occurred: ${error.message || "Please try again."}`
        );
      } finally {
        setLoading(false);
      }
    },
    [
      inputMessage,
      isLoading,
      setInputMessage,
      addMessage,
      setLoading,
      selectedDocs,
      documents,
      sessionId,
      selectedModel,
      takEnabled,
      takCredentials,
      updateLastMessage,
    ]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage(e as any);
      }
    },
    [handleSendMessage]
  );

  return (
    <div className="flex-1 flex flex-col bg-white dark:bg-[#0a0a0a] relative min-h-0">
      {/* Loading Session Overlay */}
      {isLoadingSession && (
        <div className="absolute inset-0 bg-white/70 dark:bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center">
          <div className="w-10 h-10 border-2 border-border border-t-border dark:border-border dark:border-t-border rounded-full animate-spin" />
        </div>
      )}

      {/* Mission Selector */}
      <MissionSelector
        missions={missions}
        currentMissionId={currentMissionId}
        onSelect={handleSelectMission}
        onCreate={handleCreateMissionOpen}
      />

      {/* Content Area */}
      <div className="flex-1 flex flex-col min-h-0">
        <div className="flex-1 overflow-y-auto p-6 tactical-scrollbar min-h-0">
          {messages.length === 0 ? (
            <EmptyState />
          ) : (
            <MessageList
              messages={messages}
              sourceUrls={sourceUrls}
              onCitationHover={handleCitationHover}
              onCitationLeave={handleCitationLeave}
              onCitationClick={handleCitationClick}
              onOpenGraph={handleOpenGraph}
            />
          )}
        </div>
      </div>

      {/* Input Area */}
      <ChatInput
        inputMessage={inputMessage}
        isLoading={isLoading}
        selectedDocsCount={selectedDocs.size}
        takEnabled={takEnabled}
        takCredentials={takCredentials}
        onSend={handleSendMessage}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
      />

      {/* New Mission Modal */}
      {showNewMissionModal && (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-sm z-50 flex items-center justify-center px-4">
          <div className="bg-white dark:bg-background rounded-xl border border-border dark:border-border shadow-2xl max-w-md w-full overflow-hidden">
            <div className="px-6 py-4 border-b border-border dark:border-border">
              <h3 className="text-base font-semibold text-foreground dark:text-foreground">
                New mission
              </h3>
              <p className="text-sm text-muted-foreground dark:text-muted-foreground mt-0.5">
                Missions group related conversations and documents.
              </p>
            </div>
            <div className="p-6">
              <label className="block text-xs font-medium text-muted-foreground dark:text-foreground mb-1.5">
                Mission name
              </label>
              <input
                type="text"
                value={newMissionName}
                onChange={(e) => setNewMissionName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleCreateMission();
                  if (e.key === "Escape") {
                    setShowNewMissionModal(false);
                    setNewMissionName("");
                  }
                }}
                placeholder="e.g. Op Nightwatch"
                className="tactical-input"
                autoFocus
              />
            </div>
            <div className="border-t border-border dark:border-border px-6 py-4 flex gap-2 justify-end bg-surface-2 dark:bg-card/40">
              <button
                onClick={() => {
                  setShowNewMissionModal(false);
                  setNewMissionName("");
                }}
                className="px-4 py-2 rounded-lg text-sm font-medium text-muted-foreground dark:text-foreground hover:bg-secondary dark:hover:bg-accent transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateMission}
                disabled={!newMissionName.trim()}
                className="px-4 py-2 rounded-lg bg-brand text-brand-foreground text-sm font-medium hover:bg-brand-hover shadow-accent disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Create mission
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Video Clip Viewer Modal */}
      {videoViewer.isOpen && (
        <VideoClipViewer
          videoUrl={videoViewer.url}
          filename={videoViewer.filename}
          clipStart={videoViewer.clipStart}
          clipEnd={videoViewer.clipEnd}
          onClose={() =>
            setVideoViewer({ isOpen: false, url: "", filename: "" })
          }
        />
      )}

      {/* Knowledge Graph Viewer Modal */}
      {graphView && (
        <KnowledgeGraphView
          graph={graphView}
          onClose={() => setGraphView(null)}
        />
      )}

      {/* Citation Hover Tooltip */}
      <CitationTooltip
        citation={hoveredCitation}
        messages={messages}
        onMouseEnter={handleTooltipMouseEnter}
        onMouseLeave={handleTooltipMouseLeave}
      />
    </div>
  );
}
