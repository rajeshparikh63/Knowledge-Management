"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useDocumentStore } from "@/lib/stores/documentStore";
import { mindmapApi, MindMapResponse } from "@/lib/api/mindmap";
import { flashcardsApi, FlashcardData } from "@/lib/api/flashcards";
import MindMapViewer from "./MindMapViewer";
import ReportStudio from "./ReportStudio";
import FlashcardViewer from "./FlashcardViewer";
import PodcastGenerator from "../PodcastGenerator";

interface WorkflowOption {
  id: string;
  title: string;
  icon: React.ReactNode;
  available: boolean;
  description: string;
}

interface WorkflowPanelProps {
  isCollapsed?: boolean;
  onToggleCollapse?: (collapsed: boolean) => void;
}

export default function WorkflowPanel({
  isCollapsed: externalCollapsed,
  onToggleCollapse,
}: WorkflowPanelProps = {}) {
  const [selectedWorkflow, setSelectedWorkflow] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [mindMapData, setMindMapData] = useState<MindMapResponse | null>(null);
  const [showReportStudio, setShowReportStudio] = useState(false);
  const [flashcardData, setFlashcardData] = useState<FlashcardData | null>(null);
  const [showPodcastGenerator, setShowPodcastGenerator] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { selectedDocs } = useDocumentStore();
  const [internalCollapsed, setInternalCollapsed] = useState(false);

  const isCollapsed = externalCollapsed !== undefined ? externalCollapsed : internalCollapsed;
  const setIsCollapsed = (value: boolean) => {
    if (onToggleCollapse) {
      onToggleCollapse(value);
    } else {
      setInternalCollapsed(value);
    }
  };

  const icon = (path: React.ReactNode) => (
    <svg
      className="w-5 h-5"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {path}
    </svg>
  );

  const workflows: WorkflowOption[] = [
    {
      id: "audio-overview",
      title: "Audio overview",
      available: true,
      description: "Generate an AI-powered podcast discussion from your documents.",
      icon: icon(
        <>
          <path d="M12 2a3 3 0 00-3 3v6a3 3 0 006 0V5a3 3 0 00-3-3z" />
          <path d="M19 11a7 7 0 01-14 0M12 19v3" />
        </>
      ),
    },
    {
      id: "video-overview",
      title: "Video overview",
      available: false,
      description: "Create a video summary with visuals.",
      icon: icon(
        <>
          <rect x="2" y="6" width="14" height="12" rx="2" />
          <path d="M22 8l-6 4 6 4V8z" />
        </>
      ),
    },
    {
      id: "mind-map",
      title: "Mind map",
      available: true,
      description: "Generate an interactive mind map from your knowledge base.",
      icon: icon(
        <>
          <circle cx="5" cy="6" r="2" />
          <circle cx="5" cy="18" r="2" />
          <circle cx="19" cy="12" r="2" />
          <path d="M7 6h4a2 2 0 012 2v8a2 2 0 01-2 2H7M13 12h4" />
        </>
      ),
    },
    {
      id: "reports",
      title: "Reports",
      available: true,
      description: "Generate comprehensive reports and summaries.",
      icon: icon(
        <>
          <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
          <path d="M14 2v6h6M8 13h8M8 17h5" />
        </>
      ),
    },
    {
      id: "flashcards",
      title: "Flashcards",
      available: true,
      description: "Create study flashcards from your documents.",
      icon: icon(
        <>
          <rect x="3" y="5" width="18" height="14" rx="2" />
          <path d="M7 10h10M7 14h6" />
        </>
      ),
    },
    {
      id: "quiz",
      title: "Quiz",
      available: false,
      description: "Generate interactive quizzes.",
      icon: icon(
        <>
          <circle cx="12" cy="12" r="10" />
          <path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3M12 17h.01" />
        </>
      ),
    },
    {
      id: "infographic",
      title: "Infographic",
      available: false,
      description: "Design visual infographics.",
      icon: icon(
        <>
          <rect x="3" y="3" width="18" height="18" rx="2" />
          <path d="M7 13v4M12 9v8M17 5v12" />
        </>
      ),
    },
    {
      id: "slide-deck",
      title: "Slide deck",
      available: false,
      description: "Create presentation slides automatically.",
      icon: icon(
        <>
          <rect x="2" y="3" width="20" height="14" rx="2" />
          <path d="M8 21h8M12 17v4" />
        </>
      ),
    },
  ];

  const handleWorkflowClick = async (workflow: WorkflowOption) => {
    if (!workflow.available) return;

    setSelectedWorkflow(workflow.id);
    setError(null);

    if (workflow.id === "mind-map") {
      if (selectedDocs.size === 0) {
        setError("Select at least one document to generate a mind map.");
        setTimeout(() => setError(null), 5000);
        return;
      }
      try {
        setIsGenerating(true);
        const documentIds = Array.from(selectedDocs);
        const response = await mindmapApi.generate(documentIds);
        setMindMapData(response);
        setIsGenerating(false);
      } catch (err: any) {
        setError(err.response?.data?.detail || "Failed to generate mind map.");
        setIsGenerating(false);
        setTimeout(() => setError(null), 5000);
      }
    }

    if (workflow.id === "flashcards") {
      if (selectedDocs.size === 0) {
        setError("Select at least one document to generate flashcards.");
        setTimeout(() => setError(null), 5000);
        return;
      }
      try {
        setIsGenerating(true);
        const documentIds = Array.from(selectedDocs);
        const response = await flashcardsApi.generate(documentIds);
        setFlashcardData(response);
        setIsGenerating(false);
      } catch (err: any) {
        setError(err.response?.data?.detail || "Failed to generate flashcards.");
        setIsGenerating(false);
        setTimeout(() => setError(null), 5000);
      }
    }

    if (workflow.id === "reports") {
      setShowReportStudio(true);
    }

    if (workflow.id === "audio-overview") {
      if (selectedDocs.size === 0) {
        setError("Select at least one document to generate an audio overview.");
        setTimeout(() => setError(null), 5000);
        return;
      }
      setShowPodcastGenerator(true);
    }
  };

  const handleCloseMindMap = () => {
    setMindMapData(null);
    setSelectedWorkflow(null);
  };
  const handleCloseFlashcards = () => {
    setFlashcardData(null);
    setSelectedWorkflow(null);
  };
  const handleCloseReportStudio = () => {
    setShowReportStudio(false);
    setSelectedWorkflow(null);
  };
  const handleClosePodcast = () => {
    setShowPodcastGenerator(false);
    setSelectedWorkflow(null);
  };

  // Collapsed
  if (isCollapsed) {
    return (
      <div className="w-14 bg-white dark:bg-[#0a0a0a] border-l border-zinc-200 dark:border-zinc-800 flex flex-col relative">
        <button
          onClick={() => setIsCollapsed(false)}
          className="w-full h-12 hover:bg-zinc-50 dark:hover:bg-zinc-900 transition-colors border-b border-zinc-200 dark:border-zinc-800 flex items-center justify-center"
          title="Expand workflows"
        >
          <svg className="w-4 h-4 text-zinc-600 dark:text-zinc-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M11 19l-7-7 7-7M19 19l-7-7 7-7" />
          </svg>
        </button>

        <div className="flex-1 overflow-y-auto tactical-scrollbar">
          <div className="flex flex-col gap-1 p-2">
            {workflows.map((workflow) => (
              <button
                key={workflow.id}
                onClick={() => handleWorkflowClick(workflow)}
                disabled={!workflow.available}
                className={`relative w-10 h-10 rounded-lg flex items-center justify-center transition-all ${
                  workflow.available
                    ? "text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-900"
                    : "text-zinc-300 dark:text-zinc-700 cursor-not-allowed"
                }`}
                title={workflow.title + (workflow.available ? "" : " (coming soon)")}
              >
                {workflow.icon}
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 bg-white dark:bg-[#0a0a0a] border-l border-zinc-200 dark:border-zinc-800 flex flex-col relative">
      {/* Header */}
      <div className="border-b border-zinc-200 dark:border-zinc-800 px-4 py-3">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xs font-semibold tracking-wider uppercase text-zinc-500 dark:text-zinc-400">
              Workflows
            </h2>
            <p className="text-[11px] text-zinc-400 dark:text-zinc-500 mt-0.5">
              Generate intelligence products
            </p>
          </div>
          <button
            onClick={() => setIsCollapsed(true)}
            className="p-1.5 rounded-md text-zinc-500 hover:text-zinc-900 dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-zinc-900 transition-colors"
            title="Collapse"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M13 19l7-7-7-7M5 19l7-7-7-7" />
            </svg>
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-3 tactical-scrollbar">
        <div className="grid grid-cols-2 gap-2">
          {workflows.map((workflow, idx) => (
            <motion.button
              key={workflow.id}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2, delay: idx * 0.03 }}
              onClick={() => handleWorkflowClick(workflow)}
              disabled={!workflow.available}
              className={`relative rounded-xl border p-3 text-left transition-all group ${
                workflow.available
                  ? "border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 hover:border-zinc-300 dark:hover:border-zinc-700 hover:bg-zinc-50 dark:hover:bg-zinc-900/60 cursor-pointer"
                  : "border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950/60 opacity-60 cursor-not-allowed"
              }`}
              title={workflow.description}
            >
              {!workflow.available && (
                <span className="absolute top-2 right-2 px-1.5 py-0.5 rounded-md bg-zinc-200 dark:bg-zinc-800 text-[9px] font-medium text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">
                  Soon
                </span>
              )}

              <div
                className={`mb-2 ${
                  workflow.available
                    ? "text-zinc-700 dark:text-zinc-300"
                    : "text-zinc-400 dark:text-zinc-600"
                }`}
              >
                {workflow.icon}
              </div>

              <div
                className={`text-[13px] font-medium leading-tight ${
                  workflow.available
                    ? "text-zinc-900 dark:text-zinc-100"
                    : "text-zinc-500 dark:text-zinc-500"
                }`}
              >
                {workflow.title}
              </div>
            </motion.button>
          ))}
        </div>

        <AnimatePresence>
          {selectedDocs.size > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2 }}
            >
              <div className="mt-3 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/40 p-3">
                <div className="flex items-center gap-2 mb-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                  <span className="text-xs font-medium text-zinc-900 dark:text-zinc-100">
                    {selectedDocs.size} document{selectedDocs.size !== 1 ? "s" : ""} selected
                  </span>
                </div>
                <p className="text-xs text-zinc-500 dark:text-zinc-400">
                  Ready to generate intelligence products.
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {error && (
          <div className="mt-3 rounded-xl border border-red-200 dark:border-red-900/50 bg-red-50 dark:bg-red-950/30 p-3">
            <div className="text-xs font-medium text-red-700 dark:text-red-400 mb-0.5">
              Error
            </div>
            <div className="text-xs text-red-600 dark:text-red-400/90 leading-relaxed">
              {error}
            </div>
          </div>
        )}
      </div>

      {isGenerating && (
        <div className="absolute inset-0 bg-white/80 dark:bg-black/70 backdrop-blur-sm flex items-center justify-center z-10">
          <div className="text-center">
            <div className="inline-block w-8 h-8 border-2 border-zinc-300 border-t-zinc-900 dark:border-zinc-700 dark:border-t-zinc-100 rounded-full animate-spin mb-3" />
            <p className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
              {selectedWorkflow === "audio-overview" && "Generating audio overview"}
              {selectedWorkflow === "mind-map" && "Generating mind map"}
              {selectedWorkflow === "flashcards" && "Generating flashcards"}
              {selectedWorkflow === "reports" && "Generating report"}
            </p>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">
              Analyzing {selectedDocs.size} document{selectedDocs.size !== 1 ? "s" : ""}
            </p>
          </div>
        </div>
      )}

      {mindMapData && <MindMapViewer mindMapData={mindMapData} onClose={handleCloseMindMap} />}
      {flashcardData && (
        <FlashcardViewer flashcardData={flashcardData} onClose={handleCloseFlashcards} />
      )}
      {showReportStudio && <ReportStudio onClose={handleCloseReportStudio} />}
      {showPodcastGenerator && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
          <PodcastGenerator
            selectedDocumentIds={Array.from(selectedDocs)}
            onClose={handleClosePodcast}
          />
        </div>
      )}
    </div>
  );
}
