"use client";

import { useState, useEffect } from "react";
import { useDocumentStore } from "@/lib/stores/documentStore";
import { useChatStore } from "@/lib/stores/chatStore";
import { HARDCODED_FORMATS, ReportFormat } from "@/lib/constants/reportFormats";
import { reportsApi, SuggestionsResponse, FormatSuggestion } from "@/lib/api/reports";
import FormatCard from "./FormatCard";
import PromptEditor from "./PromptEditor";
import ReportViewer from "./ReportViewer";

const generateId = () => `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

interface ReportStudioProps {
  onClose: () => void;
}

export default function ReportStudio({ onClose }: ReportStudioProps) {
  const { selectedDocs } = useDocumentStore();
  const { addMessage, updateLastMessage, setLoading } = useChatStore();

  const [selectedFormat, setSelectedFormat] = useState<ReportFormat | null>(null);
  const [customPrompt, setCustomPrompt] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showCustomPrompt, setShowCustomPrompt] = useState(false);
  const [customFormatPrompt, setCustomFormatPrompt] = useState("");
  const [showPromptEditor, setShowPromptEditor] = useState(false);

  // Report Viewer State
  const [showReportViewer, setShowReportViewer] = useState(false);
  const [reportContent, setReportContent] = useState("");
  const [reportTitle, setReportTitle] = useState("");

  // AI Suggestions State
  const [aiSuggestions, setAiSuggestions] = useState<FormatSuggestion[]>([]);
  const [suggestionsStatus, setSuggestionsStatus] = useState<'idle' | 'loading' | 'completed' | 'failed'>('idle');
  const [isTriggeringAI, setIsTriggeringAI] = useState(false);

  // Convert selected docs to array
  const documentIds = Array.from(selectedDocs);

  // Trigger AI suggestions when documents are selected
  useEffect(() => {
    if (documentIds.length > 0 && suggestionsStatus === 'idle') {
      handleTriggerAISuggestions();
    }
  }, [documentIds.length]);

  // Poll for AI suggestions
  useEffect(() => {
    if (suggestionsStatus === 'loading') {
      const pollInterval = setInterval(async () => {
        try {
          const response = await reportsApi.getSuggestions(documentIds);

          if (response.status === 'completed' && response.suggestions) {
            setAiSuggestions(response.suggestions);
            setSuggestionsStatus('completed');
            clearInterval(pollInterval);
          } else if (response.status === 'failed') {
            setSuggestionsStatus('failed');
            setError('Failed to generate AI suggestions');
            clearInterval(pollInterval);
            setTimeout(() => setError(null), 5000);
          }
        } catch (err: any) {
          // Polling error - will retry on next interval
        }
      }, 2000);

      return () => clearInterval(pollInterval);
    }
  }, [suggestionsStatus, documentIds]);

  const handleTriggerAISuggestions = async () => {
    if (documentIds.length === 0) return;

    try {
      setIsTriggeringAI(true);
      setSuggestionsStatus('loading');
      await reportsApi.triggerFormatSuggestions(documentIds);
    } catch (err: any) {
      // AI suggestions trigger failed
      setSuggestionsStatus('failed');
      setError(err.response?.data?.detail || 'Failed to trigger AI suggestions');
      setTimeout(() => setError(null), 5000);
    } finally {
      setIsTriggeringAI(false);
    }
  };

  const handleFormatSelect = (format: ReportFormat) => {
    setSelectedFormat(format);
    setCustomPrompt(format.prompt);
    setShowPromptEditor(true);
    setShowCustomPrompt(false);
  };

  const handleFormatEdit = (format: ReportFormat) => {
    setSelectedFormat(format);
    setCustomPrompt(format.prompt);
    setShowPromptEditor(true);
    setShowCustomPrompt(false);
  };

  const handleCreateYourOwn = () => {
    setSelectedFormat(null);
    setShowCustomPrompt(true);
    setShowPromptEditor(true);
    setCustomPrompt("");
  };

  const handleBackToFormats = () => {
    setShowPromptEditor(false);
    setShowCustomPrompt(false);
    setSelectedFormat(null);
    setCustomPrompt("");
    setCustomFormatPrompt("");
  };

  const handleGenerateReport = async () => {
    const promptToUse = showCustomPrompt ? customFormatPrompt : customPrompt;

    if (!promptToUse.trim() || documentIds.length === 0) {
      return;
    }

    try {
      setIsGenerating(true);
      setLoading(true);
      setError(null);

      // Store report name for the viewer title
      const reportName = showCustomPrompt ? 'Custom' : selectedFormat?.name || 'Custom';
      setReportTitle(reportName + ' Report');

      // Start streaming
      const stream = await reportsApi.generateReport(documentIds, promptToUse);
      const reader = stream.getReader();
      const decoder = new TextDecoder();

      let fullReportContent = '';
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');

        // Keep the last incomplete line in the buffer
        buffer = lines.pop() || '';

        let i = 0;
        while (i < lines.length) {
          const line = lines[i];

          if (line.startsWith('event: ')) {
            const event = line.substring(7).trim();

            // Next line should be data
            if (i + 1 < lines.length && lines[i + 1].startsWith('data: ')) {
              try {
                const data = JSON.parse(lines[i + 1].substring(6));

                if (event === 'report') {
                  fullReportContent = data.content || '';
                } else if (event === 'error') {
                  throw new Error(data.error || 'Unknown error occurred');
                }
              } catch (parseError) {
                // SSE parse error - continue processing
              }

              i += 2; // Skip event and data lines
              continue;
            }
          }

          i++;
        }
      }

      setIsGenerating(false);
      setLoading(false);

      // Show report in viewer modal
      setReportContent(fullReportContent);
      setShowReportViewer(true);

    } catch (err: any) {
      // Report generation failed
      setError(err.message || 'Failed to generate report');
      setIsGenerating(false);
      setLoading(false);
      setTimeout(() => setError(null), 5000);
    }
  };

  const handleCloseReportViewer = () => {
    setShowReportViewer(false);
    setReportContent("");
    setReportTitle("");
    // Close the entire Report Studio as well
    onClose();
  };

  // No documents selected
  if (documentIds.length === 0) {
    return (
      <>
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-200/90 dark:bg-slate-950/90 backdrop-blur-sm">
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg p-8 max-w-md mx-4">
          <div className="text-center">
            <svg className="w-16 h-16 text-slate-400 dark:text-slate-700 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <h3 className="text-sm font-bold text-slate-600 dark:text-slate-400 tracking-wider mb-2">
              NO DOCUMENTS SELECTED
            </h3>
            <p className="text-[11px] text-slate-500 mb-6">
              Select one or more documents from the sidebar to generate reports
            </p>
            <button
              onClick={onClose}
              className="px-4 py-2 bg-slate-200 dark:bg-slate-800 hover:bg-slate-300 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 text-sm font-semibold tracking-wider transition-colors"
            >
              CLOSE
            </button>
          </div>
        </div>
        </div>

        {/* Report Viewer Modal */}
        {showReportViewer && (
          <ReportViewer
            reportContent={reportContent}
            reportTitle={reportTitle}
            onClose={handleCloseReportViewer}
          />
        )}
      </>
    );
  }

  // Show prompt editor view
  if (showPromptEditor) {
    return (
      <>
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-200/90 dark:bg-slate-950/90 backdrop-blur-sm p-4">
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg max-w-3xl w-full max-h-[90vh] overflow-hidden flex flex-col">
          {/* Header */}
          <div className="border-b border-slate-200 dark:border-slate-800 p-4 flex items-center justify-between bg-slate-50 dark:bg-slate-900/50">
            <div className="flex items-center gap-3">
              <button
                onClick={handleBackToFormats}
                className="p-1 hover:bg-slate-200 dark:hover:bg-slate-800 rounded transition-colors"
                title="Back to formats"
              >
                <svg className="w-5 h-5 text-slate-400 hover:text-blue-600 dark:hover:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
              </button>
              <div className="flex items-center gap-2">
                <svg className="w-5 h-5 text-blue-600 dark:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                </svg>
                <h2 className="text-sm font-bold text-blue-700 dark:text-amber-400 tracking-wider">
                  {showCustomPrompt ? 'CREATE YOUR OWN' : selectedFormat?.name.toUpperCase()}
                </h2>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-1 hover:bg-slate-200 dark:hover:bg-slate-800 rounded transition-colors"
              title="Close"
            >
              <svg className="w-5 h-5 text-slate-400 hover:text-blue-600 dark:hover:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-6 tactical-scrollbar">
            {error && (
              <div className="mb-4 tactical-panel p-3 bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-400/30">
                <div className="flex items-start gap-2">
                  <svg className="w-4 h-4 text-red-600 dark:text-red-400 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <div className="text-[10px] text-red-600 dark:text-red-300">{error}</div>
                </div>
              </div>
            )}

            {showCustomPrompt ? (
              <div>
                <div className="mb-4">
                  <label className="text-xs font-bold text-blue-700 dark:text-amber-400 tracking-wider block mb-2">
                    DESCRIBE YOUR REPORT
                  </label>
                  <p className="text-[10px] text-slate-500 mb-3">
                    For example: Create a formal competitive review of the 2026 functional beverage market for a new wellness drink.
                  </p>
                  <textarea
                    value={customFormatPrompt}
                    onChange={(e) => setCustomFormatPrompt(e.target.value)}
                    disabled={isGenerating}
                    className="w-full h-64 bg-slate-100 dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded px-4 py-3 text-sm text-slate-700 dark:text-slate-300 leading-relaxed resize-none focus:outline-none focus:border-blue-500 dark:focus:border-amber-400 transition-colors tactical-scrollbar disabled:opacity-50 disabled:cursor-not-allowed"
                    placeholder="Describe the report you want to create..."
                  />
                </div>

                <button
                  onClick={handleGenerateReport}
                  disabled={isGenerating || !customFormatPrompt.trim()}
                  className="w-full px-6 py-4 bg-blue-600 dark:bg-amber-400 hover:bg-blue-700 dark:hover:bg-amber-500 disabled:bg-slate-300 dark:disabled:bg-slate-700 disabled:cursor-not-allowed text-white dark:text-slate-950 disabled:text-slate-400 dark:disabled:text-slate-500 font-bold text-sm tracking-wider transition-colors flex items-center justify-center gap-2"
                >
                  {isGenerating ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white dark:border-slate-950 border-t-transparent rounded-full animate-spin"></div>
                      GENERATING...
                    </>
                  ) : (
                    <>
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      GENERATE REPORT
                    </>
                  )}
                </button>
              </div>
            ) : (
              <PromptEditor
                initialPrompt={customPrompt}
                onPromptChange={setCustomPrompt}
                onGenerate={handleGenerateReport}
                isGenerating={isGenerating}
                disabled={isGenerating}
              />
            )}
          </div>
        </div>
        </div>

        {/* Report Viewer Modal */}
        {showReportViewer && (
          <ReportViewer
            reportContent={reportContent}
            reportTitle={reportTitle}
            onClose={handleCloseReportViewer}
          />
        )}
      </>
    );
  }

  // Show format selection view
  return (
    <>
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-200/90 dark:bg-slate-950/90 backdrop-blur-sm p-4">
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="border-b border-slate-200 dark:border-slate-800 p-4 flex items-center justify-between bg-slate-50 dark:bg-slate-900/50">
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5 text-blue-600 dark:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <h2 className="text-sm font-bold text-blue-700 dark:text-amber-400 tracking-wider">CREATE REPORT</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 hover:bg-slate-200 dark:hover:bg-slate-800 rounded transition-colors"
            title="Close"
          >
            <svg className="w-5 h-5 text-slate-400 hover:text-blue-600 dark:hover:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 tactical-scrollbar">
          {error && (
            <div className="mb-6 tactical-panel p-3 bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-400/30">
              <div className="flex items-start gap-2">
                <svg className="w-4 h-4 text-red-600 dark:text-red-400 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div className="text-[10px] text-red-600 dark:text-red-300">{error}</div>
              </div>
            </div>
          )}

          {/* Format Section */}
          <div className="mb-8">
            <h3 className="text-xs font-bold text-slate-600 dark:text-slate-400 tracking-wider mb-4">FORMAT</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {/* Create Your Own */}
              <button
                onClick={handleCreateYourOwn}
                className="tactical-panel p-4 text-left transition-all group hover:border-blue-400 dark:hover:border-amber-400/50 min-h-[140px] flex flex-col"
              >
                <div className="text-sm font-semibold text-slate-700 dark:text-slate-200 group-hover:text-blue-700 dark:group-hover:text-amber-400 mb-2">
                  Create Your Own
                </div>
                <div className="text-[10px] text-slate-500 leading-relaxed flex-1">
                  Craft reports your way by specifying structure, style, tone, and more
                </div>
              </button>

              {/* Hardcoded Formats */}
              {HARDCODED_FORMATS.map((format) => (
                <button
                  key={format.id}
                  onClick={() => handleFormatSelect(format)}
                  className="tactical-panel p-4 text-left transition-all group hover:border-blue-400 dark:hover:border-amber-400/50 min-h-[140px] flex flex-col relative"
                >
                  {/* Edit Icon */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleFormatEdit(format);
                    }}
                    className="absolute top-3 right-3 p-1 opacity-0 group-hover:opacity-100 hover:bg-slate-200 dark:hover:bg-slate-800 rounded transition-all"
                    title="Edit prompt"
                  >
                    <svg className="w-4 h-4 text-slate-400 hover:text-blue-600 dark:hover:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                    </svg>
                  </button>

                  <div className="text-sm font-semibold text-slate-700 dark:text-slate-200 group-hover:text-blue-700 dark:group-hover:text-amber-400 mb-2">
                    {format.name}
                  </div>
                  <div className="text-[10px] text-slate-500 leading-relaxed flex-1">
                    {format.description}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Suggested Format Section */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <svg className="w-4 h-4 text-tactical-green" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M11 3a1 1 0 10-2 0v1a1 1 0 102 0V3zM15.657 5.757a1 1 0 00-1.414-1.414l-.707.707a1 1 0 001.414 1.414l.707-.707zM18 10a1 1 0 01-1 1h-1a1 1 0 110-2h1a1 1 0 011 1zM5.05 6.464A1 1 0 106.464 5.05l-.707-.707a1 1 0 00-1.414 1.414l.707.707zM5 10a1 1 0 01-1 1H3a1 1 0 110-2h1a1 1 0 011 1zM8 16v-1h4v1a2 2 0 11-4 0zM12 14c.015-.34.208-.646.477-.859a4 4 0 10-4.954 0c.27.213.462.519.476.859h4.002z" />
                </svg>
                <h3 className="text-xs font-bold text-slate-600 dark:text-slate-400 tracking-wider">SUGGESTED FORMAT</h3>
              </div>
              {suggestionsStatus === 'loading' && (
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 border-2 border-tactical-green border-t-transparent rounded-full animate-spin"></div>
                  <span className="text-[9px] text-tactical-green">Analyzing...</span>
                </div>
              )}
            </div>

            {suggestionsStatus === 'loading' && (
              <div className="tactical-panel p-6 bg-slate-100 dark:bg-slate-900/30 text-center">
                <div className="text-[11px] text-slate-500">
                  AI is analyzing your documents to suggest custom formats...
                </div>
              </div>
            )}

            {suggestionsStatus === 'completed' && aiSuggestions.length > 0 && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {aiSuggestions.map((suggestion, index) => {
                  const format: ReportFormat = {
                    id: `ai-${index}`,
                    name: suggestion.name,
                    description: suggestion.description,
                    prompt: suggestion.prompt,
                  };
                  return (
                    <button
                      key={format.id}
                      onClick={() => handleFormatSelect(format)}
                      className="tactical-panel p-4 text-left transition-all group hover:border-tactical-green/50 min-h-[140px] flex flex-col relative border-tactical-green/30"
                    >
                      {/* AI Badge */}
                      <div className="absolute top-2 right-2 px-1.5 py-0.5 bg-tactical-green text-slate-950 text-[8px] font-bold tracking-wider">
                        AI
                      </div>

                      {/* Edit Icon */}
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleFormatEdit(format);
                        }}
                        className="absolute top-10 right-3 p-1 opacity-0 group-hover:opacity-100 hover:bg-slate-200 dark:hover:bg-slate-800 rounded transition-all"
                        title="Edit prompt"
                      >
                        <svg className="w-4 h-4 text-slate-400 hover:text-tactical-green" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                        </svg>
                      </button>

                      <div className="text-sm font-semibold text-slate-700 dark:text-slate-200 group-hover:text-tactical-green mb-2">
                        {format.name}
                      </div>
                      <div className="text-[10px] text-slate-500 leading-relaxed flex-1">
                        {format.description}
                      </div>
                    </button>
                  );
                })}
              </div>
            )}

            {suggestionsStatus === 'failed' && (
              <div className="tactical-panel p-4 bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-400/30">
                <div className="text-[10px] text-red-600 dark:text-red-400">
                  Failed to generate AI suggestions. Using preset formats only.
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
      </div>

      {/* Report Viewer Modal */}
      {showReportViewer && (
        <ReportViewer
          reportContent={reportContent}
          reportTitle={reportTitle}
          onClose={handleCloseReportViewer}
        />
      )}
    </>
  );
}
