"use client";

import { useState, useEffect } from "react";

interface PromptEditorProps {
  initialPrompt: string;
  onPromptChange: (prompt: string) => void;
  onGenerate: () => void;
  isGenerating: boolean;
  disabled?: boolean;
}

export default function PromptEditor({
  initialPrompt,
  onPromptChange,
  onGenerate,
  isGenerating,
  disabled = false,
}: PromptEditorProps) {
  const [prompt, setPrompt] = useState(initialPrompt);
  const [isEdited, setIsEdited] = useState(false);

  // Update local state when initialPrompt changes
  useEffect(() => {
    setPrompt(initialPrompt);
    setIsEdited(false);
  }, [initialPrompt]);

  const handlePromptChange = (value: string) => {
    setPrompt(value);
    setIsEdited(value !== initialPrompt);
    onPromptChange(value);
  };

  const handleReset = () => {
    setPrompt(initialPrompt);
    setIsEdited(false);
    onPromptChange(initialPrompt);
  };

  return (
    <div className="tactical-panel p-4 bg-slate-200 dark:bg-slate-900/50">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4 text-blue-600 dark:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
          </svg>
          <h3 className="text-[10px] font-bold text-blue-700 dark:text-amber-400 tracking-wider">
            CUSTOMIZE PROMPT
          </h3>
        </div>

        {/* Reset Button */}
        {isEdited && (
          <button
            onClick={handleReset}
            className="px-2 py-1 text-[9px] font-bold tracking-wider text-slate-500 dark:text-slate-400 hover:text-blue-600 dark:hover:text-amber-400 transition-colors"
            title="Reset to original prompt"
          >
            RESET
          </button>
        )}
      </div>

      {/* Prompt Text Area */}
      <textarea
        value={prompt}
        onChange={(e) => handlePromptChange(e.target.value)}
        disabled={disabled || isGenerating}
        className="w-full h-48 bg-slate-100 dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded px-3 py-2 text-[11px] text-slate-700 dark:text-slate-300 leading-relaxed resize-none focus:outline-none focus:border-blue-500 dark:focus:border-amber-400 transition-colors tactical-scrollbar disabled:opacity-50 disabled:cursor-not-allowed"
        placeholder="Enter your custom prompt here..."
      />

      {/* Footer Info */}
      <div className="flex items-center justify-between mt-3">
        <div className="flex items-start gap-2">
          <svg className="w-3 h-3 text-tactical-green mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p className="text-[9px] text-slate-500 leading-relaxed">
            {isEdited
              ? 'Prompt has been modified. Click RESET to restore original.'
              : 'Edit the prompt to customize your report generation.'
            }
          </p>
        </div>

        {/* Character Count */}
        <div className="text-[9px] text-slate-600 font-mono">
          {prompt.length} chars
        </div>
      </div>

      {/* Generate Button */}
      <button
        onClick={onGenerate}
        disabled={disabled || isGenerating || !prompt.trim()}
        className="w-full mt-4 px-4 py-3 bg-blue-600 dark:bg-amber-400 hover:bg-blue-700 dark:hover:bg-amber-500 disabled:bg-slate-300 dark:disabled:bg-slate-700 disabled:cursor-not-allowed text-white dark:text-slate-950 disabled:text-slate-400 dark:disabled:text-slate-500 font-bold text-sm tracking-wider transition-colors flex items-center justify-center gap-2"
      >
        {isGenerating ? (
          <>
            <div className="w-4 h-4 border-2 border-white dark:border-slate-950 border-t-transparent rounded-full animate-spin"></div>
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
  );
}
