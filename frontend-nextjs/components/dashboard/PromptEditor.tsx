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
    <div className="tactical-panel p-4 bg-secondary dark:bg-card/50">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4 text-brand dark:text-brand" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
          </svg>
          <h3 className="text-[10px] font-bold text-brand dark:text-brand tracking-wider">
            CUSTOMIZE PROMPT
          </h3>
        </div>

        {/* Reset Button */}
        {isEdited && (
          <button
            onClick={handleReset}
            className="px-2 py-1 text-[9px] font-bold tracking-wider text-muted-foreground dark:text-muted-foreground hover:text-brand dark:hover:text-brand transition-colors"
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
        className="w-full h-48 bg-muted dark:bg-background border border-border dark:border-border rounded px-3 py-2 text-[11px] text-foreground dark:text-muted-foreground leading-relaxed resize-none focus:outline-none focus:border-brand dark:focus:border-brand transition-colors tactical-scrollbar disabled:opacity-50 disabled:cursor-not-allowed"
        placeholder="Enter your custom prompt here..."
      />

      {/* Footer Info */}
      <div className="flex items-center justify-between mt-3">
        <div className="flex items-start gap-2">
          <svg className="w-3 h-3 text-tactical-green mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p className="text-[9px] text-muted-foreground leading-relaxed">
            {isEdited
              ? 'Prompt has been modified. Click RESET to restore original.'
              : 'Edit the prompt to customize your report generation.'
            }
          </p>
        </div>

        {/* Character Count */}
        <div className="text-[9px] text-muted-foreground font-mono">
          {prompt.length} chars
        </div>
      </div>

      {/* Generate Button */}
      <button
        onClick={onGenerate}
        disabled={disabled || isGenerating || !prompt.trim()}
        className="w-full mt-4 px-4 py-3 bg-brand dark:bg-brand hover:bg-brand-hover dark:hover:bg-brand disabled:bg-secondary dark:disabled:bg-secondary disabled:cursor-not-allowed text-white dark:text-foreground disabled:text-muted-foreground dark:disabled:text-muted-foreground font-bold text-sm tracking-wider transition-colors flex items-center justify-center gap-2"
      >
        {isGenerating ? (
          <>
            <div className="w-4 h-4 border-2 border-white dark:border-border border-t-transparent rounded-full animate-spin"></div>
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
