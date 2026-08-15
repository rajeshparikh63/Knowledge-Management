"use client";

import { useState, useEffect, useRef } from "react";
import { useChatStore } from "@/lib/stores/chatStore";
import { Z_INDEX } from "@/lib/constants/zIndex";

interface Model {
  id: string;
  name: string;
}

const AVAILABLE_MODELS: Model[] = [
  { id: "google/gemini-3-pro-preview", name: "Gemini 3 Pro" },
  { id: "anthropic/claude-sonnet-4.5", name: "Claude Sonnet 4.5" },
  { id: "google/gemini-3-flash-preview", name: "Gemini 3 Flash" },
  { id: "google/gemini-2.5-pro", name: "Gemini 2.5 Pro" },
  { id: "anthropic/claude-haiku-4.5", name: "Claude Haiku 4.5" },
  { id: "functiongemma:270m", name: "FunctionGemma 270M" },
];

export default function ModelSelector() {
  const { selectedModel, setSelectedModel } = useChatStore();
  const [models] = useState<Model[]>(AVAILABLE_MODELS);
  const [isOpen, setIsOpen] = useState(false);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const [dropdownPosition, setDropdownPosition] = useState({ top: 0, left: 0 });

  useEffect(() => {
    if (isOpen && buttonRef.current) {
      const rect = buttonRef.current.getBoundingClientRect();
      setDropdownPosition({
        top: rect.bottom + 6,
        left: rect.left,
      });
    }
  }, [isOpen]);

  const selectedModelData = models.find((m) => m.id === selectedModel);

  const handleModelSelect = (modelId: string) => {
    setSelectedModel(modelId);
    setIsOpen(false);
  };

  return (
    <div className="relative">
      <button
        ref={buttonRef}
        onClick={() => setIsOpen(!isOpen)}
        className="inline-flex items-center gap-1.5 px-2.5 h-8 rounded-lg border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 hover:bg-zinc-50 dark:hover:bg-zinc-900 transition-colors text-xs font-medium text-zinc-700 dark:text-zinc-300"
      >
        <svg className="w-3.5 h-3.5 text-zinc-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="3" y="3" width="18" height="18" rx="2" />
          <path d="M9 9h6v6H9z" />
        </svg>
        <span className="truncate max-w-[140px]">
          {selectedModelData?.name || "Select model"}
        </span>
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
        <>
          <div
            className="fixed inset-0"
            style={{ zIndex: Z_INDEX.DROPDOWN }}
            onClick={() => setIsOpen(false)}
          />
          <div
            className="fixed w-72 rounded-xl bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 shadow-xl max-h-96 overflow-y-auto tactical-scrollbar p-1"
            style={{
              zIndex: Z_INDEX.DROPDOWN + 1,
              top: `${dropdownPosition.top}px`,
              left: `${dropdownPosition.left}px`,
            }}
          >
            <div className="px-2.5 py-2 text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
              Select model
            </div>
            {models.map((model) => {
              const isSelected = selectedModel === model.id;
              return (
                <button
                  key={model.id}
                  onClick={() => handleModelSelect(model.id)}
                  className={`w-full text-left px-2.5 py-2 rounded-lg transition-colors ${
                    isSelected
                      ? "bg-zinc-100 dark:bg-zinc-900"
                      : "hover:bg-zinc-50 dark:hover:bg-zinc-900/60"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
                      {model.name}
                    </span>
                    {isSelected && (
                      <svg className="w-4 h-4 text-zinc-900 dark:text-zinc-100 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M5 13l4 4L19 7" />
                      </svg>
                    )}
                  </div>
                  <div className="text-[11px] text-zinc-500 mt-0.5 font-mono truncate">
                    {model.id}
                  </div>
                </button>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
