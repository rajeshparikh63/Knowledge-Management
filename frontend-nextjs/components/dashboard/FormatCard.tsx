"use client";

import { ReportFormat } from "@/lib/constants/reportFormats";

interface FormatCardProps {
  format: ReportFormat;
  isSelected: boolean;
  onSelect: (format: ReportFormat) => void;
  isAISuggested?: boolean;
}

export default function FormatCard({ format, isSelected, onSelect, isAISuggested = false }: FormatCardProps) {
  return (
    <button
      onClick={() => onSelect(format)}
      className={`tactical-panel p-4 text-left transition-all group relative ${
        isSelected
          ? 'border-amber-400 bg-amber-900/20'
          : 'hover:border-amber-400/50'
      }`}
    >
      {/* AI Suggested Badge */}
      {isAISuggested && (
        <div className="absolute -top-1 -right-1 px-2 py-0.5 bg-tactical-green border border-tactical-green/50 text-[8px] font-bold text-slate-950 tracking-wider flex items-center gap-1">
          <svg className="w-2.5 h-2.5" fill="currentColor" viewBox="0 0 20 20">
            <path d="M11 3a1 1 0 10-2 0v1a1 1 0 102 0V3zM15.657 5.757a1 1 0 00-1.414-1.414l-.707.707a1 1 0 001.414 1.414l.707-.707zM18 10a1 1 0 01-1 1h-1a1 1 0 110-2h1a1 1 0 011 1zM5.05 6.464A1 1 0 106.464 5.05l-.707-.707a1 1 0 00-1.414 1.414l.707.707zM5 10a1 1 0 01-1 1H3a1 1 0 110-2h1a1 1 0 011 1zM8 16v-1h4v1a2 2 0 11-4 0zM12 14c.015-.34.208-.646.477-.859a4 4 0 10-4.954 0c.27.213.462.519.476.859h4.002z" />
          </svg>
          AI
        </div>
      )}

      {/* Selection Indicator */}
      <div className={`absolute top-4 right-4 w-4 h-4 rounded-full border-2 transition-all ${
        isSelected
          ? 'border-amber-400 bg-amber-400'
          : 'border-slate-600 bg-transparent group-hover:border-amber-400/50'
      }`}>
        {isSelected && (
          <svg className="w-full h-full text-slate-950" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
          </svg>
        )}
      </div>

      {/* Content */}
      <div className="pr-6">
        {/* Format Name */}
        <div className={`text-sm font-semibold tracking-wide mb-2 transition-colors ${
          isSelected
            ? 'text-amber-400'
            : 'text-slate-200 group-hover:text-amber-400'
        }`}>
          {format.name}
        </div>

        {/* Format Description */}
        <div className="text-[11px] text-slate-400 leading-relaxed">
          {format.description}
        </div>
      </div>
    </button>
  );
}
