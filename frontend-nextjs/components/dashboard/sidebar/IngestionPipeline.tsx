"use client";

/**
 * IngestionPipeline — compact 4-step status indicator shown on processing docs.
 *
 * Backend stages from ingestion_service + graphrag_client map onto 4 visible
 * steps. The current step pulses, completed steps show a check, pending steps
 * stay neutral. Below the dots we surface the human description and, when
 * available, the chunk-progress count (e.g. "17/64").
 */

import React from "react";
import { Document } from "@/types";

interface IngestionPipelineProps {
  document: Document;
}

type StepState = "done" | "active" | "pending";

interface Step {
  key: string;
  label: string;
}

const STEPS: Step[] = [
  { key: "upload", label: "Upload" },
  { key: "extract", label: "Extract" },
  { key: "embed", label: "Entities" },
  { key: "graph", label: "Graph" },
];

// Map backend stage → which step it belongs to (zero-indexed).
// Anything not listed defaults to step 2 (embed) so we never look "stuck"
// at the very start.
const STAGE_TO_STEP: Record<string, number> = {
  // Step 0 — Upload (incl. the just-created "initializing" state)
  initializing: 0,
  uploading: 0,
  uploading_extracting: 0,

  // Step 1 — Content extraction
  content_extracted: 1,

  // Step 2 — Chunk / embed / extract entities
  cognifying: 2,
  schema_detect: 2,
  chunking: 2,
  extracting: 2,

  // Step 3 — Resolve + write graph
  resolving: 3,
  writing: 3,

  // Terminal
  completed: 4,
  failed: -1,
};

function stepStateFor(currentStepIndex: number, idx: number): StepState {
  if (currentStepIndex >= STEPS.length) return "done";
  if (idx < currentStepIndex) return "done";
  if (idx === currentStepIndex) return "active";
  return "pending";
}

const IngestionPipeline = React.memo(function IngestionPipeline({
  document: doc,
}: IngestionPipelineProps) {
  const stage = doc.processing_stage || "uploading";
  const failed = doc.status === "failed" || stage === "failed";
  const currentStepIndex = failed
    ? -1
    : STAGE_TO_STEP[stage] ?? 2; // unknown stage → assume mid-pipeline

  const progress = doc.processing_progress;
  const showCount =
    progress &&
    typeof progress.current === "number" &&
    typeof progress.total === "number" &&
    progress.total > 0;

  return (
    <div className="mt-1">
      {/* Step dots + connectors */}
      <div className="flex items-center gap-1">
        {STEPS.map((step, idx) => {
          const state = failed
            ? idx < (STAGE_TO_STEP[stage] ?? 99)
              ? "done"
              : "pending"
            : stepStateFor(currentStepIndex, idx);

          const isLast = idx === STEPS.length - 1;
          return (
            <React.Fragment key={step.key}>
              <div
                className="flex flex-col items-center"
                title={step.label}
              >
                <div
                  className={[
                    "w-2.5 h-2.5 rounded-full flex items-center justify-center text-[8px] leading-none transition-colors",
                    failed && state === "pending"
                      ? "bg-secondary dark:bg-secondary"
                      : failed && state === "done"
                        ? "bg-red-500"
                        : state === "done"
                          ? "bg-emerald-500"
                          : state === "active"
                            ? "bg-card dark:bg-secondary animate-pulse"
                            : "bg-secondary dark:bg-secondary",
                  ].join(" ")}
                >
                  {state === "done" && !failed && (
                    <svg
                      className="w-1.5 h-1.5 text-white"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth={4}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                  )}
                </div>
                <div
                  className={[
                    "text-[8px] mt-0.5 leading-none tracking-tight",
                    state === "active"
                      ? "text-foreground dark:text-foreground font-medium"
                      : "text-muted-foreground dark:text-muted-foreground",
                  ].join(" ")}
                >
                  {step.label}
                </div>
              </div>
              {!isLast && (
                <div
                  className={[
                    "flex-1 h-px mb-3 transition-colors",
                    failed
                      ? "bg-secondary dark:bg-secondary"
                      : idx < currentStepIndex
                        ? "bg-emerald-500/60"
                        : "bg-secondary dark:bg-secondary",
                  ].join(" ")}
                />
              )}
            </React.Fragment>
          );
        })}
      </div>

      {/* Stage description + optional progress count */}
      {doc.processing_stage_description && (
        <div
          className={`text-[10px] mt-1.5 truncate ${
            failed
              ? "text-red-500 dark:text-red-400"
              : "text-muted-foreground dark:text-muted-foreground"
          }`}
        >
          {doc.processing_stage_description}
        </div>
      )}
      {showCount && !failed && (
        <div className="text-[10px] text-muted-foreground dark:text-muted-foreground mt-0.5">
          {progress!.current} / {progress!.total} chunks
        </div>
      )}
    </div>
  );
});

export default IngestionPipeline;
