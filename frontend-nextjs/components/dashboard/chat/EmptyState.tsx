"use client";

import React from "react";
import { motion } from "framer-motion";

const EmptyState = React.memo(function EmptyState() {
  return (
    <div className="h-full flex items-center justify-center px-6">
      <motion.div
        className="max-w-xl mx-auto text-center"
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: "easeOut" }}
      >
        <div className="inline-flex items-center justify-center w-10 h-10 rounded-lg bg-zinc-900 dark:bg-white mb-5">
          <svg
            className="w-5 h-5 text-white dark:text-zinc-900"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
            <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
          </svg>
        </div>

        <h2 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-100 mb-3">
          What would you like to know?
        </h2>
        <p className="text-sm text-zinc-500 dark:text-zinc-400 leading-relaxed mb-6">
          Select documents from your repository, or ask a question in general mode.
          Answers are grounded in your sources with inline citations.
        </p>

        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-zinc-200 dark:border-zinc-800 text-xs text-zinc-500 dark:text-zinc-400">
          <svg
            className="w-3.5 h-3.5"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="12" cy="12" r="10" />
            <path d="M12 16v-4M12 8h.01" />
          </svg>
          <span>Tip: select documents on the left for grounded answers</span>
        </div>
      </motion.div>
    </div>
  );
});

export default EmptyState;
