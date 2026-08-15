"use client";

import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

export interface Mission {
  id: string;
  name: string;
  createdAt: string;
}

interface MissionSelectorProps {
  missions: Mission[];
  currentMissionId: string | null;
  onSelect: (mission: Mission) => void;
  onCreate: () => void;
}

const MissionSelector = React.memo(function MissionSelector({
  missions,
  currentMissionId,
  onSelect,
  onCreate,
}: MissionSelectorProps) {
  const [showDropdown, setShowDropdown] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setShowDropdown(false);
      }
    };

    if (showDropdown) {
      document.addEventListener("mousedown", handleClickOutside);
      return () =>
        document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [showDropdown]);

  const currentMissionName = currentMissionId
    ? missions.find((m) => m.id === currentMissionId)?.name || "Select mission"
    : "Select mission";

  const handleSelect = (mission: Mission) => {
    onSelect(mission);
    setShowDropdown(false);
  };

  return (
    <div className="border-b border-border dark:border-border bg-white dark:bg-[#0a0a0a] px-4 py-2.5">
      <div className="flex items-center gap-2 max-w-4xl mx-auto">
        <span className="text-xs font-medium text-muted-foreground dark:text-muted-foreground">
          Mission
        </span>
        <div className="relative flex-1 max-w-sm" ref={dropdownRef}>
          <button
            onClick={() => setShowDropdown(!showDropdown)}
            className="w-full flex items-center justify-between gap-2 px-3 py-1.5 rounded-lg border border-border dark:border-border bg-white dark:bg-background text-sm hover:border-border dark:hover:border-border transition-colors"
          >
            <span className="truncate text-foreground dark:text-foreground">
              {currentMissionName}
            </span>
            <svg
              className={`w-3.5 h-3.5 text-muted-foreground transition-transform flex-shrink-0 ${
                showDropdown ? "rotate-180" : ""
              }`}
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              viewBox="0 0 24 24"
            >
              <path d="M6 9l6 6 6-6" />
            </svg>
          </button>

          <AnimatePresence>
            {showDropdown && (
              <motion.div
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: 0.15, ease: "easeOut" }}
                className="absolute top-full left-0 right-0 mt-1.5 rounded-xl bg-white dark:bg-background border border-border dark:border-border shadow-xl z-50 overflow-hidden"
              >
                {missions.length === 0 ? (
                  <div className="px-4 py-3 text-xs text-muted-foreground dark:text-muted-foreground text-center">
                    No missions yet. Create one to get started.
                  </div>
                ) : (
                  <div className="max-h-64 overflow-y-auto tactical-scrollbar p-1">
                    {missions.map((mission) => (
                      <button
                        key={mission.id}
                        onClick={() => handleSelect(mission)}
                        className={`w-full px-2.5 py-2 text-left rounded-lg hover:bg-secondary dark:hover:bg-accent transition-colors ${
                          mission.id === currentMissionId
                            ? "bg-secondary dark:bg-card"
                            : ""
                        }`}
                      >
                        <div className="text-sm font-medium text-foreground dark:text-foreground truncate">
                          {mission.name}
                        </div>
                        <div className="text-[11px] text-muted-foreground dark:text-muted-foreground mt-0.5">
                          {new Date(mission.createdAt).toLocaleString()}
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <button
          onClick={onCreate}
          className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-brand text-brand-foreground hover:bg-brand-hover shadow-accent transition-all"
          title="New mission"
          aria-label="Create new mission"
        >
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            viewBox="0 0 24 24"
          >
            <path d="M12 5v14M5 12h14" />
          </svg>
        </button>
      </div>
    </div>
  );
});

export default MissionSelector;
