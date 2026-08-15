"use client";

import { useState, useEffect } from "react";
import { FlashcardData } from "@/lib/api/flashcards";
import { useThemeStore } from "@/lib/stores/themeStore";

interface FlashcardViewerProps {
  flashcardData: FlashcardData;
  onClose: () => void;
}

export default function FlashcardViewer({ flashcardData, onClose }: FlashcardViewerProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isFlipped, setIsFlipped] = useState(false);
  const theme = useThemeStore((state) => state.theme);
  const isDark = theme === 'dark';

  const { title, cards } = flashcardData;
  const totalCards = cards.length;
  const currentCard = cards[currentIndex];
  const sourceCount = flashcardData.document_count || 0;

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case ' ':
        case 'Spacebar':
          e.preventDefault();
          setIsFlipped(!isFlipped);
          break;
        case 'ArrowLeft':
          e.preventDefault();
          handlePrevious();
          break;
        case 'ArrowRight':
          e.preventDefault();
          handleNext();
          break;
        case 'Escape':
          e.preventDefault();
          onClose();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isFlipped, currentIndex]);

  const handleNext = () => {
    if (currentIndex < totalCards - 1) {
      setCurrentIndex(currentIndex + 1);
      setIsFlipped(false);
    }
  };

  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
      setIsFlipped(false);
    }
  };

  const handleFlip = () => {
    setIsFlipped(!isFlipped);
  };

  return (
    <div className="fixed inset-0 z-50 bg-slate-50 dark:bg-slate-950 flex flex-col">
      {/* Header */}
      <div className="border-b border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/50 backdrop-blur-sm p-6">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-2xl font-bold text-slate-800 dark:text-slate-200 mb-2">{title}</h1>
              <p className="text-sm text-slate-600 dark:text-slate-500">
                Based on {sourceCount} {sourceCount === 1 ? 'source' : 'sources'}
              </p>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-slate-100 dark:hover:bg-slate-800 rounded transition-colors"
              title="Close (Esc)"
            >
              <svg className="w-6 h-6 text-slate-400 hover:text-blue-600 dark:hover:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col items-center justify-center p-8">
        {/* Instructions */}
        <div className="text-center mb-8">
          <p className="text-sm text-slate-600 dark:text-slate-500">
            Press <kbd className="px-2 py-1 bg-slate-200 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded text-xs">Space</kbd> to flip,
            <kbd className="px-2 py-1 bg-slate-200 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded text-xs ml-2">← / →</kbd> to navigate
          </p>
        </div>

        {/* Flashcard */}
        <div className="w-full max-w-4xl perspective-container mb-8">
          <div
            className={`flashcard-container ${isFlipped ? 'flipped' : ''}`}
            onClick={handleFlip}
          >
            {/* Front */}
            <div className="flashcard flashcard-front">
              <div className="flex flex-col h-full">
                <div className="flex-1 flex items-center justify-center p-12">
                  <p className="text-3xl text-center text-slate-800 dark:text-slate-200 leading-relaxed">
                    {currentCard.front}
                  </p>
                </div>
                <div className="text-center pb-6">
                  <button className="text-sm text-slate-600 dark:text-slate-500 hover:text-blue-600 dark:hover:text-amber-400 transition-colors">
                    See answer
                  </button>
                </div>
              </div>
            </div>

            {/* Back */}
            <div className="flashcard flashcard-back">
              <div className="flex flex-col h-full">
                <div className="flex-1 flex items-center justify-center p-12">
                  <p className="text-2xl text-center text-slate-700 dark:text-slate-300 leading-relaxed">
                    {currentCard.back}
                  </p>
                </div>
                <div className="text-center pb-6">
                  {/* Optional: Explain button for future enhancement */}
                  {/* <button className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm rounded transition-colors">
                    <svg className="w-4 h-4 inline mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Explain
                  </button> */}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <div className="flex items-center gap-8">
          {/* Previous Button */}
          <button
            onClick={handlePrevious}
            disabled={currentIndex === 0}
            className="w-14 h-14 rounded-full bg-slate-200 dark:bg-slate-800 hover:bg-slate-300 dark:hover:bg-slate-700 disabled:bg-slate-100 dark:disabled:bg-slate-900 disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center transition-colors group"
            title="Previous (←)"
          >
            <svg className="w-6 h-6 text-slate-600 dark:text-slate-400 group-hover:text-blue-600 dark:group-hover:text-amber-400 group-disabled:text-slate-400 dark:group-disabled:text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>

          {/* Progress */}
          <div className="flex flex-col items-center gap-2">
            <div className="text-lg font-semibold text-slate-600 dark:text-slate-400">
              {currentIndex + 1} / {totalCards} cards
            </div>
            <div className="w-64 h-1 bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-600 dark:bg-amber-400 transition-all duration-300"
                style={{ width: `${((currentIndex + 1) / totalCards) * 100}%` }}
              />
            </div>
          </div>

          {/* Next Button */}
          <button
            onClick={handleNext}
            disabled={currentIndex === totalCards - 1}
            className="w-14 h-14 rounded-full bg-slate-200 dark:bg-slate-800 hover:bg-slate-300 dark:hover:bg-slate-700 disabled:bg-slate-100 dark:disabled:bg-slate-900 disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center transition-colors group"
            title="Next (→)"
          >
            <svg className="w-6 h-6 text-slate-600 dark:text-slate-400 group-hover:text-blue-600 dark:group-hover:text-amber-400 group-disabled:text-slate-400 dark:group-disabled:text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        </div>

        {/* Optional: Feedback Buttons */}
        {/* <div className="mt-12 flex gap-4">
          <button className="px-6 py-3 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded transition-colors flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
            </svg>
            Good content
          </button>
          <button className="px-6 py-3 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded transition-colors flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018a2 2 0 01.485.06l3.76.94m-7 10v5a2 2 0 002 2h.096c.5 0 .905-.405.905-.904 0-.715.211-1.413.608-2.008L17 13V4m-7 10h2m5-10h2a2 2 0 012 2v6a2 2 0 01-2 2h-2.5" />
            </svg>
            Bad content
          </button>
        </div> */}
      </div>

      {/* CSS for flip animation */}
      <style jsx>{`
        .perspective-container {
          perspective: 1000px;
        }

        .flashcard-container {
          position: relative;
          width: 100%;
          height: 500px;
          cursor: pointer;
          transform-style: preserve-3d;
          transition: transform 0.6s;
        }

        .flashcard-container.flipped {
          transform: rotateY(180deg);
        }

        .flashcard {
          position: absolute;
          width: 100%;
          height: 100%;
          backface-visibility: hidden;
          border-radius: 12px;
          box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .flashcard-front {
          background: ${isDark
            ? 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)'
            : 'linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)'};
          border: 1px solid ${isDark ? '#334155' : '#cbd5e1'};
        }

        .flashcard-back {
          background: ${isDark
            ? 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)'
            : 'linear-gradient(135deg, #f1f5f9 0%, #f8fafc 100%)'};
          border: 1px solid ${isDark ? '#334155' : '#cbd5e1'};
          transform: rotateY(180deg);
        }

        kbd {
          font-family: ui-monospace, monospace;
        }
      `}</style>
    </div>
  );
}
