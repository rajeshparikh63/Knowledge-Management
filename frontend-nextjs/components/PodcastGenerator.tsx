"use client";

import React, { useState, useEffect } from 'react';
import { podcastApi, PodcastEpisode } from '@/lib/api/podcast';

interface PodcastGeneratorProps {
  selectedDocumentIds: string[];
  onClose?: () => void;
}

export default function PodcastGenerator({ selectedDocumentIds, onClose }: PodcastGeneratorProps) {
  const [isGenerating, setIsGenerating] = useState(false);
  const [episode, setEpisode] = useState<PodcastEpisode | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  const handleGenerate = async () => {
    if (selectedDocumentIds.length === 0) {
      setError('Please select at least one document');
      return;
    }

    setIsGenerating(true);
    setError(null);

    try {
      // Start generation
      const response = await podcastApi.generatePodcast(selectedDocumentIds);
      
      // Poll for status
      const episodeId = response.episode_id;
      pollStatus(episodeId);
    } catch (err: any) {
      setError(err.message || 'Failed to start audio overview generation');
      setIsGenerating(false);
    }
  };

  const pollStatus = async (episodeId: string) => {
    const maxAttempts = 60; // 60 attempts = 10 minutes with 10s intervals
    let attempts = 0;

    const interval = setInterval(async () => {
      attempts++;
      
      if (attempts > maxAttempts) {
        clearInterval(interval);
        setError('Audio overview generation timed out');
        setIsGenerating(false);
        return;
      }

      try {
        const statusData = await podcastApi.getPodcastStatus(episodeId);
        setEpisode(statusData);

        if (statusData.status === 'completed') {
          clearInterval(interval);
          setIsGenerating(false);
          
          // Fetch presigned URL for audio
          if (statusData.audio_file_key) {
            fetchAudioUrl(statusData.audio_file_key);
          }
        } else if (statusData.status === 'failed') {
          clearInterval(interval);
          setError(statusData.error_message || 'Audio overview generation failed');
          setIsGenerating(false);
        }
      } catch (err: any) {
        // Polling error - will retry
        if (attempts > 3) {
          clearInterval(interval);
          setError('Failed to check audio overview status');
          setIsGenerating(false);
        }
      }
    }, 10000); // Poll every 10 seconds
  };

  const fetchAudioUrl = async (fileKey: string) => {
    try {
      const response = await fetch(`/api/files/presigned-url?file_key=${encodeURIComponent(fileKey)}`);
      const data = await response.json();
      
      if (data.url) {
        setAudioUrl(data.url);
      } else {
        setError('Failed to generate audio URL');
      }
    } catch (err) {
      // Audio URL fetch failed
      setError('Failed to load audio file');
    }
  };

  const getStatusIcon = () => {
    if (!episode) return null;

    switch (episode.status) {
      case 'processing':
        return (
          <div className="w-5 h-5 border-2 border-brand dark:border-brand/50 border-t-transparent rounded-full animate-spin" />
        );
      case 'script_generated':
        return (
          <div className="w-5 h-5 border-2 border-brand dark:border-brand/50 border-t-transparent rounded-full animate-spin" />
        );
      case 'completed':
        return (
          <svg className="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        );
      case 'failed':
        return (
          <svg className="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        );
      default:
        return null;
    }
  };

  const getStatusText = () => {
    if (!episode) return '';

    switch (episode.status) {
      case 'processing':
        return 'Generating script...';
      case 'script_generated':
        return 'Generating audio...';
      case 'completed':
        return 'Audio overview ready!';
      case 'failed':
        return 'Generation failed';
      default:
        return episode.status;
    }
  };

  return (
    <div className="w-full max-w-2xl bg-white dark:bg-card rounded-lg shadow-xl border border-border dark:border-border">
      {/* Header */}
      <div className="border-b border-border dark:border-border p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <svg className="w-6 h-6 text-brand dark:text-brand" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
            </svg>
            <h2 className="text-xl font-bold text-foreground dark:text-foreground">Audio Overview Generator</h2>
          </div>
          {onClose && (
            <button
              onClick={onClose}
              className="p-2 hover:bg-muted dark:hover:bg-muted rounded transition-colors"
            >
              <svg className="w-5 h-5 text-muted-foreground hover:text-muted-foreground dark:hover:text-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
        <p className="text-sm text-muted-foreground dark:text-muted-foreground mt-2">
          Generate an AI audio overview from {selectedDocumentIds.length} selected document(s)
        </p>
      </div>

      {/* Content */}
      <div className="p-6 space-y-4">
        {error && (
          <div className="p-4 bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 rounded-md">
            <p className="text-sm text-red-800 dark:text-red-200">{error}</p>
          </div>
        )}

        {(!episode || episode.status === 'processing' || episode.status === 'script_generated' || isGenerating) && (
          <button 
            onClick={handleGenerate} 
            disabled={selectedDocumentIds.length === 0 || isGenerating || Boolean(episode && episode.status !== 'failed')}
            className="w-full px-4 py-3 bg-brand hover:bg-brand-hover dark:bg-brand dark:hover:bg-brand disabled:bg-secondary dark:disabled:bg-secondary text-white dark:text-foreground disabled:text-muted-foreground dark:disabled:text-muted-foreground rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
          >
            {isGenerating || (episode && (episode.status === 'processing' || episode.status === 'script_generated')) ? (
              <>
                <div className="w-5 h-5 border-2 border-white/20 dark:border-border/20 border-t-white dark:border-t-border rounded-full animate-spin" />
                Generating Audio Overview...
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
                Generate Audio Overview
              </>
            )}
          </button>
        )}

        {episode && (
          <div className="space-y-4">

            {episode.summary && (
              <div className="p-4 bg-muted dark:bg-muted rounded-md">
                <p className="text-sm font-medium text-foreground dark:text-foreground mb-2">Summary</p>
                <p className="text-sm text-muted-foreground dark:text-muted-foreground">{episode.summary}</p>
              </div>
            )}

            {audioUrl && episode.status === 'completed' && (
              <div className="space-y-3">
                <audio 
                  controls 
                  className="w-full"
                  src={audioUrl}
                >
                  Your browser does not support the audio element.
                </audio>
                <button
                  className="w-full px-4 py-2 border border-border dark:border-border hover:bg-muted dark:hover:bg-muted text-foreground dark:text-muted-foreground rounded-lg transition-colors flex items-center justify-center gap-2"
                  onClick={() => window.open(audioUrl, '_blank')}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Download Audio Overview
                </button>
              </div>
            )}
          </div>
        )}

        {onClose && (
          <button 
            onClick={onClose}
            className="w-full px-4 py-2 border border-border dark:border-border hover:bg-muted dark:hover:bg-muted text-foreground dark:text-muted-foreground rounded-lg transition-colors"
          >
            Close
          </button>
        )}
      </div>
    </div>
  );
}
