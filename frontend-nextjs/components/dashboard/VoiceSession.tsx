"use client";

import { useEffect } from "react";
import {
  BarVisualizer,
  LiveKitRoom,
  RoomAudioRenderer,
  useVoiceAssistant,
} from "@livekit/components-react";
import "@livekit/components-styles";
import { X } from "lucide-react";

import { useVoiceStore } from "@/lib/stores/voiceStore";

/**
 * Full-screen voice call UX — mounted once at the dashboard root. When the voice
 * store has `connectionDetails`, renders a `<LiveKitRoom>` covering the viewport
 * with a bar visualizer, agent-state label, and end-call button.
 */
export default function VoiceSession() {
  const { state, details, error, stop } = useVoiceStore();
  const active = state === "connected" && details !== null;

  // Auto-clear after a brief error flash so the UI doesn't stay stuck.
  useEffect(() => {
    if (state !== "error") return;
    const t = setTimeout(() => stop(), 4000);
    return () => clearTimeout(t);
  }, [state, stop]);

  if (!active) {
    if (state === "error" && error) {
      return (
        <div className="fixed top-4 left-1/2 -translate-x-1/2 z-50 rounded-lg bg-red-600 px-4 py-2 text-sm text-white shadow-lg">
          {error}
        </div>
      );
    }
    if (state === "fetching-token") {
      return (
        <div className="fixed top-4 left-1/2 -translate-x-1/2 z-50 rounded-lg bg-zinc-900 px-4 py-2 text-sm text-white shadow-lg">
          Connecting…
        </div>
      );
    }
    return null;
  }

  return (
    <div className="fixed inset-0 z-50 bg-black/95 backdrop-blur-sm">
      <LiveKitRoom
        token={details.participantToken}
        serverUrl={details.serverUrl}
        connect={true}
        audio={true}
        video={false}
        onDisconnected={() => stop()}
        onError={(err) => {
          console.error("LiveKit error:", err);
          stop();
        }}
        className="h-full w-full"
      >
        <VoiceRoomContent />
        <RoomAudioRenderer />
      </LiveKitRoom>
    </div>
  );
}

function VoiceRoomContent() {
  const { state: agentState, audioTrack } = useVoiceAssistant();
  const stop = useVoiceStore((s) => s.stop);

  const label = agentStateLabel(agentState);

  return (
    <div className="relative flex h-full w-full flex-col items-center justify-center">
      {/* End call — top-right */}
      <button
        type="button"
        onClick={stop}
        aria-label="End call"
        className="absolute right-6 top-6 flex items-center gap-2 rounded-full bg-white/10 px-4 py-2 text-sm font-medium text-white hover:bg-white/20 transition-colors"
      >
        <X className="h-4 w-4" />
        End
      </button>

      {/* Visualizer */}
      <div className="flex h-56 w-full max-w-lg items-center justify-center">
        <BarVisualizer
          state={agentState}
          barCount={7}
          trackRef={audioTrack}
          className="h-full w-full [&>.lk-audio-bar]:bg-white"
        />
      </div>

      {/* State label */}
      <div className="mt-10 text-center">
        <div className="text-xs uppercase tracking-[0.2em] text-white/50">
          {label.eyebrow}
        </div>
        <div className="mt-2 text-lg font-medium text-white">{label.main}</div>
      </div>
    </div>
  );
}

function agentStateLabel(state: ReturnType<typeof useVoiceAssistant>["state"]) {
  switch (state) {
    case "connecting":
      return { eyebrow: "Voice call", main: "Connecting…" };
    case "initializing":
      return { eyebrow: "Voice call", main: "Starting up…" };
    case "listening":
      return { eyebrow: "Voice call", main: "Listening" };
    case "thinking":
      return { eyebrow: "Voice call", main: "Thinking…" };
    case "speaking":
      return { eyebrow: "Voice call", main: "Speaking" };
    case "disconnected":
      return { eyebrow: "Voice call", main: "Disconnected" };
    default:
      return { eyebrow: "Voice call", main: "Ready" };
  }
}
