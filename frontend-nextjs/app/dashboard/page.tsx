"use client";

import { useEffect, useState, useCallback, lazy, Suspense } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/lib/stores/authStore";
import { useDocuments, useKnowledgeBases } from "@/lib/hooks/useDocuments";
import Header from "@/components/dashboard/Header";
import Sidebar from "@/components/dashboard/Sidebar";
import ChatArea from "@/components/dashboard/ChatArea";
import ErrorBoundary from "@/components/ErrorBoundary";
import ChatSkeleton from "@/components/skeletons/ChatSkeleton";
import SidebarSkeleton from "@/components/skeletons/SidebarSkeleton";
import WorkflowSkeleton from "@/components/skeletons/WorkflowSkeleton";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";

// Lazy load heavy workflow panel
const WorkflowPanel = lazy(
  () => import("@/components/dashboard/WorkflowPanel")
);

// Lazy load voice session (only loaded when user starts a voice call)
const VoiceSession = lazy(
  () => import("@/components/dashboard/VoiceSession")
);

const MIN_PANEL_WIDTH = 280;
const MAX_PANEL_WIDTH = 600;
const DEFAULT_LEFT_WIDTH = 320;
const DEFAULT_RIGHT_WIDTH = 320;
const COLLAPSED_WIDTH = 56;
const MOBILE_BREAKPOINT = 768;
const TABLET_BREAKPOINT = 1024;

export default function DashboardPage() {
  const router = useRouter();
  const user = useAuthStore((s) => s.user);
  const isInitializing = useAuthStore((s) => s.isInitializing);

  const [leftWidth, setLeftWidth] = useState(DEFAULT_LEFT_WIDTH);
  const [rightWidth, setRightWidth] = useState(DEFAULT_RIGHT_WIDTH);
  const [isResizingLeft, setIsResizingLeft] = useState(false);
  const [isResizingRight, setIsResizingRight] = useState(false);
  const [isWorkflowCollapsed, setIsWorkflowCollapsed] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [isTablet, setIsTablet] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [workflowOpen, setWorkflowOpen] = useState(false);

  const actualRightWidth = isWorkflowCollapsed ? COLLAPSED_WIDTH : rightWidth;

  // Responsive breakpoint detection
  useEffect(() => {
    const checkBreakpoint = () => {
      const w = window.innerWidth;
      setIsMobile(w < MOBILE_BREAKPOINT);
      setIsTablet(w >= MOBILE_BREAKPOINT && w < TABLET_BREAKPOINT);
    };
    checkBreakpoint();
    window.addEventListener("resize", checkBreakpoint);
    return () => window.removeEventListener("resize", checkBreakpoint);
  }, []);

  // Resize handler
  useEffect(() => {
    if (!isResizingLeft && !isResizingRight) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (isResizingLeft) {
        const w = Math.min(MAX_PANEL_WIDTH, Math.max(MIN_PANEL_WIDTH, e.clientX));
        setLeftWidth(w);
      } else if (isResizingRight && !isWorkflowCollapsed) {
        const w = Math.min(MAX_PANEL_WIDTH, Math.max(MIN_PANEL_WIDTH, window.innerWidth - e.clientX));
        setRightWidth(w);
      }
    };

    const handleMouseUp = () => {
      setIsResizingLeft(false);
      setIsResizingRight(false);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [isResizingLeft, isResizingRight, isWorkflowCollapsed]);

  // Auth redirect
  useEffect(() => {
    if (!isInitializing && !user) {
      router.push("/auth/login");
    }
  }, [user, isInitializing, router]);

  // Google Drive OAuth redirect fallback: when the popup is blocked, the
  // backend callback redirects here with ?drive_connected=0&drive_message=…
  // instead of postMessage. Surface the message, then clean the URL.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    if (!params.has("drive_connected")) return;
    const ok = params.get("drive_connected") === "1";
    const message = params.get("drive_message");
    if (!ok && message) {
      // Defer so it doesn't fight with first paint
      setTimeout(() => alert(`Google Drive: ${message}`), 100);
    }
    // Strip the drive_* params from the URL without a reload
    ["drive_connected", "drive_email", "drive_message"].forEach((k) =>
      params.delete(k)
    );
    const qs = params.toString();
    window.history.replaceState(
      {},
      "",
      window.location.pathname + (qs ? `?${qs}` : "")
    );
  }, []);

  // React Query: cached, deduped, auto-refetches when stale
  useDocuments(user?.organization_id);
  useKnowledgeBases(user?.organization_id);

  const handleToggleWorkflow = useCallback((collapsed: boolean | ((prev: boolean) => boolean)) => {
    setIsWorkflowCollapsed(collapsed);
  }, []);

  if (isInitializing || !user) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-white dark:bg-[#0a0a0a]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-6 h-6 border-2 border-border border-t-border dark:border-border dark:border-t-border rounded-full animate-spin" />
          <span className="text-muted-foreground text-xs">Loading</span>
        </div>
      </div>
    );
  }

  // Mobile layout
  if (isMobile) {
    return (
      <div className="h-screen flex flex-col bg-white dark:bg-[#0a0a0a]">
        <Header />
        <div className="flex-1 flex flex-col overflow-hidden relative">
          {/* Mobile toolbar */}
          <div className="flex items-center justify-between px-3 py-2 border-b border-border dark:border-border bg-white/90 dark:bg-card/90">
            <Sheet open={sidebarOpen} onOpenChange={setSidebarOpen}>
              <SheetTrigger>
                <button className="tactical-btn p-2 text-xs">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h7" />
                  </svg>
                </button>
              </SheetTrigger>
              <SheetContent side="left" className="w-[85vw] max-w-[360px] p-0 bg-white dark:bg-[#0a0a0a]">
                <ErrorBoundary>
                  <Suspense fallback={<SidebarSkeleton />}>
                    <Sidebar />
                  </Suspense>
                </ErrorBoundary>
              </SheetContent>
            </Sheet>

            <span className="text-xs text-muted-foreground tracking-widest">SOLDIERIQ</span>

            <Sheet open={workflowOpen} onOpenChange={setWorkflowOpen}>
              <SheetTrigger>
                <button className="tactical-btn p-2 text-xs">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                  </svg>
                </button>
              </SheetTrigger>
              <SheetContent side="right" className="w-[85vw] max-w-[360px] p-0 bg-white dark:bg-[#0a0a0a]">
                <ErrorBoundary>
                  <Suspense fallback={<WorkflowSkeleton />}>
                    <WorkflowPanel isCollapsed={false} onToggleCollapse={() => setWorkflowOpen(false)} />
                  </Suspense>
                </ErrorBoundary>
              </SheetContent>
            </Sheet>
          </div>

          {/* Chat takes full width on mobile */}
          <ErrorBoundary>
            <Suspense fallback={<ChatSkeleton />}>
              <ChatArea />
            </Suspense>
          </ErrorBoundary>
        </div>
      </div>
    );
  }

  // Desktop / Tablet layout
  return (
    <div className="h-screen flex flex-col bg-white dark:bg-[#0a0a0a]">
      <Header />
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar */}
        {!isTablet && (
          <div className="relative flex flex-shrink-0" style={{ width: leftWidth }}>
            <ErrorBoundary>
              <Suspense fallback={<SidebarSkeleton />}>
                <Sidebar />
              </Suspense>
            </ErrorBoundary>
            <div
              className="absolute right-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-secondary dark:hover:bg-secondary transition-colors z-10"
              onMouseDown={() => setIsResizingLeft(true)}
            >
              <div className="absolute inset-y-0 -left-1 -right-1" />
            </div>
          </div>
        )}

        {/* Tablet: sidebar as sheet */}
        {isTablet && (
          <Sheet open={sidebarOpen} onOpenChange={setSidebarOpen}>
            <SheetTrigger>
              <button className="absolute left-2 top-16 z-20 tactical-btn p-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h7" />
                </svg>
              </button>
            </SheetTrigger>
            <SheetContent side="left" className="w-[320px] p-0 bg-white dark:bg-[#0a0a0a]">
              <ErrorBoundary>
                <Suspense fallback={<SidebarSkeleton />}>
                  <Sidebar />
                </Suspense>
              </ErrorBoundary>
            </SheetContent>
          </Sheet>
        )}

        {/* Chat Area */}
        <ErrorBoundary>
          <Suspense fallback={<ChatSkeleton />}>
            <ChatArea />
          </Suspense>
        </ErrorBoundary>

        {/* Right Workflow Panel */}
        <div className="relative flex flex-shrink-0" style={{ width: actualRightWidth }}>
          {!isWorkflowCollapsed && (
            <div
              className="absolute left-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-secondary dark:hover:bg-secondary transition-colors z-10"
              onMouseDown={() => setIsResizingRight(true)}
            >
              <div className="absolute inset-y-0 -left-1 -right-1" />
            </div>
          )}
          <ErrorBoundary>
            <Suspense fallback={<WorkflowSkeleton />}>
              <WorkflowPanel
                isCollapsed={isWorkflowCollapsed}
                onToggleCollapse={handleToggleWorkflow}
              />
            </Suspense>
          </ErrorBoundary>
        </div>
      </div>

      <Suspense fallback={null}>
        <VoiceSession />
      </Suspense>
    </div>
  );
}
