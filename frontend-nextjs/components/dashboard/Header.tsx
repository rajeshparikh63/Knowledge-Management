"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/lib/stores/authStore";
import { useChatStore } from "@/lib/stores/chatStore";
import ModelSelector from "./ModelSelector";
import ThemeToggle from "../ThemeToggle";
import SessionDropdown from "./SessionDropdown";
import TicketCreationModal from "../TicketCreationModal";
import TAKSettingsModal from "./TAKSettingsModal";
import { Z_INDEX } from "@/lib/constants/zIndex";

export default function Header() {
  const router = useRouter();
  const { user, logout } = useAuthStore();
  const { messages } = useChatStore();
  const [showMenu, setShowMenu] = useState(false);
  const [showTicketModal, setShowTicketModal] = useState(false);
  const [showTAKSettings, setShowTAKSettings] = useState(false);
  const [comingSoonDialog, setComingSoonDialog] = useState<{ show: boolean; feature: string }>({
    show: false,
    feature: "",
  });

  const handleLogout = async () => {
    await logout();
    router.push("/auth/login");
  };

  const showComingSoonDialog = (feature: string) => {
    setComingSoonDialog({ show: true, feature });
  };

  const closeComingSoonDialog = () => {
    setComingSoonDialog({ show: false, feature: "" });
  };

  const openCalendly = () => {
    const calendlyUrl = "https://calendly.com/soldieriq-io/30min";
    if (typeof window !== "undefined" && window.Calendly) {
      window.Calendly.initPopupWidget({ url: calendlyUrl });
    }
  };

  const hasUserMessages = messages.some((msg) => msg.role === "user");

  const navBtnClass =
    "inline-flex items-center gap-1.5 px-3 h-8 rounded-lg text-xs font-medium text-zinc-700 dark:text-zinc-300 hover:text-zinc-900 dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors";

  return (
    <header className="h-14 bg-white dark:bg-[#0a0a0a] border-b border-zinc-200 dark:border-zinc-800 flex items-center justify-between px-4 relative z-30">
      <div className="flex items-center gap-4 min-w-0">
        {/* Logo */}
        <div className="flex items-center gap-2.5 flex-shrink-0">
          <div className="w-7 h-7 rounded-md bg-zinc-900 dark:bg-white flex items-center justify-center">
            <svg
              className="w-4 h-4 text-white dark:text-zinc-900"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
              <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
            </svg>
          </div>
          <span className="text-sm font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
            SoldierIQ
          </span>
        </div>

        <div className="h-5 w-px bg-zinc-200 dark:bg-zinc-800" />

        {/* Model Selector */}
        <ModelSelector />

        {/* Session Dropdown */}
        <SessionDropdown hasUserMessages={hasUserMessages} />
      </div>

      <div className="flex items-center gap-1">
        <button onClick={() => setShowTicketModal(true)} className={navBtnClass}>
          <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M15 5v2m0 4v2m0 4v2M5 5a2 2 0 00-2 2v3a2 2 0 110 4v3a2 2 0 002 2h14a2 2 0 002-2v-3a2 2 0 110-4V7a2 2 0 00-2-2H5z" />
          </svg>
          <span className="hidden md:inline">New ticket</span>
        </button>

        <button
          onClick={() => showComingSoonDialog("Automated Agent Chat")}
          className={navBtnClass}
        >
          <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
          </svg>
          <span className="hidden md:inline">Agent chat</span>
        </button>

        <button onClick={openCalendly} className={navBtnClass}>
          <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="4" width="18" height="18" rx="2" />
            <path d="M16 2v4M8 2v4M3 10h18" />
          </svg>
          <span className="hidden md:inline">Book call</span>
        </button>

        <div className="mx-1 h-5 w-px bg-zinc-200 dark:bg-zinc-800" />

        <ThemeToggle />

        <div className="relative">
          <button
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setShowMenu((prev) => !prev);
            }}
            className="flex items-center gap-2 pl-1.5 pr-2.5 h-9 rounded-lg hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors"
          >
            <div className="w-6 h-6 rounded-full bg-zinc-900 dark:bg-zinc-100 text-white dark:text-zinc-900 text-[11px] font-semibold flex items-center justify-center">
              {user?.email?.charAt(0).toUpperCase()}
            </div>
            <span className="hidden sm:inline text-xs font-medium text-zinc-700 dark:text-zinc-300 max-w-[140px] truncate">
              {user?.email}
            </span>
            <svg
              className={`w-3.5 h-3.5 text-zinc-500 transition-transform ${showMenu ? "rotate-180" : ""}`}
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

          {showMenu && (
            <>
              <div
                className="fixed inset-0"
                style={{ zIndex: Z_INDEX.DROPDOWN }}
                onClick={(e) => {
                  e.stopPropagation();
                  setShowMenu(false);
                }}
              />
              <div
                className="absolute right-0 top-full mt-1.5 w-56 rounded-xl bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 shadow-xl overflow-hidden"
                style={{ zIndex: Z_INDEX.DROPDOWN + 1 }}
              >
                <div className="px-3 py-2.5 border-b border-zinc-200 dark:border-zinc-800">
                  <div className="text-[11px] text-zinc-500">Signed in as</div>
                  <div className="text-xs font-medium text-zinc-900 dark:text-zinc-100 truncate">
                    {user?.email}
                  </div>
                </div>
                <div className="p-1">
                  <button
                    onClick={() => {
                      setShowMenu(false);
                      setShowTAKSettings(true);
                    }}
                    className="w-full flex items-center gap-2 px-2.5 py-2 text-left text-sm text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-lg transition-colors"
                  >
                    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
                      <circle cx="12" cy="12" r="3" />
                    </svg>
                    TAK settings
                  </button>
                  <button
                    onClick={() => {
                      setShowMenu(false);
                      handleLogout();
                    }}
                    className="w-full flex items-center gap-2 px-2.5 py-2 text-left text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-950/30 rounded-lg transition-colors"
                  >
                    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4M16 17l5-5-5-5M21 12H9" />
                    </svg>
                    Sign out
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Coming Soon Dialog */}
      {comingSoonDialog.show && (
        <>
          <div
            className="fixed inset-0 bg-black/40 backdrop-blur-sm animate-in fade-in duration-150"
            style={{ zIndex: Z_INDEX.MODAL }}
            onClick={closeComingSoonDialog}
          />
          <div
            className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-md animate-in zoom-in-95 duration-150 px-4"
            style={{ zIndex: Z_INDEX.MODAL + 1 }}
          >
            <div className="rounded-xl bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 shadow-2xl overflow-hidden">
              <div className="px-6 py-5 border-b border-zinc-200 dark:border-zinc-800">
                <h3 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">
                  Coming soon
                </h3>
                <p className="text-sm text-zinc-500 dark:text-zinc-400 mt-0.5">
                  {comingSoonDialog.feature} is in development.
                </p>
              </div>
              <div className="px-6 py-4 text-sm text-zinc-600 dark:text-zinc-400 leading-relaxed">
                We&apos;re actively working on this feature. Check back soon for updates.
              </div>
              <div className="px-6 py-4 border-t border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/50 flex justify-end">
                <button
                  onClick={closeComingSoonDialog}
                  className="px-4 py-2 rounded-lg bg-zinc-900 text-white text-sm font-medium hover:bg-zinc-800 dark:bg-white dark:text-zinc-900 dark:hover:bg-zinc-200 transition-colors"
                >
                  Got it
                </button>
              </div>
            </div>
          </div>
        </>
      )}

      {showTicketModal && <TicketCreationModal onClose={() => setShowTicketModal(false)} />}
      <TAKSettingsModal isOpen={showTAKSettings} onClose={() => setShowTAKSettings(false)} />
    </header>
  );
}
