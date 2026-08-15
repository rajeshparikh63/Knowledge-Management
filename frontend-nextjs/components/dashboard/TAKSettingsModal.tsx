"use client";

import { useState, useEffect } from "react";
import { configureTAK, getTAKConfig, deleteTAKConfig } from "@/lib/api/tak";
import { useChatStore } from "@/lib/stores/chatStore";
import { Z_INDEX } from "@/lib/constants/zIndex";

interface TAKSettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function TAKSettingsModal({ isOpen, onClose }: TAKSettingsModalProps) {
  const { setTAKCredentials, setTAKEnabled } = useChatStore();

  const [takEnabled, setLocalTAKEnabled] = useState(false);
  const [takHost, setTakHost] = useState("");
  const [takPort, setTakPort] = useState(8087);
  const [takUsername, setTakUsername] = useState("");
  const [takPassword, setTakPassword] = useState("");
  const [agentCallsign, setAgentCallsign] = useState("SoldierIQ-Agent");
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  // Fetch TAK config when modal opens
  useEffect(() => {
    if (isOpen) {
      loadTAKConfig();
    }
  }, [isOpen]);

  const loadTAKConfig = async () => {
    setIsLoading(true);
    setError("");
    try {
      const config = await getTAKConfig();
      setLocalTAKEnabled(config.tak_enabled);
      setTakHost(config.tak_host);
      setTakPort(config.tak_port);
      setTakUsername(config.tak_username);
      setAgentCallsign(config.agent_callsign);
      // Password is not returned from backend for security
      setTakPassword("");
    } catch (err: any) {
      // TAK not configured yet - show empty form
      // TAK not configured yet - expected on first load
    } finally {
      setIsLoading(false);
    }
  };

  const handleSave = async () => {
    setError("");
    setSuccess("");

    // Validation
    if (takEnabled) {
      if (!takHost.trim()) {
        setError("TAK Host is required");
        return;
      }
      if (!takPort || takPort < 1 || takPort > 65535) {
        setError("Valid TAK Port is required (1-65535)");
        return;
      }
      // Username and password are optional for public servers
    }

    setIsSaving(true);
    try {
      const config = await configureTAK({
        tak_enabled: takEnabled,
        tak_host: takHost.trim(),
        tak_port: takPort,
        tak_username: takUsername.trim(),
        tak_password: takPassword.trim(),
        agent_callsign: agentCallsign.trim(),
      });

      // Store password in localStorage for chat integration (can be empty for public servers)
      if (takEnabled) {
        if (takPassword) {
          localStorage.setItem('tak_password', takPassword);
        } else {
          localStorage.removeItem('tak_password');
        }

        // Update chat store with credentials
        setTAKCredentials({
          tak_host: config.tak_host,
          tak_port: config.tak_port,
          tak_username: config.tak_username,
          tak_password: takPassword,
          agent_callsign: config.agent_callsign,
        });
        setTAKEnabled(true);
      } else {
        localStorage.removeItem('tak_password');
        setTAKCredentials(null);
        setTAKEnabled(false);
      }

      setSuccess("TAK configuration saved successfully!");
      setTimeout(() => {
        onClose();
      }, 1500);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || "Failed to save TAK configuration");
    } finally {
      setIsSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!confirm("Are you sure you want to delete the TAK configuration?")) {
      return;
    }

    setIsSaving(true);
    setError("");
    try {
      await deleteTAKConfig();
      localStorage.removeItem('tak_password');
      setTAKCredentials(null);
      setTAKEnabled(false);
      setSuccess("TAK configuration deleted successfully!");
      setTimeout(() => {
        onClose();
      }, 1500);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || "Failed to delete TAK configuration");
    } finally {
      setIsSaving(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center" style={{ zIndex: Z_INDEX.MODAL }}>
      <div className="bg-white dark:bg-slate-900 border-2 border-amber-400/30 w-full max-w-2xl max-h-[90vh] overflow-y-auto tactical-panel">
        {/* Header */}
        <div className="border-b-2 border-amber-400/20 p-6 bg-gradient-to-r from-slate-50 to-amber-50/20 dark:from-slate-900 dark:to-amber-900/10">
          <h2 className="text-2xl font-bold text-slate-900 dark:text-amber-400 tracking-wider">
            TAK CONFIGURATION
          </h2>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
            Configure Team Awareness Kit integration for AI agent
          </p>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {isLoading ? (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-amber-400 mx-auto"></div>
              <p className="text-slate-600 dark:text-slate-400 mt-4">Loading configuration...</p>
            </div>
          ) : (
            <>
              {/* Enable TAK Toggle */}
              <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-800/50 border border-amber-400/20">
                <div>
                  <label className="text-sm font-bold text-slate-900 dark:text-amber-400 tracking-wide">
                    ENABLE TAK INTEGRATION
                  </label>
                  <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">
                    Allow AI agent to interact with TAK network
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => setLocalTAKEnabled(!takEnabled)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    takEnabled ? 'bg-amber-500' : 'bg-slate-300 dark:bg-slate-700'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      takEnabled ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>

              {takEnabled && (
                <>
                  {/* TAK Host */}
                  <div>
                    <label className="block text-sm font-bold text-slate-900 dark:text-amber-400 tracking-wide mb-2">
                      TAK HOST *
                    </label>
                    <input
                      type="text"
                      value={takHost}
                      onChange={(e) => setTakHost(e.target.value)}
                      placeholder="e.g., tak.company.com or 192.168.1.100"
                      className="w-full px-4 py-2 bg-white dark:bg-slate-800 border-2 border-slate-300 dark:border-amber-400/20 text-slate-900 dark:text-slate-100 focus:border-amber-500 dark:focus:border-amber-400 focus:outline-none font-mono"
                    />
                  </div>

                  {/* TAK Port */}
                  <div>
                    <label className="block text-sm font-bold text-slate-900 dark:text-amber-400 tracking-wide mb-2">
                      TAK PORT *
                    </label>
                    <input
                      type="number"
                      value={takPort}
                      onChange={(e) => setTakPort(parseInt(e.target.value) || 8087)}
                      min="1"
                      max="65535"
                      className="w-full px-4 py-2 bg-white dark:bg-slate-800 border-2 border-slate-300 dark:border-amber-400/20 text-slate-900 dark:text-slate-100 focus:border-amber-500 dark:focus:border-amber-400 focus:outline-none font-mono"
                    />
                  </div>

                  {/* TAK Username */}
                  <div>
                    <label className="block text-sm font-bold text-slate-900 dark:text-amber-400 tracking-wide mb-2">
                      TAK USERNAME
                    </label>
                    <input
                      type="text"
                      value={takUsername}
                      onChange={(e) => setTakUsername(e.target.value)}
                      placeholder="e.g., soldieriq-agent (optional for public servers)"
                      className="w-full px-4 py-2 bg-white dark:bg-slate-800 border-2 border-slate-300 dark:border-amber-400/20 text-slate-900 dark:text-slate-100 focus:border-amber-500 dark:focus:border-amber-400 focus:outline-none font-mono"
                    />
                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                      Optional - leave empty for public servers
                    </p>
                  </div>

                  {/* TAK Password */}
                  <div>
                    <label className="block text-sm font-bold text-slate-900 dark:text-amber-400 tracking-wide mb-2">
                      TAK PASSWORD
                    </label>
                    <input
                      type="password"
                      value={takPassword}
                      onChange={(e) => setTakPassword(e.target.value)}
                      placeholder="Enter TAK server password (optional)"
                      className="w-full px-4 py-2 bg-white dark:bg-slate-800 border-2 border-slate-300 dark:border-amber-400/20 text-slate-900 dark:text-slate-100 focus:border-amber-500 dark:focus:border-amber-400 focus:outline-none font-mono"
                    />
                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                      Optional - password is stored locally for chat integration
                    </p>
                  </div>

                  {/* Agent Callsign */}
                  <div>
                    <label className="block text-sm font-bold text-slate-900 dark:text-amber-400 tracking-wide mb-2">
                      AGENT CALLSIGN
                    </label>
                    <input
                      type="text"
                      value={agentCallsign}
                      onChange={(e) => setAgentCallsign(e.target.value)}
                      placeholder="SoldierIQ-Agent"
                      className="w-full px-4 py-2 bg-white dark:bg-slate-800 border-2 border-slate-300 dark:border-amber-400/20 text-slate-900 dark:text-slate-100 focus:border-amber-500 dark:focus:border-amber-400 focus:outline-none font-mono"
                    />
                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                      Identifier shown on TAK network for AI agent
                    </p>
                  </div>
                </>
              )}

              {/* Error Message */}
              {error && (
                <div className="p-4 bg-red-50 dark:bg-red-900/20 border-2 border-red-500/50 text-red-700 dark:text-red-400">
                  <p className="font-semibold">ERROR</p>
                  <p className="text-sm">{error}</p>
                </div>
              )}

              {/* Success Message */}
              {success && (
                <div className="p-4 bg-green-50 dark:bg-green-900/20 border-2 border-green-500/50 text-green-700 dark:text-green-400">
                  <p className="font-semibold">SUCCESS</p>
                  <p className="text-sm">{success}</p>
                </div>
              )}

              {/* Actions */}
              <div className="flex gap-4 pt-4">
                <button
                  onClick={handleSave}
                  disabled={isSaving}
                  className="flex-1 px-6 py-3 bg-amber-500 hover:bg-amber-600 text-white font-bold tracking-wider transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{
                    clipPath: 'polygon(8px 0, 100% 0, 100% calc(100% - 8px), calc(100% - 8px) 100%, 0 100%, 0 8px)'
                  }}
                >
                  {isSaving ? 'SAVING...' : 'SAVE CONFIGURATION'}
                </button>

                {takHost && (
                  <button
                    onClick={handleDelete}
                    disabled={isSaving}
                    className="px-6 py-3 bg-red-500 hover:bg-red-600 text-white font-bold tracking-wider transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    style={{
                      clipPath: 'polygon(8px 0, 100% 0, 100% calc(100% - 8px), calc(100% - 8px) 100%, 0 100%, 0 8px)'
                    }}
                  >
                    DELETE
                  </button>
                )}

                <button
                  onClick={onClose}
                  disabled={isSaving}
                  className="px-6 py-3 bg-slate-500 hover:bg-slate-600 text-white font-bold tracking-wider transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{
                    clipPath: 'polygon(8px 0, 100% 0, 100% calc(100% - 8px), calc(100% - 8px) 100%, 0 100%, 0 8px)'
                  }}
                >
                  CANCEL
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
