"use client";

import { useState } from "react";
import { useAuthStore } from "@/lib/stores/authStore";
import { Z_INDEX } from "@/lib/constants/zIndex";

interface TicketCreationModalProps {
  onClose: () => void;
}

export default function TicketCreationModal({ onClose }: TicketCreationModalProps) {
  const { user } = useAuthStore();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ticketId, setTicketId] = useState<string | null>(null);

  const [formData, setFormData] = useState({
    name: user?.email?.split('@')[0] || '',
    email: user?.email || '',
    subject: '',
    description: '',
    priority: 'normal',
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);

    try {
      const response = await fetch('/api/tickets/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to create ticket');
      }

      setSuccess(true);
      setTicketId(data.ticket_id);
    } catch (err: any) {
      setError(err.message || 'Failed to submit ticket');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  if (success) {
    return (
      <>
        {/* Backdrop */}
        <div
          className="fixed inset-0 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200"
          style={{ zIndex: Z_INDEX.MODAL }}
          onClick={onClose}
        />

        {/* Success Dialog */}
        <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-md animate-in zoom-in-95 duration-200" style={{ zIndex: Z_INDEX.MODAL + 1 }}>
          <div className="relative bg-white dark:bg-card border-2 border-green-500 shadow-2xl mx-4">
            {/* Tactical corners */}
            <div className="absolute top-0 left-0 w-4 h-4 border-l-2 border-t-2 border-green-500"></div>
            <div className="absolute top-0 right-0 w-4 h-4 border-r-2 border-t-2 border-green-500"></div>
            <div className="absolute bottom-0 left-0 w-4 h-4 border-l-2 border-b-2 border-green-500"></div>
            <div className="absolute bottom-0 right-0 w-4 h-4 border-r-2 border-b-2 border-green-500"></div>

            {/* Header */}
            <div className="bg-gradient-to-r from-green-500/10 to-tactical-green/10 border-b-2 border-green-500/30 p-6">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-green-500/20 border border-green-500/50 flex items-center justify-center">
                  <svg className="w-6 h-6 text-tactical-green" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-xl font-bold text-tactical-green tracking-wider">TICKET SUBMITTED</h3>
                  <p className="text-xs text-muted-foreground dark:text-muted-foreground tracking-widest mt-1">CONFIRMATION</p>
                </div>
              </div>
            </div>

            {/* Content */}
            <div className="p-6 space-y-4">
              <div className="bg-muted dark:bg-muted/50 border border-border dark:border-border p-4">
                <p className="text-sm font-semibold text-foreground dark:text-foreground mb-2">
                  Your support ticket has been created successfully!
                </p>
                {ticketId && (
                  <div className="text-xs text-muted-foreground dark:text-muted-foreground mt-2">
                    <span className="font-semibold">Ticket ID:</span> #{ticketId}
                  </div>
                )}
              </div>

              <div className="flex items-start gap-2 text-xs text-muted-foreground dark:text-muted-foreground">
                <svg className="w-4 h-4 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p>Our support team will review your ticket and respond via email shortly.</p>
              </div>
            </div>

            {/* Footer */}
            <div className="border-t-2 border-green-500/30 p-4 bg-muted dark:bg-muted/30">
              <button
                onClick={onClose}
                className="w-full px-4 py-3 bg-gradient-to-r from-tactical-green to-green-600 hover:from-green-600 hover:to-green-700 text-white font-bold text-sm tracking-wider transition-all duration-200 border border-tactical-green"
                style={{
                  clipPath: 'polygon(6px 0, 100% 0, 100% calc(100% - 6px), calc(100% - 6px) 100%, 0 100%, 0 6px)'
                }}
              >
                <span className="flex items-center justify-center gap-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  CLOSE
                </span>
              </button>
            </div>
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200"
        style={{ zIndex: Z_INDEX.MODAL }}
        onClick={onClose}
      />

      {/* Modal */}
      <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-2xl animate-in zoom-in-95 duration-200" style={{ zIndex: Z_INDEX.MODAL + 1 }}>
        <div className="relative bg-white dark:bg-card border-2 border-brand/50 dark:border-brand/50 shadow-2xl mx-4">
          {/* Tactical corners */}
          <div className="absolute top-0 left-0 w-4 h-4 border-l-2 border-t-2 border-brand/50 dark:border-brand"></div>
          <div className="absolute top-0 right-0 w-4 h-4 border-r-2 border-t-2 border-brand/50 dark:border-brand"></div>
          <div className="absolute bottom-0 left-0 w-4 h-4 border-l-2 border-b-2 border-brand/50 dark:border-brand"></div>
          <div className="absolute bottom-0 right-0 w-4 h-4 border-r-2 border-b-2 border-brand/50 dark:border-brand"></div>

          {/* Header */}
          <div className="bg-gradient-to-r from-brand/10 to-brand/10 dark:from-brand/10 dark:to-brand/10 border-b-2 border-brand/30 dark:border-brand/30 p-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-brand/20 dark:bg-brand/20 border border-brand/50 dark:border-brand/50 flex items-center justify-center">
                  <svg className="w-6 h-6 text-brand dark:text-brand" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 5v2m0 4v2m0 4v2M5 5a2 2 0 00-2 2v3a2 2 0 110 4v3a2 2 0 002 2h14a2 2 0 002-2v-3a2 2 0 110-4V7a2 2 0 00-2-2H5z" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-xl font-bold text-brand dark:text-brand tracking-wider">CREATE SUPPORT TICKET</h3>
                  <p className="text-xs text-muted-foreground dark:text-muted-foreground tracking-widest mt-1">SUBMIT YOUR REQUEST</p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="p-2 hover:bg-muted dark:hover:bg-muted rounded transition-colors"
              >
                <svg className="w-5 h-5 text-muted-foreground hover:text-muted-foreground dark:hover:text-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="p-6 space-y-4 max-h-[60vh] overflow-y-auto tactical-scrollbar">
            {error && (
              <div className="bg-red-50 dark:bg-red-900/20 border-2 border-red-400 dark:border-red-400/50 p-4">
                <div className="flex items-start gap-2">
                  <svg className="w-5 h-5 text-red-600 dark:text-red-400 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <div>
                    <div className="text-sm font-semibold text-red-700 dark:text-red-400 mb-1">ERROR</div>
                    <div className="text-xs text-red-600 dark:text-red-300">{error}</div>
                  </div>
                </div>
              </div>
            )}

            <div className="grid grid-cols-2 gap-4">
              {/* Name */}
              <div>
                <label className="block text-xs text-muted-foreground dark:text-muted-foreground tracking-wider mb-2 font-semibold uppercase">
                  Your Name *
                </label>
                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  required
                  className="w-full px-3 py-2 bg-white dark:bg-muted border border-border dark:border-border text-foreground dark:text-foreground text-sm focus:outline-none focus:border-brand/50 dark:focus:border-brand transition-colors"
                  style={{
                    clipPath: 'polygon(4px 0, 100% 0, 100% calc(100% - 4px), calc(100% - 4px) 100%, 0 100%, 0 4px)'
                  }}
                />
              </div>

              {/* Email */}
              <div>
                <label className="block text-xs text-muted-foreground dark:text-muted-foreground tracking-wider mb-2 font-semibold uppercase">
                  Email Address *
                </label>
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  required
                  className="w-full px-3 py-2 bg-white dark:bg-muted border border-border dark:border-border text-foreground dark:text-foreground text-sm focus:outline-none focus:border-brand/50 dark:focus:border-brand transition-colors"
                  style={{
                    clipPath: 'polygon(4px 0, 100% 0, 100% calc(100% - 4px), calc(100% - 4px) 100%, 0 100%, 0 4px)'
                  }}
                />
              </div>
            </div>

            {/* Priority */}
            <div>
              <label className="block text-xs text-muted-foreground dark:text-muted-foreground tracking-wider mb-2 font-semibold uppercase">
                Priority Level
              </label>
              <select
                name="priority"
                value={formData.priority}
                onChange={handleChange}
                className="w-full px-3 py-2 bg-white dark:bg-muted border border-border dark:border-border text-foreground dark:text-foreground text-sm focus:outline-none focus:border-brand/50 dark:focus:border-brand transition-colors"
                style={{
                  clipPath: 'polygon(4px 0, 100% 0, 100% calc(100% - 4px), calc(100% - 4px) 100%, 0 100%, 0 4px)'
                }}
              >
                <option value="low">Low - General inquiry</option>
                <option value="normal">Normal - Standard issue</option>
                <option value="high">High - Urgent issue</option>
                <option value="urgent">Urgent - Critical problem</option>
              </select>
            </div>

            {/* Subject */}
            <div>
              <label className="block text-xs text-muted-foreground dark:text-muted-foreground tracking-wider mb-2 font-semibold uppercase">
                Subject *
              </label>
              <input
                type="text"
                name="subject"
                value={formData.subject}
                onChange={handleChange}
                required
                placeholder="Brief description of your issue"
                className="w-full px-3 py-2 bg-white dark:bg-muted border border-border dark:border-border text-foreground dark:text-foreground text-sm focus:outline-none focus:border-brand/50 dark:focus:border-brand transition-colors placeholder:text-muted-foreground dark:placeholder:text-muted-foreground"
                style={{
                  clipPath: 'polygon(4px 0, 100% 0, 100% calc(100% - 4px), calc(100% - 4px) 100%, 0 100%, 0 4px)'
                }}
              />
            </div>

            {/* Description */}
            <div>
              <label className="block text-xs text-muted-foreground dark:text-muted-foreground tracking-wider mb-2 font-semibold uppercase">
                Description *
              </label>
              <textarea
                name="description"
                value={formData.description}
                onChange={handleChange}
                required
                rows={6}
                placeholder="Provide detailed information about your issue..."
                className="w-full px-3 py-2 bg-white dark:bg-muted border border-border dark:border-border text-foreground dark:text-foreground text-sm focus:outline-none focus:border-brand/50 dark:focus:border-brand transition-colors placeholder:text-muted-foreground dark:placeholder:text-muted-foreground resize-none"
                style={{
                  clipPath: 'polygon(4px 0, 100% 0, 100% calc(100% - 4px), calc(100% - 4px) 100%, 0 100%, 0 4px)'
                }}
              />
            </div>

            <div className="flex items-start gap-2 text-xs text-muted-foreground dark:text-muted-foreground bg-brand/10 dark:bg-muted/50 border border-brand/30 dark:border-border p-3">
              <svg className="w-4 h-4 flex-shrink-0 mt-0.5 text-brand dark:text-brand" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p>Our support team will review your ticket and respond to your email address within 24-48 hours.</p>
            </div>
          </form>

          {/* Footer */}
          <div className="border-t-2 border-brand/30 dark:border-brand/30 p-4 bg-muted dark:bg-muted/30 flex gap-3">
            <button
              type="button"
              onClick={onClose}
              disabled={isSubmitting}
              className="flex-1 px-4 py-3 bg-secondary dark:bg-secondary hover:bg-secondary dark:hover:bg-secondary disabled:bg-muted dark:disabled:bg-muted disabled:text-muted-foreground text-foreground dark:text-muted-foreground font-bold text-sm tracking-wider transition-all duration-200 border border-border dark:border-border"
              style={{
                clipPath: 'polygon(6px 0, 100% 0, 100% calc(100% - 6px), calc(100% - 6px) 100%, 0 100%, 0 6px)'
              }}
            >
              CANCEL
            </button>
            <button
              type="submit"
              onClick={handleSubmit}
              disabled={isSubmitting}
              className="flex-1 px-4 py-3 bg-gradient-to-r from-brand to-brand-hover dark:from-brand dark:to-brand-hover hover:from-brand hover:to-brand-hover dark:hover:from-brand dark:hover:to-brand-hover disabled:from-secondary disabled:to-secondary dark:disabled:from-secondary dark:disabled:to-muted text-white dark:text-foreground disabled:text-muted-foreground font-bold text-sm tracking-wider transition-all duration-200 border border-brand/50 dark:border-brand disabled:border-border"
              style={{
                clipPath: 'polygon(6px 0, 100% 0, 100% calc(100% - 6px), calc(100% - 6px) 100%, 0 100%, 0 6px)'
              }}
            >
              {isSubmitting ? (
                <span className="flex items-center justify-center gap-2">
                  <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin"></div>
                  SUBMITTING...
                </span>
              ) : (
                <span className="flex items-center justify-center gap-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                  </svg>
                  SUBMIT TICKET
                </span>
              )}
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
