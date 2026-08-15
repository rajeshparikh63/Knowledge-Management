"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { jsPDF } from "jspdf";

interface ReportViewerProps {
  reportContent: string;
  reportTitle: string;
  onClose: () => void;
}

export default function ReportViewer({ reportContent, reportTitle, onClose }: ReportViewerProps) {
  const [copied, setCopied] = useState(false);
  const [downloading, setDownloading] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(reportContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      // Copy failed silently
    }
  };

  const handleDownloadPDF = async () => {
    try {
      setDownloading(true);

      // Create a new jsPDF instance
      const pdf = new jsPDF({
        orientation: "portrait",
        unit: "mm",
        format: "a4",
      });

      // Set up styling
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const margin = 20;
      const maxWidth = pageWidth - margin * 2;
      let yPosition = margin;

      // Add title
      pdf.setFontSize(18);
      pdf.setFont("helvetica", "bold");
      pdf.text(reportTitle, margin, yPosition);
      yPosition += 12;

      // Process markdown content
      const lines = reportContent.split('\n');

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];

        // Check if we need a new page
        if (yPosition > pageHeight - margin) {
          pdf.addPage();
          yPosition = margin;
        }

        // Skip empty lines but add space
        if (!line.trim()) {
          yPosition += 3;
          continue;
        }

        // Handle H1 headings (# )
        if (line.startsWith('# ')) {
          pdf.setFontSize(16);
          pdf.setFont("helvetica", "bold");
          const text = line.substring(2);
          const wrappedLines = pdf.splitTextToSize(text, maxWidth);
          for (const wrappedLine of wrappedLines) {
            if (yPosition > pageHeight - margin) {
              pdf.addPage();
              yPosition = margin;
            }
            pdf.text(wrappedLine, margin, yPosition);
            yPosition += 8;
          }
          yPosition += 3;
          continue;
        }

        // Handle H2 headings (## )
        if (line.startsWith('## ')) {
          pdf.setFontSize(14);
          pdf.setFont("helvetica", "bold");
          const text = line.substring(3);
          const wrappedLines = pdf.splitTextToSize(text, maxWidth);
          for (const wrappedLine of wrappedLines) {
            if (yPosition > pageHeight - margin) {
              pdf.addPage();
              yPosition = margin;
            }
            pdf.text(wrappedLine, margin, yPosition);
            yPosition += 7;
          }
          yPosition += 2;
          continue;
        }

        // Handle H3 headings (### )
        if (line.startsWith('### ')) {
          pdf.setFontSize(12);
          pdf.setFont("helvetica", "bold");
          const text = line.substring(4);
          const wrappedLines = pdf.splitTextToSize(text, maxWidth);
          for (const wrappedLine of wrappedLines) {
            if (yPosition > pageHeight - margin) {
              pdf.addPage();
              yPosition = margin;
            }
            pdf.text(wrappedLine, margin, yPosition);
            yPosition += 6;
          }
          yPosition += 2;
          continue;
        }

        // Handle unordered list items (- or *)
        if (line.match(/^[\s]*[-*]\s/)) {
          pdf.setFontSize(10);
          pdf.setFont("helvetica", "normal");
          const indent = line.search(/[-*]/);
          const text = line.substring(line.indexOf(' ', indent) + 1);
          const wrappedLines = pdf.splitTextToSize('• ' + text, maxWidth - indent * 2);
          for (const wrappedLine of wrappedLines) {
            if (yPosition > pageHeight - margin) {
              pdf.addPage();
              yPosition = margin;
            }
            pdf.text(wrappedLine, margin + indent * 2, yPosition);
            yPosition += 5;
          }
          continue;
        }

        // Handle ordered list items (1. 2. etc.)
        if (line.match(/^[\s]*\d+\.\s/)) {
          pdf.setFontSize(10);
          pdf.setFont("helvetica", "normal");
          const indent = line.search(/\d/);
          const wrappedLines = pdf.splitTextToSize(line.substring(indent), maxWidth - indent * 2);
          for (const wrappedLine of wrappedLines) {
            if (yPosition > pageHeight - margin) {
              pdf.addPage();
              yPosition = margin;
            }
            pdf.text(wrappedLine, margin + indent * 2, yPosition);
            yPosition += 5;
          }
          continue;
        }

        // Handle bold text (**text** or __text__)
        let processedLine = line.replace(/\*\*([^*]+)\*\*/g, '$1').replace(/__([^_]+)__/g, '$1');

        // Handle italic text (*text* or _text_)
        processedLine = processedLine.replace(/\*([^*]+)\*/g, '$1').replace(/_([^_]+)_/g, '$1');

        // Handle inline code (`code`)
        processedLine = processedLine.replace(/`([^`]+)`/g, '$1');

        // Regular paragraph text
        pdf.setFontSize(10);
        pdf.setFont("helvetica", "normal");
        const wrappedLines = pdf.splitTextToSize(processedLine, maxWidth);
        for (const wrappedLine of wrappedLines) {
          if (yPosition > pageHeight - margin) {
            pdf.addPage();
            yPosition = margin;
          }
          pdf.text(wrappedLine, margin, yPosition);
          yPosition += 5;
        }
      }

      // Save the PDF
      const fileName = `${reportTitle.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.pdf`;
      pdf.save(fileName);

      setDownloading(false);
    } catch (err) {
      // PDF generation failed
      setDownloading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-slate-200/90 dark:bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="w-full h-full max-w-5xl max-h-[90vh] bg-white dark:bg-slate-950 border border-blue-200 dark:border-amber-400/30 shadow-2xl flex flex-col">
        {/* Header */}
        <div className="border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/50 backdrop-blur-sm p-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <svg className="w-6 h-6 text-blue-600 dark:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <div>
              <h2 className="text-lg font-bold text-blue-700 dark:text-amber-400 tracking-wider">{reportTitle}</h2>
              <p className="text-[10px] text-slate-500 tracking-wider">GENERATED REPORT</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Copy Button */}
            <button
              onClick={handleCopy}
              className="tactical-panel px-3 py-2 hover:bg-slate-200 dark:hover:bg-slate-800 transition-colors group flex items-center gap-2"
              title="Copy to clipboard"
            >
              {copied ? (
                <>
                  <svg className="w-4 h-4 text-tactical-green" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="text-xs text-tactical-green font-semibold tracking-wide">COPIED</span>
                </>
              ) : (
                <>
                  <svg className="w-4 h-4 text-blue-600 dark:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                  <span className="text-xs text-slate-700 dark:text-slate-300 font-semibold tracking-wide">COPY</span>
                </>
              )}
            </button>

            {/* Download PDF Button */}
            <button
              onClick={handleDownloadPDF}
              disabled={downloading}
              className="tactical-panel px-3 py-2 hover:bg-slate-200 dark:hover:bg-slate-800 transition-colors group flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              title="Download as PDF"
            >
              {downloading ? (
                <>
                  <div className="w-4 h-4 border-2 border-blue-600 dark:border-amber-400 border-t-transparent rounded-full animate-spin"></div>
                  <span className="text-xs text-slate-700 dark:text-slate-300 font-semibold tracking-wide">DOWNLOADING...</span>
                </>
              ) : (
                <>
                  <svg className="w-4 h-4 text-blue-600 dark:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <span className="text-xs text-slate-700 dark:text-slate-300 font-semibold tracking-wide">PDF</span>
                </>
              )}
            </button>

            {/* Close Button */}
            <button
              onClick={onClose}
              className="tactical-panel p-2 hover:bg-slate-200 dark:hover:bg-slate-800 transition-colors group"
              title="Close"
            >
              <svg className="w-5 h-5 text-slate-400 group-hover:text-blue-600 dark:group-hover:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Report Content */}
        <div className="flex-1 overflow-y-auto px-8 py-10 md:px-12 lg:px-16 tactical-scrollbar bg-white dark:bg-[#0d1117]">
          <div className="max-w-3xl mx-auto">
            <div className="
              prose prose-lg dark:prose-invert
              font-[Inter,system-ui,sans-serif]
              text-[16px] leading-[2] tracking-[0.01em]
              text-slate-700 dark:text-slate-200

              prose-headings:font-display prose-headings:tracking-wide prose-headings:leading-snug
              dark:prose-headings:text-amber-400 prose-headings:text-blue-700
              prose-h1:text-[26px] prose-h1:font-bold prose-h1:mb-8 prose-h1:mt-14 prose-h1:pb-3 prose-h1:border-b prose-h1:border-slate-200 dark:prose-h1:border-slate-700/50
              prose-h2:text-[22px] prose-h2:font-bold prose-h2:mb-6 prose-h2:mt-12
              prose-h3:text-[18px] prose-h3:font-semibold prose-h3:mb-5 prose-h3:mt-10
              prose-h4:text-[16px] prose-h4:font-semibold prose-h4:mb-4 prose-h4:mt-8 dark:prose-h4:text-amber-300 prose-h4:text-blue-600 prose-h4:italic

              prose-p:mb-6 prose-p:leading-[2]

              prose-ul:my-6 prose-ul:space-y-3
              prose-ol:my-6 prose-ol:space-y-3
              prose-li:leading-[1.9] prose-li:pl-2 prose-li:my-1
              [&_li>p]:my-1 [&_li>p]:leading-[1.9]
              [&_ul]:list-disc [&_ul]:pl-6
              [&_ol]:list-decimal [&_ol]:pl-6
              [&_ul_ul]:my-2 [&_ol_ol]:my-2

              prose-blockquote:border-l-4 dark:prose-blockquote:border-amber-400/40 prose-blockquote:border-blue-400/40
              prose-blockquote:pl-6 prose-blockquote:my-8 prose-blockquote:py-2 prose-blockquote:italic
              dark:prose-blockquote:text-slate-300 prose-blockquote:text-slate-600
              dark:prose-blockquote:bg-slate-800/30 prose-blockquote:bg-blue-50/50 prose-blockquote:rounded-r-lg

              dark:prose-strong:text-white prose-strong:text-slate-900 prose-strong:font-semibold
              dark:prose-em:text-slate-300 prose-em:text-slate-600

              prose-a:text-tactical-green prose-a:underline prose-a:underline-offset-2 hover:prose-a:text-green-300

              dark:prose-code:text-amber-300 prose-code:text-blue-600
              dark:prose-code:bg-slate-800/60 prose-code:bg-slate-100
              prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded-md prose-code:text-[14px] prose-code:font-mono
              prose-code:before:content-[''] prose-code:after:content-['']

              dark:prose-pre:bg-slate-900 prose-pre:bg-slate-50
              dark:prose-pre:border prose-pre:border dark:prose-pre:border-slate-700/50 prose-pre:border-slate-200
              prose-pre:my-8 prose-pre:p-5 prose-pre:rounded-xl prose-pre:overflow-x-auto

              prose-hr:my-16 prose-hr:py-4 dark:prose-hr:border-slate-600 prose-hr:border-slate-300 prose-hr:border-t-2

              prose-table:my-8 prose-table:text-[15px] prose-table:w-full
              dark:prose-thead:border-slate-600 prose-thead:border-slate-300 prose-thead:border-b-2
              dark:prose-th:text-amber-400 prose-th:text-blue-700 prose-th:py-3 prose-th:px-5 prose-th:text-left prose-th:font-semibold prose-th:text-[14px] prose-th:uppercase prose-th:tracking-wider
              dark:prose-td:border-slate-800 prose-td:border-slate-100 prose-td:py-3.5 prose-td:px-5 prose-td:border-t

              [&>*:first-child]:mt-0 [&>*:last-child]:mb-0
            ">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {reportContent}
              </ReactMarkdown>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
