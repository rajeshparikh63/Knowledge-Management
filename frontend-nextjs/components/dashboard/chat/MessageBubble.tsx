"use client";

import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ChatMessage } from "@/types";
import { DocumentSource } from "@/lib/stores/chatStore";

interface MessageBubbleProps {
  message: ChatMessage;
  sourceUrls: Map<string, string>;
  animationDelay: number;
  onCitationHover: (
    index: number,
    messageId: string,
    x: number,
    y: number
  ) => void;
  onCitationLeave: () => void;
  onCitationClick: (source: DocumentSource, url: string | undefined) => void;
  onOpenGraph?: (message: ChatMessage) => void;
}

const PROSE_CLASSES = `prose dark:prose-invert max-w-none font-sans
  text-[15px] leading-[1.75]
  prose-headings:font-semibold prose-headings:tracking-tight prose-headings:leading-tight
  prose-headings:text-zinc-900 dark:prose-headings:text-zinc-100
  prose-h1:text-2xl prose-h1:mb-6 prose-h1:mt-10
  prose-h2:text-xl prose-h2:mb-5 prose-h2:mt-8
  prose-h3:text-lg prose-h3:mb-4 prose-h3:mt-7
  prose-h4:text-base prose-h4:mb-3 prose-h4:mt-6
  prose-p:mb-5 prose-p:leading-[1.75]
  prose-p:text-zinc-700 dark:prose-p:text-zinc-300
  prose-ul:my-5 prose-ul:space-y-2
  prose-ol:my-5 prose-ol:space-y-2
  prose-li:text-zinc-700 dark:prose-li:text-zinc-300 prose-li:my-1
  prose-blockquote:border-l-2 prose-blockquote:border-zinc-300 dark:prose-blockquote:border-zinc-700
  prose-blockquote:pl-4 prose-blockquote:my-5 prose-blockquote:text-zinc-600 dark:prose-blockquote:text-zinc-400 prose-blockquote:italic prose-blockquote:not-italic
  prose-a:text-zinc-900 dark:prose-a:text-zinc-100 prose-a:underline prose-a:underline-offset-4
  prose-strong:text-zinc-900 dark:prose-strong:text-zinc-100 prose-strong:font-semibold
  prose-em:text-zinc-700 dark:prose-em:text-zinc-300
  prose-code:text-zinc-900 dark:prose-code:text-zinc-100
  prose-code:bg-zinc-100 dark:prose-code:bg-zinc-900
  prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-[13px] prose-code:font-mono
  prose-code:before:content-[''] prose-code:after:content-['']
  prose-pre:bg-zinc-50 dark:prose-pre:bg-zinc-950 prose-pre:border prose-pre:border-zinc-200 dark:prose-pre:border-zinc-800
  prose-pre:my-5 prose-pre:p-4 prose-pre:rounded-lg prose-pre:overflow-x-auto
  prose-hr:border-zinc-200 dark:prose-hr:border-zinc-800 prose-hr:my-8
  prose-table:my-5 prose-table:text-sm
  prose-thead:border-b prose-thead:border-zinc-200 dark:prose-thead:border-zinc-800
  prose-th:text-zinc-900 dark:prose-th:text-zinc-100 prose-th:py-2 prose-th:px-3 prose-th:text-left prose-th:font-semibold
  prose-td:border-t prose-td:border-zinc-100 dark:prose-td:border-zinc-900 prose-td:py-2 prose-td:px-3
  [&>*:first-child]:mt-0 [&>*:last-child]:mb-0
  [&_ul]:list-disc [&_ul]:pl-6
  [&_ol]:list-decimal [&_ol]:pl-6`;

function processContent(message: ChatMessage): string {
  let content = message.content || "";
  if (
    message.isStreaming !== true &&
    message.sources &&
    message.sources.length > 0
  ) {
    content = content.replace(/\[\s*(\d+)\s*\]/g, (match, id) => {
      const index = parseInt(id, 10);
      if (index > 0 && index <= message.sources!.length) {
        return ` [${index}](#source-${index})`;
      }
      return match;
    });
  }
  return content;
}

const MessageBubble = React.memo(function MessageBubble({
  message,
  sourceUrls,
  animationDelay,
  onCitationHover,
  onCitationLeave,
  onCitationClick,
  onOpenGraph,
}: MessageBubbleProps) {
  const hasGraph =
    !!message.graph &&
    ((message.graph.triples?.length ?? 0) > 0 ||
      (message.graph.anchors?.length ?? 0) > 0);
  const isUser = message.role === "user";

  const timeStr = new Date(message.timestamp).toLocaleTimeString("en-US", {
    hour12: true,
    hour: "numeric",
    minute: "2-digit",
  });

  if (isUser) {
    return (
      <div
        className="data-load flex justify-end"
        style={{ animationDelay: `${animationDelay}ms` }}
      >
        <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-zinc-100 dark:bg-zinc-900 px-4 py-2.5">
          <div className="text-[15px] text-zinc-900 dark:text-zinc-100 whitespace-pre-wrap leading-relaxed">
            {message.content}
          </div>
          <div className="text-[11px] text-zinc-500 dark:text-zinc-500 mt-1 text-right">
            {timeStr}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className="data-load flex justify-start"
      style={{ animationDelay: `${animationDelay}ms` }}
    >
      <div className="max-w-[92%] w-full">
        <div className="flex items-center gap-2 mb-2">
          <div className="w-6 h-6 rounded-md bg-zinc-900 dark:bg-white flex items-center justify-center flex-shrink-0">
            <svg
              className="w-3.5 h-3.5 text-white dark:text-zinc-900"
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
          <span className="text-xs font-medium text-zinc-900 dark:text-zinc-100">
            SoldierIQ
          </span>
          <span className="text-[11px] text-zinc-500">· {timeStr}</span>
        </div>

        <div className="pl-8">
          {message.content ? (
            <div className={PROSE_CLASSES}>
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  a: ({ node, href, children, ...props }) => {
                    if (href?.startsWith("#source-")) {
                      const index = parseInt(href.replace("#source-", ""), 10);
                      if (
                        !isNaN(index) &&
                        message.sources &&
                        message.sources[index - 1]
                      ) {
                        const source = message.sources[
                          index - 1
                        ] as DocumentSource;
                        const fileKey = source.file_key;
                        const finalUrl = sourceUrls.get(fileKey);

                        return (
                          <span
                            className="inline-flex items-center justify-center min-w-[18px] h-[18px] px-1 ml-0.5 text-[10px] font-semibold text-zinc-700 dark:text-zinc-300 bg-zinc-100 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-md cursor-pointer hover:bg-zinc-200 dark:hover:bg-zinc-800 transition-colors align-super"
                            onMouseEnter={(e) => {
                              const rect = e.currentTarget.getBoundingClientRect();
                              onCitationHover(
                                index - 1,
                                message.id,
                                rect.left + rect.width / 2,
                                rect.top
                              );
                            }}
                            onMouseLeave={onCitationLeave}
                            onClick={(e) => {
                              e.preventDefault();
                              e.stopPropagation();
                              onCitationClick(source, finalUrl);
                            }}
                          >
                            {index}
                          </span>
                        );
                      }
                    }
                    return (
                      <a
                        href={href}
                        {...props}
                        className="text-zinc-900 dark:text-zinc-100 underline underline-offset-4 hover:text-zinc-700 dark:hover:text-zinc-300"
                      >
                        {children}
                      </a>
                    );
                  },
                }}
              >
                {processContent(message)}
              </ReactMarkdown>
            </div>
          ) : (
            <div className="flex items-center gap-2 text-zinc-500 text-sm">
              <div className="w-3.5 h-3.5 border-2 border-zinc-300 border-t-zinc-700 dark:border-zinc-700 dark:border-t-zinc-300 rounded-full animate-spin" />
              Thinking…
            </div>
          )}

          {hasGraph && message.isStreaming !== true && onOpenGraph && (
            <button
              onClick={() => onOpenGraph(message)}
              className="mt-3 inline-flex items-center gap-1.5 text-[12px] font-medium px-2.5 py-1 rounded-md border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors"
              title="See how the retrieved entities and relations connect"
            >
              <svg
                className="w-3.5 h-3.5"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="6" cy="6" r="2.5" />
                <circle cx="18" cy="6" r="2.5" />
                <circle cx="12" cy="18" r="2.5" />
                <line x1="8" y1="7" x2="16" y2="7" />
                <line x1="7" y1="8" x2="11" y2="16" />
                <line x1="17" y1="8" x2="13" y2="16" />
              </svg>
              View knowledge graph
              <span className="text-[10px] text-zinc-500 dark:text-zinc-500">
                {message.graph!.anchors?.length ?? 0} entities ·{" "}
                {message.graph!.triples?.length ?? 0} relations
              </span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
});

export default MessageBubble;
