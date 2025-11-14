"use client";

import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import { streamChat, ChatMessage, ChatContext } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent } from "@/components/ui/card";
import { Trash2, Send, Loader2, Bot, User } from "lucide-react";
import TopNav from "@/components/top-nav";
import "katex/dist/katex.min.css";
import "highlight.js/styles/github.css";

const STORAGE_KEY = "spot-scam-chat-history";

type ChatAssistantProps = {
  initialContext?: ChatContext | null;
};

export default function ChatAssistant({ initialContext }: ChatAssistantProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState("");
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const originalHtmlOverflow = document.documentElement.style.overflow;
    const originalBodyOverflow = document.body.style.overflow;
    document.documentElement.style.overflow = "hidden";
    document.body.style.overflow = "hidden";
    return () => {
      document.documentElement.style.overflow = originalHtmlOverflow;
      document.body.style.overflow = originalBodyOverflow;
    };
  }, []);

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setMessages(parsed);
      } catch (e) {
        console.error("Failed to parse chat history:", e);
      }
    }
  }, []);

  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    }
  }, [messages]);

  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    const distanceFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
    const shouldStickToBottom = distanceFromBottom < 120 || distanceFromBottom === 0;

    if (shouldStickToBottom || streamingMessage || messages.length === 0) {
      container.scrollTo({
        top: container.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [messages, streamingMessage]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      role: "user",
      content: input.trim(),
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setStreamingMessage("");

    const context = messages.length === 0 ? initialContext : null;
    let accumulatedResponse = "";

    await streamChat(
      {
        message: userMessage.content,
        context,
        history: messages,
      },
      (chunk) => {
        accumulatedResponse += chunk;
        setStreamingMessage(accumulatedResponse);
      },
      () => {
        const assistantMessage: ChatMessage = {
          role: "assistant",
          content: accumulatedResponse,
          timestamp: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, assistantMessage]);
        setStreamingMessage("");
        setIsLoading(false);
      },
      (error) => {
        console.error("Chat error:", error);
        const errorMessage: ChatMessage = {
          role: "assistant",
          content: `Error: ${error.message}. Please make sure the GEMINI_API_KEY is set in your backend environment variables.`,
          timestamp: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, errorMessage]);
        setStreamingMessage("");
        setIsLoading(false);
      }
    );
  };

  const handleClearHistory = () => {
    if (window.confirm("Are you sure you want to clear the chat history?")) {
      setMessages([]);
      setStreamingMessage("");
      localStorage.removeItem(STORAGE_KEY);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="h-screen bg-white flex flex-col overflow-hidden">
      <TopNav />

      <main className="flex-1 mx-auto w-full max-w-6xl px-4 sm:px-8 py-6 flex flex-col overflow-hidden min-h-0">
        <div className="mb-4 flex items-center justify-between flex-shrink-0">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-slate-100 p-2.5 border border-slate-200">
              <Bot className="h-6 w-6 text-slate-700" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-slate-900">AI Assistant</h1>
              <p className="text-sm text-slate-600">Powered by Google Gemini</p>
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={handleClearHistory}
            className="border-slate-300 text-slate-700 hover:bg-slate-100"
          >
            <Trash2 className="mr-2 h-4 w-4" />
            Clear History
          </Button>
        </div>

        <Card className="bg-white border-slate-200 shadow-sm flex-1 flex flex-col overflow-hidden min-h-0 p-0">
          <CardContent className="p-0 flex-1 flex flex-col overflow-hidden min-h-0">
            {/* Messages Area */}
            <div
              ref={messagesContainerRef}
              className="flex-1 overflow-y-auto p-6 space-y-4 bg-slate-50 min-h-0"
            >
              {messages.length === 0 && !streamingMessage && (
                <div className="flex h-full items-center justify-center">
                  <div className="text-center space-y-3 max-w-md">
                    <div className="mx-auto rounded-full bg-slate-100 p-6 w-fit border border-slate-200">
                      <Bot className="h-10 w-10 text-slate-700" />
                    </div>
                    <h3 className="text-lg font-semibold text-slate-900">
                      Welcome to AI Assistant
                    </h3>
                    <p className="text-sm text-slate-600">
                      {initialContext?.prediction
                        ? "I can help you understand the fraud detection results and answer questions about the job posting."
                        : "Ask me anything about job fraud detection, or analyze a job posting on the Score page first."}
                    </p>
                  </div>
                </div>
              )}

              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex gap-3 ${message.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  {message.role === "assistant" && (
                    <div className="flex-shrink-0 rounded-lg bg-slate-100 p-2 h-fit border border-slate-200">
                      <Bot className="h-4 w-4 text-slate-700" />
                    </div>
                  )}
                  <div
                    className={`max-w-[85%] rounded-lg px-4 py-3 ${
                      message.role === "user"
                        ? "bg-black text-white"
                        : "bg-white text-slate-900 border border-slate-200 shadow-sm"
                    }`}
                  >
                    {message.role === "assistant" ? (
                      <div className="prose prose-sm max-w-none prose-slate">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm, remarkMath]}
                          rehypePlugins={[rehypeKatex, rehypeHighlight]}
                          components={{
                            p: ({ node, ...props }) => (
                              <p
                                className="mb-3 last:mb-0 leading-relaxed text-slate-900"
                                {...props}
                              />
                            ),
                            ul: ({ node, ...props }) => (
                              <ul
                                className="list-disc pl-6 mb-3 space-y-1.5 text-slate-900"
                                {...props}
                              />
                            ),
                            ol: ({ node, ...props }) => (
                              <ol
                                className="list-decimal pl-6 mb-3 space-y-1.5 text-slate-900"
                                {...props}
                              />
                            ),
                            li: ({ node, ...props }) => (
                              <li className="leading-relaxed text-slate-900" {...props} />
                            ),
                            h1: ({ node, ...props }) => (
                              <h1
                                className="text-xl font-bold mb-3 mt-4 first:mt-0 text-slate-900"
                                {...props}
                              />
                            ),
                            h2: ({ node, ...props }) => (
                              <h2
                                className="text-lg font-bold mb-2 mt-3 first:mt-0 text-slate-900"
                                {...props}
                              />
                            ),
                            h3: ({ node, ...props }) => (
                              <h3
                                className="text-base font-semibold mb-2 mt-3 first:mt-0 text-slate-900"
                                {...props}
                              />
                            ),
                            strong: ({ node, ...props }) => (
                              <strong className="font-semibold text-slate-900" {...props} />
                            ),
                            em: ({ node, ...props }) => (
                              <em className="italic text-slate-900" {...props} />
                            ),
                            blockquote: ({ node, ...props }) => (
                              <blockquote
                                className="border-l-4 border-slate-300 pl-4 italic my-3 text-slate-700"
                                {...props}
                              />
                            ),
                            table: ({ node, ...props }) => (
                              <div className="overflow-x-auto my-4 rounded-lg border border-slate-200">
                                <table
                                  className="min-w-full divide-y divide-slate-200"
                                  {...props}
                                />
                              </div>
                            ),
                            thead: ({ node, ...props }) => (
                              <thead className="bg-slate-50" {...props} />
                            ),
                            th: ({ node, ...props }) => (
                              <th
                                className="px-4 py-2 text-left text-xs font-medium text-slate-700 uppercase tracking-wider"
                                {...props}
                              />
                            ),
                            td: ({ node, ...props }) => (
                              <td
                                className="px-4 py-2 text-sm text-slate-900 border-t border-slate-200"
                                {...props}
                              />
                            ),
                            code: ({ node, inline, className, children, ...props }: any) => {
                              const content = String(children ?? "").replace(/\n+$/, "");
                              if (inline) {
                                return (
                                  <code
                                    className="px-1.5 py-0.5 bg-slate-100 text-slate-900 rounded text-sm font-mono border border-slate-200"
                                    {...props}
                                  >
                                    {content}
                                  </code>
                                );
                              }
                              return (
                                <pre className="my-3 overflow-x-auto rounded-lg bg-slate-50 border border-slate-200">
                                  <code
                                    className={`block p-4 text-sm font-mono ${className ?? ""}`}
                                    {...props}
                                  >
                                    {content}
                                  </code>
                                </pre>
                              );
                            },
                            pre: ({ node, ...props }) => (
                              <pre className="my-3 overflow-x-auto" {...props} />
                            ),
                            a: ({ node, ...props }) => (
                              <a
                                className="text-blue-600 hover:underline"
                                target="_blank"
                                rel="noopener noreferrer"
                                {...props}
                              />
                            ),
                            hr: ({ node, ...props }) => (
                              <hr className="my-4 border-slate-200" {...props} />
                            ),
                          }}
                        >
                          {message.content}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <p className="text-sm whitespace-pre-wrap break-words leading-relaxed">
                        {message.content}
                      </p>
                    )}
                    {message.timestamp && (
                      <p
                        className={`mt-2 text-xs ${message.role === "user" ? "text-white/70" : "text-slate-500"}`}
                      >
                        {new Date(message.timestamp).toLocaleTimeString()}
                      </p>
                    )}
                  </div>
                  {message.role === "user" && (
                    <div className="flex-shrink-0 rounded-lg bg-slate-200 p-2 h-fit border border-slate-300">
                      <User className="h-4 w-4 text-slate-700" />
                    </div>
                  )}
                </div>
              ))}

              {isLoading && streamingMessage && (
                <div className="flex gap-3 justify-start">
                  <div className="flex-shrink-0 rounded-lg bg-slate-100 p-2 h-fit border border-slate-200">
                    <Bot className="h-4 w-4 text-slate-700" />
                  </div>
                  <div className="max-w-[85%] rounded-lg px-4 py-3 bg-white border border-slate-200 shadow-sm">
                    <div className="prose prose-sm max-w-none prose-slate">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm, remarkMath]}
                        rehypePlugins={[rehypeKatex, rehypeHighlight]}
                        components={{
                          p: ({ node, ...props }) => (
                            <p
                              className="mb-3 last:mb-0 leading-relaxed text-slate-900"
                              {...props}
                            />
                          ),
                          ul: ({ node, ...props }) => (
                            <ul
                              className="list-disc pl-6 mb-3 space-y-1.5 text-slate-900"
                              {...props}
                            />
                          ),
                          ol: ({ node, ...props }) => (
                            <ol
                              className="list-decimal pl-6 mb-3 space-y-1.5 text-slate-900"
                              {...props}
                            />
                          ),
                          li: ({ node, ...props }) => (
                            <li className="leading-relaxed text-slate-900" {...props} />
                          ),
                          h1: ({ node, ...props }) => (
                            <h1
                              className="text-xl font-bold mb-3 mt-4 first:mt-0 text-slate-900"
                              {...props}
                            />
                          ),
                          h2: ({ node, ...props }) => (
                            <h2
                              className="text-lg font-bold mb-2 mt-3 first:mt-0 text-slate-900"
                              {...props}
                            />
                          ),
                          h3: ({ node, ...props }) => (
                            <h3
                              className="text-base font-semibold mb-2 mt-3 first:mt-0 text-slate-900"
                              {...props}
                            />
                          ),
                          strong: ({ node, ...props }) => (
                            <strong className="font-semibold text-slate-900" {...props} />
                          ),
                          em: ({ node, ...props }) => (
                            <em className="italic text-slate-900" {...props} />
                          ),
                          blockquote: ({ node, ...props }) => (
                            <blockquote
                              className="border-l-4 border-slate-300 pl-4 italic my-3 text-slate-700"
                              {...props}
                            />
                          ),
                          table: ({ node, ...props }) => (
                            <div className="overflow-x-auto my-4 rounded-lg border border-slate-200">
                              <table className="min-w-full divide-y divide-slate-200" {...props} />
                            </div>
                          ),
                          thead: ({ node, ...props }) => (
                            <thead className="bg-slate-50" {...props} />
                          ),
                          th: ({ node, ...props }) => (
                            <th
                              className="px-4 py-2 text-left text-xs font-medium text-slate-700 uppercase tracking-wider"
                              {...props}
                            />
                          ),
                          td: ({ node, ...props }) => (
                            <td
                              className="px-4 py-2 text-sm text-slate-900 border-t border-slate-200"
                              {...props}
                            />
                          ),
                          code: ({ node, inline, className, children, ...props }: any) => {
                            const content = String(children ?? "").replace(/\n+$/, "");
                            if (inline) {
                              return (
                                <code
                                  className="px-1.5 py-0.5 bg-slate-100 text-slate-900 rounded text-sm font-mono border border-slate-200"
                                  {...props}
                                >
                                  {content}
                                </code>
                              );
                            }
                            return (
                              <pre className="my-3 overflow-x-auto rounded-lg bg-slate-50 border border-slate-200">
                                <code
                                  className={`block p-4 text-sm font-mono ${className ?? ""}`}
                                  {...props}
                                >
                                  {content}
                                </code>
                              </pre>
                            );
                          },
                          pre: ({ node, ...props }) => (
                            <pre className="my-3 overflow-x-auto" {...props} />
                          ),
                          a: ({ node, ...props }) => (
                            <a
                              className="text-blue-600 hover:underline"
                              target="_blank"
                              rel="noopener noreferrer"
                              {...props}
                            />
                          ),
                          hr: ({ node, ...props }) => (
                            <hr className="my-4 border-slate-200" {...props} />
                          ),
                        }}
                      >
                        {streamingMessage}
                      </ReactMarkdown>
                    </div>
                    <div className="mt-2 flex items-center gap-1">
                      <div className="h-2 w-2 rounded-full bg-slate-400 animate-pulse" />
                      <div
                        className="h-2 w-2 rounded-full bg-slate-400 animate-pulse"
                        style={{ animationDelay: "150ms" }}
                      />
                      <div
                        className="h-2 w-2 rounded-full bg-slate-400 animate-pulse"
                        style={{ animationDelay: "300ms" }}
                      />
                    </div>
                  </div>
                </div>
              )}

              {isLoading && !streamingMessage && (
                <div className="flex gap-3 justify-start">
                  <div className="flex-shrink-0 rounded-lg bg-slate-100 p-2 h-fit border border-slate-200">
                    <Bot className="h-4 w-4 text-slate-700" />
                  </div>
                  <div className="rounded-lg px-4 py-3 bg-white border border-slate-200 shadow-sm">
                    <div className="flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin text-slate-700" />
                      <p className="text-sm text-slate-700">Thinking...</p>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Input Area */}
            <div className="border-t border-slate-200 bg-white p-4 flex-shrink-0">
              <div className="flex gap-3 items-end">
                <Textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask a question about job fraud detection..."
                  className="flex-1 resize-none bg-white border-slate-300 text-slate-900 placeholder:text-slate-500 focus:border-slate-500 focus:ring-slate-500/20 min-h-[60px] max-h-[200px]"
                  rows={2}
                  disabled={isLoading}
                />
                <Button
                  onClick={handleSend}
                  disabled={!input.trim() || isLoading}
                  className="bg-slate-900 hover:bg-slate-800 text-white h-[60px] px-6"
                >
                  {isLoading ? (
                    <Loader2 className="h-5 w-5 animate-spin" />
                  ) : (
                    <Send className="h-5 w-5" />
                  )}
                </Button>
              </div>
              <p className="mt-2 text-xs text-slate-500">
                Press{" "}
                <kbd className="px-1.5 py-0.5 bg-slate-100 border border-slate-300 rounded text-slate-700">
                  Enter
                </kbd>{" "}
                to send,{" "}
                <kbd className="px-1.5 py-0.5 bg-slate-100 border border-slate-300 rounded text-slate-700">
                  Shift + Enter
                </kbd>{" "}
                for new line
              </p>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
