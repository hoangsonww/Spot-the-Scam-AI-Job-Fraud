"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import ChatAssistant from "@/components/chat-assistant";
import { ChatContext } from "@/lib/api";

function ChatPageContent() {
  const searchParams = useSearchParams();
  const [context, setContext] = useState<ChatContext | null>(null);

  useEffect(() => {
    const contextParam = searchParams.get("context");
    if (contextParam) {
      try {
        const parsed = JSON.parse(decodeURIComponent(contextParam));
        setContext(parsed);
      } catch (e) {
        console.error("Failed to parse context:", e);
      }
    } else {
      const saved = localStorage.getItem("spot-scam-chat-context");
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          setContext(parsed);
          localStorage.removeItem("spot-scam-chat-context");
        } catch (e) {
          console.error("Failed to parse saved context:", e);
        }
      }
    }
  }, [searchParams]);

  return <ChatAssistant initialContext={context} />;
}

export default function ChatPage() {
  return (
    <div className="h-screen overflow-hidden">
      <Suspense
        fallback={
          <div className="flex items-center justify-center h-screen bg-white">
            <div className="text-slate-600">Loading...</div>
          </div>
        }
      >
        <ChatPageContent />
      </Suspense>
    </div>
  );
}
