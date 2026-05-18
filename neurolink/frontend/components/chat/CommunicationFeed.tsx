'use client';

import { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageCircle, Loader2 } from 'lucide-react';
import { useCommunicationStore } from '@/stores/communication-store';
import MessageBubble from './MessageBubble';
import { cn } from '@/lib/cn';

function TypingIndicator() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="flex items-center gap-2 max-w-[80%] mr-auto"
    >
      <div className="w-8 h-8 rounded-full bg-purple-500/20 text-purple-400 border border-purple-500/20 flex items-center justify-center text-xs font-medium flex-shrink-0">
        AI
      </div>
      <div className="glass rounded-2xl rounded-tl-md px-4 py-3">
        <div className="flex items-center gap-2">
          <Loader2 className="w-3.5 h-3.5 text-neuro-400 animate-spin" />
          <span className="text-xs text-gray-400">Processing...</span>
        </div>
      </div>
    </motion.div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full py-12 text-center">
      <div className="w-16 h-16 rounded-2xl bg-neuro-500/10 flex items-center justify-center mb-4">
        <MessageCircle className="w-8 h-8 text-neuro-400" />
      </div>
      <h3 className="text-lg font-semibold text-white mb-1">No Messages Yet</h3>
      <p className="text-sm text-gray-500 max-w-xs">
        Start a conversation using gesture, speech, or text input below.
      </p>
    </div>
  );
}

function SessionHeader({ messageCount }: { messageCount: number }) {
  return (
    <div className="flex items-center justify-between mb-4 px-1">
      <div className="flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
        <span className="text-xs text-emerald-400 font-medium">Active Session</span>
      </div>
      <span className="text-xs text-gray-600">{messageCount} messages</span>
    </div>
  );
}

export default function CommunicationFeed() {
  const { messages, isProcessing } = useCommunicationStore();
  const feedEndRef = useRef<HTMLDivElement>(null);
  const feedContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    const container = feedContainerRef.current;
    if (container) {
      const isNearBottom =
        container.scrollHeight - container.scrollTop - container.clientHeight < 100;

      if (isNearBottom) {
        feedEndRef.current?.scrollIntoView({ behavior: 'smooth' });
      }
    }
  }, [messages.length, isProcessing]);

  // Scroll to bottom on new message
  useEffect(() => {
    feedEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages.length]);

  return (
    <div
      className="flex flex-col"
      role="log"
      aria-label="Communication feed"
      aria-live="polite"
    >
      {/* Header */}
      {messages.length > 0 && <SessionHeader messageCount={messages.length} />}

      {/* Messages */}
      <div
        ref={feedContainerRef}
        className={cn(
          'flex-1 overflow-y-auto space-y-3',
          messages.length > 0 ? 'min-h-[300px] max-h-[500px]' : 'min-h-[300px]'
        )}
      >
        <AnimatePresence mode="popLayout">
          {messages.length === 0 ? (
            <EmptyState />
          ) : (
            <div className="space-y-3 px-1 py-2">
              {messages.map((message, index) => (
                <MessageBubble
                  key={message.id}
                  message={message}
                  isLast={index === messages.length - 1}
                  showTimestamp={
                    index === messages.length - 1 ||
                    (index > 0 &&
                      message.timestamp - messages[index - 1].timestamp > 60000)
                  }
                />
              ))}

              {/* Typing Indicator */}
              <AnimatePresence>
                {isProcessing && <TypingIndicator />}
              </AnimatePresence>
            </div>
          )}
        </AnimatePresence>

        <div ref={feedEndRef} />
      </div>
    </div>
  );
}
