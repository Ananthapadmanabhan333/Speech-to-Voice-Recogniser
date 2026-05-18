'use client';

import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain,
  Lightbulb,
  MessageSquare,
  AlertCircle,
} from 'lucide-react';
import GlassCard from '@/components/ui/GlassCard';
import { useCommunicationStore } from '@/stores/communication-store';
import { cn } from '@/lib/cn';

function IntentDisplay({ intent, confidence }: { intent: string; confidence: number }) {
  return (
    <div className="p-3 rounded-xl bg-neuro-500/10 border border-neuro-500/20">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Lightbulb className="w-4 h-4 text-neuro-400" />
          <span className="text-xs text-gray-400 uppercase tracking-wider">Detected Intent</span>
        </div>
        <span className="text-xs font-mono text-neuro-400">
          {(confidence * 100).toFixed(0)}%
        </span>
      </div>
      <p className="text-lg font-semibold text-white capitalize">{intent.replace(/_/g, ' ')}</p>
    </div>
  );
}

function UrgencyIndicator({ urgency }: { urgency: number }) {
  const level = urgency > 0.7 ? 'high' : urgency > 0.4 ? 'medium' : 'low';
  const colors = {
    high: 'text-red-400 bg-red-500/10 border-red-500/20',
    medium: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
    low: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
  };

  return (
    <div
      className={cn(
        'px-3 py-2 rounded-lg border text-xs font-medium flex items-center gap-2',
        colors[level]
      )}
    >
      <AlertCircle className="w-3.5 h-3.5" />
      <span className="capitalize">{level} Urgency</span>
      <span className="opacity-60 ml-auto font-mono">
        {(urgency * 100).toFixed(0)}%
      </span>
    </div>
  );
}

function SuggestedResponses({
  suggestions,
  onSelect,
}: {
  suggestions: string[];
  onSelect: (text: string) => void;
}) {
  return (
    <div className="space-y-1.5">
      <span className="text-[10px] text-gray-500 uppercase tracking-wider">
        Suggested Responses
      </span>
      <div className="flex flex-wrap gap-2">
        {suggestions.map((suggestion) => (
          <button
            key={suggestion}
            onClick={() => onSelect(suggestion)}
            className="px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-xs text-gray-300 hover:bg-white/10 hover:text-white transition-all hover:border-neuro-500/30"
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
}

function ModalityBadge({ modality }: { modality: string }) {
  const colors: Record<string, string> = {
    gesture: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    speech: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
    text: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
    emotion: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    multimodal: 'bg-neuro-500/10 text-neuro-400 border-neuro-500/20',
  };

  return (
    <span
      className={cn(
        'px-2 py-0.5 rounded text-[10px] font-medium border capitalize',
        colors[modality] || colors.multimodal
      )}
    >
      {modality}
    </span>
  );
}

function CommunicationLog({ messages }: { messages: Array<{ content: string; modality: string; timestamp: number }> }) {
  if (messages.length === 0) return null;

  return (
    <div className="space-y-2">
      <span className="text-xs text-gray-500 uppercase tracking-wider flex items-center gap-2">
        <MessageSquare className="w-3.5 h-3.5" />
        Recent Interpretations
      </span>
      <div className="space-y-1 max-h-32 overflow-y-auto">
        {messages.slice(-5).reverse().map((msg) => (
          <div
            key={msg.timestamp}
            className="flex items-start gap-2 p-2 rounded-lg bg-white/[0.02]"
          >
            <ModalityBadge modality={msg.modality} />
            <p className="text-xs text-gray-400 flex-1 line-clamp-2">{msg.content}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function MultimodalPanel() {
  const {
    multimodal,
    messages,
    sendMessage,
  } = useCommunicationStore();

  const result = multimodal.current;

  return (
    <GlassCard className="p-4">
      <div className="flex items-center gap-2 mb-4">
        <Brain className="w-5 h-5 text-neuro-400" />
        <h2 className="text-sm font-semibold text-white">Multimodal Fusion</h2>
        <span className="text-[10px] text-gray-600 ml-auto">
          Fusing gesture, speech & emotion
        </span>
      </div>

      <AnimatePresence mode="wait">
        {result ? (
          <motion.div
            key={result.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="space-y-3"
          >
            {/* Fusion Confidence */}
            <div className="flex items-center gap-3 flex-wrap">
              <IntentDisplay intent={result.intent} confidence={result.confidence} />
              <UrgencyIndicator urgency={result.urgency} />
            </div>

            {/* Interpretation */}
            <div className="p-3 rounded-xl bg-white/[0.02] border border-white/5">
              <span className="text-[10px] text-gray-500 uppercase tracking-wider">Interpretation</span>
              <p className="text-sm text-gray-300 mt-1">{result.interpretation}</p>
            </div>

            {/* Alternative interpretations */}
            {result.alternativeInterpretations && result.alternativeInterpretations.length > 0 && (
              <details className="text-xs text-gray-500">
                <summary className="cursor-pointer hover:text-gray-400 transition-colors">
                  Alternative interpretations ({result.alternativeInterpretations.length})
                </summary>
                <ul className="mt-1 space-y-1 pl-4">
                  {result.alternativeInterpretations.map((alt, i) => (
                    <li key={i} className="list-disc text-gray-600">{alt}</li>
                  ))}
                </ul>
              </details>
            )}

            {/* Suggested Responses */}
            {result.suggestions.length > 0 && (
              <SuggestedResponses
                suggestions={result.suggestions}
                onSelect={sendMessage}
              />
            )}

            {/* Active Modalities */}
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-[10px] text-gray-500 uppercase tracking-wider">Modalities:</span>
              {result.modalities.map((mod) => (
                <ModalityBadge key={mod} modality={mod} />
              ))}
            </div>
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex flex-col items-center justify-center py-8 text-center"
          >
            <Brain className="w-10 h-10 text-gray-700 mb-3" />
            <p className="text-sm text-gray-500">Waiting for multimodal input...</p>
            <p className="text-xs text-gray-600 mt-1">
              Use gesture, speech, or text to see fused interpretation
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Communication Log */}
      <div className="mt-4 pt-4 border-t border-white/5">
        <CommunicationLog messages={messages} />
      </div>
    </GlassCard>
  );
}
