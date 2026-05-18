'use client';

import { useEffect, useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Mic,
  MicOff,
  Move,
  StopCircle,
  Send,
  History,
  Lightbulb,
  Languages,
  Brain,
  X,
  ChevronRight,
  Loader2,
} from 'lucide-react';
import GlassCard from '@/components/ui/GlassCard';
import GestureVisualizer from '@/components/gesture/GestureVisualizer';
import SpeechWaveform from '@/components/speech/SpeechWaveform';
import EmotionIndicator from '@/components/emotion/EmotionIndicator';
import MultimodalPanel from '@/components/multimodal/MultimodalPanel';
import CommunicationFeed from '@/components/chat/CommunicationFeed';
import { useCommunicationStore } from '@/stores/communication-store';
import { cn } from '@/lib/cn';
import type { Gesture, EmotionResult } from '@/types';

const suggestedPhrases = [
  'Hello, how are you?',
  'I need help with this.',
  'Thank you very much!',
  'Nice to meet you.',
  'Can you repeat that?',
  'I understand now.',
];

function GestureInputArea() {
  const { gesture, setGestureTracking } = useCommunicationStore();
  const [isWebcamActive, setIsWebcamActive] = useState(false);

  return (
    <GlassCard className="p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Move className="w-4 h-4 text-neuro-400" />
          <h3 className="text-sm font-semibold text-white">Gesture Input</h3>
        </div>
        <button
          onClick={() => {
            const next = !isWebcamActive;
            setIsWebcamActive(next);
            setGestureTracking(next);
          }}
          className={cn(
            'px-3 py-1.5 rounded-lg text-xs font-medium transition-all',
            isWebcamActive
              ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
              : 'bg-white/5 text-gray-400 border border-white/10 hover:bg-white/10'
          )}
          aria-label={isWebcamActive ? 'Disable webcam' : 'Enable webcam'}
          aria-pressed={isWebcamActive}
        >
          {isWebcamActive ? 'Tracking Active' : 'Start Tracking'}
        </button>
      </div>

      <div
        className={cn(
          'relative rounded-xl overflow-hidden transition-all duration-300',
          isWebcamActive ? 'h-48 bg-gray-900' : 'h-32 bg-gray-900/50'
        )}
      >
        {isWebcamActive ? (
          <GestureVisualizer gesture={gesture.current} />
        ) : (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <Move className="w-8 h-8 text-gray-600 mx-auto mb-2" />
              <p className="text-xs text-gray-600">Enable camera to start gesture tracking</p>
            </div>
          </div>
        )}
      </div>

      {gesture.current && (
        <motion.div
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-2 flex items-center justify-between"
        >
          <span className="text-xs text-neuro-400 font-medium capitalize">
            {gesture.current.type.replace('_', ' ')}
          </span>
          <span className="text-xs text-gray-500">
            {(gesture.current.confidence * 100).toFixed(0)}% confidence
          </span>
        </motion.div>
      )}
    </GlassCard>
  );
}

function SpeechInputArea() {
  const { speech, setListening, setAudioLevel } = useCommunicationStore();

  return (
    <GlassCard className="p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Mic className="w-4 h-4 text-neuro-400" />
          <h3 className="text-sm font-semibold text-white">Speech Input</h3>
        </div>
        <span className="text-xs text-gray-500">
          {speech.isListening ? 'Listening...' : 'Click to speak'}
        </span>
      </div>

      <SpeechWaveform
        isListening={speech.isListening}
        audioLevel={speech.audioLevel}
        onToggle={() => setListening(!speech.isListening)}
      />

      <AnimatePresence>
        {speech.transcript && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-2"
          >
            <p className="text-sm text-gray-300 bg-white/5 rounded-lg p-3">
              {speech.transcript}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </GlassCard>
  );
}

function TextInputArea() {
  const { currentInput, setCurrentInput, sendMessage, isProcessing } =
    useCommunicationStore();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (currentInput.trim()) {
      sendMessage(currentInput);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="relative">
      <div className="flex items-end gap-2">
        <div className="flex-1 relative">
          <textarea
            value={currentInput}
            onChange={(e) => setCurrentInput(e.target.value)}
            placeholder="Type your message..."
            rows={1}
            className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white placeholder-gray-500 text-sm resize-none focus:outline-none focus:border-neuro-500/50 focus:ring-1 focus:ring-neuro-500/20 transition-all"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
            aria-label="Message input"
          />
        </div>
        <button
          type="submit"
          disabled={!currentInput.trim() || isProcessing}
          className={cn(
            'p-3 rounded-xl transition-all',
            currentInput.trim() && !isProcessing
              ? 'bg-neuro-500 text-white hover:bg-neuro-600 shadow-lg shadow-neuro-500/20'
              : 'bg-white/5 text-gray-500 cursor-not-allowed'
          )}
          aria-label="Send message"
        >
          {isProcessing ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <Send className="w-5 h-5" />
          )}
        </button>
      </div>
    </form>
  );
}

function TranslationDisplay() {
  return (
    <GlassCard className="p-4">
      <div className="flex items-center gap-2 mb-3">
        <Languages className="w-4 h-4 text-neuro-400" />
        <h3 className="text-sm font-semibold text-white">Translation</h3>
      </div>
      <div className="space-y-2">
        <div className="flex items-center justify-between p-2 rounded-lg bg-white/5">
          <span className="text-sm text-gray-400">English</span>
          <ChevronRight className="w-4 h-4 text-gray-600" />
          <span className="text-sm text-gray-400">Spanish</span>
        </div>
        <p className="text-sm text-gray-500 italic">Waiting for input...</p>
      </div>
    </GlassCard>
  );
}

function SuggestedPhrases() {
  const { sendMessage } = useCommunicationStore();

  return (
    <GlassCard className="p-4">
      <div className="flex items-center gap-2 mb-3">
        <Lightbulb className="w-4 h-4 text-amber-400" />
        <h3 className="text-sm font-semibold text-white">Suggested Phrases</h3>
      </div>
      <div className="flex flex-wrap gap-2">
        {suggestedPhrases.map((phrase) => (
          <button
            key={phrase}
            onClick={() => sendMessage(phrase)}
            className="px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-xs text-gray-300 hover:bg-white/10 hover:text-white transition-all"
          >
            {phrase}
          </button>
        ))}
      </div>
    </GlassCard>
  );
}

function EmotionBubble() {
  const { emotion } = useCommunicationStore();

  return (
    <GlassCard className="p-4">
      <div className="flex items-center gap-2 mb-3">
        <Brain className="w-4 h-4 text-neuro-400" />
        <h3 className="text-sm font-semibold text-white">Emotion State</h3>
      </div>
      <EmotionIndicator
        emotion={emotion.current}
        size="md"
        showDetails
      />
      {!emotion.current && (
        <p className="text-sm text-gray-500 italic mt-2">No emotion data detected</p>
      )}
    </GlassCard>
  );
}

export default function CommunicationPage() {
  const [showHistory, setShowHistory] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return (
    <main className="relative min-h-screen bg-[hsl(var(--background))]">
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white">Communication</h1>
            <p className="text-gray-400 mt-1">Multi-modal interaction interface</p>
          </div>
          <button
            onClick={() => setShowHistory(!showHistory)}
            className={cn(
              'flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all',
              showHistory
                ? 'bg-neuro-500/20 text-neuro-400 border border-neuro-500/30'
                : 'bg-white/5 text-gray-400 border border-white/10 hover:bg-white/10'
            )}
            aria-label="Toggle communication history"
            aria-pressed={showHistory}
          >
            <History className="w-4 h-4" />
            History
          </button>
        </div>

        <div className="flex gap-6">
          {/* Main Content */}
          <div className="flex-1 space-y-6 min-w-0">
            {/* Input Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <GestureInputArea />
              <SpeechInputArea />
            </div>

            {/* Multimodal Panel */}
            <MultimodalPanel />

            {/* Communication Feed */}
            <GlassCard className="p-4">
              <CommunicationFeed />
            </GlassCard>

            {/* Text Input */}
            <TextInputArea />
          </div>

          {/* Right Sidebar */}
          <AnimatePresence>
            {showHistory && (
              <motion.aside
                initial={{ opacity: 0, x: 100, width: 0 }}
                animate={{ opacity: 1, x: 0, width: 320 }}
                exit={{ opacity: 0, x: 100, width: 0 }}
                transition={{ duration: 0.3, ease: 'easeInOut' }}
                className="flex-shrink-0 overflow-hidden"
              >
                <div className="w-80 space-y-4">
                  <SuggestedPhrases />
                  <EmotionBubble />
                  <TranslationDisplay />
                </div>
              </motion.aside>
            )}
          </AnimatePresence>

          {!showHistory && (
            <aside className="hidden lg:block w-80 flex-shrink-0 space-y-4">
              <SuggestedPhrases />
              <EmotionBubble />
              <TranslationDisplay />
            </aside>
          )}
        </div>
      </div>
    </main>
  );
}
