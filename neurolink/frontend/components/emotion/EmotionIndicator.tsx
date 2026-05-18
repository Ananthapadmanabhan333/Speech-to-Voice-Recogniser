'use client';

import { motion, AnimatePresence } from 'framer-motion';
import type { EmotionResult } from '@/types';
import { cn } from '@/lib/cn';

const EMOTION_CONFIG: Record<
  string,
  { icon: string; color: string; bg: string; label: string }
> = {
  happy: {
    icon: '😊',
    color: 'text-emerald-400',
    bg: 'bg-emerald-500/10 border-emerald-500/20',
    label: 'Happy',
  },
  sad: {
    icon: '😢',
    color: 'text-blue-400',
    bg: 'bg-blue-500/10 border-blue-500/20',
    label: 'Sad',
  },
  angry: {
    icon: '😠',
    color: 'text-red-400',
    bg: 'bg-red-500/10 border-red-500/20',
    label: 'Angry',
  },
  surprise: {
    icon: '😮',
    color: 'text-amber-400',
    bg: 'bg-amber-500/10 border-amber-500/20',
    label: 'Surprise',
  },
  neutral: {
    icon: '😐',
    color: 'text-purple-400',
    bg: 'bg-purple-500/10 border-purple-500/20',
    label: 'Neutral',
  },
  fear: {
    icon: '😨',
    color: 'text-pink-400',
    bg: 'bg-pink-500/10 border-pink-500/20',
    label: 'Fear',
  },
  disgust: {
    icon: '🤢',
    color: 'text-teal-400',
    bg: 'bg-teal-500/10 border-teal-500/20',
    label: 'Disgust',
  },
  contempt: {
    icon: '😏',
    color: 'text-orange-400',
    bg: 'bg-orange-500/10 border-orange-500/20',
    label: 'Contempt',
  },
};

function ArousalValenceDisplay({
  arousal,
  valence,
}: {
  arousal: number;
  valence: number;
}) {
  const x = ((valence + 1) / 2) * 100;
  const y = ((1 - arousal) / 2) * 100;

  return (
    <div className="relative w-full h-20 rounded-lg bg-white/5 overflow-hidden">
      {/* Grid lines */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="w-px h-full bg-white/5 absolute left-1/2" />
        <div className="h-px w-full bg-white/5 absolute top-1/2" />
      </div>

      {/* Quadrant labels */}
      <span className="absolute top-1 left-2 text-[8px] text-gray-600">High Arousal</span>
      <span className="absolute bottom-1 left-2 text-[8px] text-gray-600">Low Arousal</span>
      <span className="absolute bottom-1 right-2 text-[8px] text-gray-600">High Valence</span>
      <span className="absolute top-1 right-2 text-[8px] text-gray-600">Low Valence</span>

      {/* Point */}
      <motion.div
        className="absolute w-3 h-3 rounded-full bg-neuro-400 shadow-lg shadow-neuro-500/40"
        animate={{ left: `${x}%`, top: `${y}%` }}
        transition={{ type: 'spring', stiffness: 100, damping: 20 }}
        style={{ transform: 'translate(-50%, -50%)' }}
      />
    </div>
  );
}

interface EmotionIndicatorProps {
  emotion: EmotionResult | null;
  size?: 'sm' | 'md' | 'lg';
  showDetails?: boolean;
  className?: string;
}

export default function EmotionIndicator({
  emotion,
  size = 'md',
  showDetails = false,
  className,
}: EmotionIndicatorProps) {
  if (!emotion) {
    return (
      <div
        className={cn(
          'flex items-center justify-center',
          size === 'sm' ? 'gap-2' : 'gap-3',
          className
        )}
      >
        <span className="text-lg" role="img" aria-label="neutral">
          😐
        </span>
        <span className="text-sm text-gray-500">No emotion data</span>
      </div>
    );
  }

  const config = EMOTION_CONFIG[emotion.emotion] || EMOTION_CONFIG.neutral;
  const confidencePercent = (emotion.confidence * 100).toFixed(0);

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={emotion.emotion}
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        transition={{ duration: 0.3 }}
        className={cn(
          'rounded-xl border',
          config.bg,
          size === 'sm' ? 'p-2' : 'p-3',
          className
        )}
        role="status"
        aria-label={`Detected emotion: ${config.label} with ${confidencePercent}% confidence`}
      >
        <div className="flex items-center gap-3">
          <motion.span
            className={cn(
              size === 'sm' ? 'text-xl' : size === 'lg' ? 'text-4xl' : 'text-3xl'
            )}
            animate={{ scale: [1, 1.1, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
            role="img"
            aria-label={config.label}
          >
            {config.icon}
          </motion.span>

          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between">
              <h4
                className={cn(
                  'font-semibold',
                  config.color,
                  size === 'sm' ? 'text-xs' : 'text-sm'
                )}
              >
                {config.label}
              </h4>
              <span
                className={cn(
                  'font-mono',
                  config.color,
                  size === 'sm' ? 'text-[10px]' : 'text-xs'
                )}
              >
                {confidencePercent}%
              </span>
            </div>

            {/* Confidence bar */}
            <div className="h-1.5 rounded-full bg-white/5 mt-1.5 overflow-hidden">
              <motion.div
                className="h-full rounded-full bg-neuro-400"
                initial={{ width: 0 }}
                animate={{ width: `${emotion.confidence * 100}%` }}
                transition={{ duration: 0.5, ease: 'easeOut' }}
              />
            </div>
          </div>
        </div>

        {showDetails && (
          <div className="mt-3 space-y-2">
            {/* Arousal / Valence */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] text-gray-500 uppercase tracking-wider">
                  Arousal / Valence Map
                </span>
              </div>
              <ArousalValenceDisplay
                arousal={emotion.arousal}
                valence={emotion.valence}
              />
            </div>

            {/* Component breakdown */}
            <div className="grid grid-cols-2 gap-2">
              <div className="p-2 rounded-lg bg-white/5">
                <span className="text-[10px] text-gray-500 uppercase">Facial</span>
                <p className={cn('text-xs font-medium', config.color)}>
                  {EMOTION_CONFIG[emotion.facial.emotion]?.label || 'N/A'}
                </p>
                <span className="text-[10px] text-gray-600">
                  {(emotion.facial.confidence * 100).toFixed(0)}%
                </span>
              </div>
              <div className="p-2 rounded-lg bg-white/5">
                <span className="text-[10px] text-gray-500 uppercase">Vocal</span>
                <p className={cn('text-xs font-medium', config.color)}>
                  {EMOTION_CONFIG[emotion.vocal.emotion]?.label || 'N/A'}
                </p>
                <span className="text-[10px] text-gray-600">
                  {(emotion.vocal.confidence * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>
        )}
      </motion.div>
    </AnimatePresence>
  );
}
