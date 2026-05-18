'use client';

import { motion } from 'framer-motion';
import {
  Check,
  CheckCheck,
  Brain,
  Move,
  Mic,
} from 'lucide-react';
import type { Message } from '@/types';
import { cn } from '@/lib/cn';
import { format } from 'date-fns';

const EMOTION_ICONS: Record<string, string> = {
  happy: '😊',
  sad: '😢',
  angry: '😠',
  surprise: '😮',
  neutral: '😐',
  fear: '😨',
  disgust: '🤢',
  contempt: '😏',
};

const MODALITY_ICONS: Record<string, React.ReactNode> = {
  gesture: <Move className="w-3 h-3" />,
  speech: <Mic className="w-3 h-3" />,
  text: null,
  emotion: <Brain className="w-3 h-3" />,
  multimodal: <Brain className="w-3 h-3" />,
};

interface MessageBubbleProps {
  message: Message;
  isLast?: boolean;
  showTimestamp?: boolean;
}

export default function MessageBubble({
  message,
  isLast = false,
  showTimestamp = true,
}: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const confidencePercent = (message.confidence * 100).toFixed(0);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className={cn(
        'flex gap-2 max-w-[85%]',
        isUser ? 'ml-auto flex-row-reverse' : 'mr-auto'
      )}
      role="listitem"
      aria-label={`${isUser ? 'Your' : 'AI'} message: ${message.content}`}
    >
      {/* Avatar */}
      <div
        className={cn(
          'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium',
          isUser
            ? 'bg-neuro-500 text-white'
            : 'bg-purple-500/20 text-purple-400 border border-purple-500/20'
        )}
        aria-hidden="true"
      >
        {isUser ? 'U' : 'AI'}
      </div>

      {/* Bubble */}
      <div className="flex flex-col gap-1 min-w-0">
        <div
          className={cn(
            'rounded-2xl px-4 py-2.5 text-sm leading-relaxed',
            isUser
              ? 'bg-neuro-500 text-white rounded-tr-md'
              : 'glass rounded-tl-md'
          )}
        >
          <p className="whitespace-pre-wrap break-words">{message.content}</p>

          {/* Translation */}
          {message.translation && (
            <div className="mt-1.5 pt-1.5 border-t border-white/10">
              <p className="text-xs opacity-80 italic">{message.translation}</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div
          className={cn(
            'flex items-center gap-2 px-1',
            isUser ? 'flex-row-reverse' : 'flex-row'
          )}
        >
          {/* Modality indicator */}
          {message.modality !== 'text' && MODALITY_ICONS[message.modality] && (
            <span
              className={cn(
                'inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium',
                message.modality === 'gesture'
                  ? 'text-emerald-400 bg-emerald-500/10'
                  : message.modality === 'speech'
                  ? 'text-blue-400 bg-blue-500/10'
                  : 'text-neuro-400 bg-neuro-500/10'
              )}
            >
              {MODALITY_ICONS[message.modality]}
            </span>
          )}

          {/* Emotion tag */}
          {message.emotion && (
            <span className="text-[10px]" role="img" aria-label={message.emotion.emotion}>
              {EMOTION_ICONS[message.emotion.emotion] || '😐'}
            </span>
          )}

          {/* Timestamp */}
          {showTimestamp && (
            <span className="text-[10px] text-gray-600">
              {format(new Date(message.timestamp), 'HH:mm')}
            </span>
          )}

          {/* Confidence */}
          <span
            className={cn(
              'text-[10px] font-mono',
              message.confidence > 0.9
                ? 'text-emerald-500'
                : message.confidence > 0.7
                ? 'text-amber-500'
                : 'text-red-500'
            )}
          >
            {confidencePercent}%
          </span>

          {/* Read status */}
          {isUser && isLast && (
            <span className="text-[10px] text-neuro-400">
              <CheckCheck className="w-3 h-3" />
            </span>
          )}
        </div>
      </div>
    </motion.div>
  );
}
