'use client';

import { useEffect, useRef, useCallback, useState } from 'react';
import { motion } from 'framer-motion';
import { Mic, MicOff } from 'lucide-react';
import { cn } from '@/lib/cn';

interface SpeechWaveformProps {
  isListening: boolean;
  audioLevel: number;
  onToggle: () => void;
  barCount?: number;
  className?: string;
}

function WaveformBar({
  height,
  index,
  isListening,
}: {
  height: number;
  index: number;
  isListening: boolean;
}) {
  return (
    <motion.div
      className={cn(
        'w-1 rounded-full transition-all',
        isListening ? 'bg-neuro-400' : 'bg-white/10'
      )}
      animate={{
        height: isListening ? Math.max(4, height * 100) : 4,
        opacity: isListening ? 1 : 0.3,
      }}
      transition={{
        duration: 0.15,
        delay: index * 0.02,
        ease: 'easeOut',
      }}
    />
  );
}

export default function SpeechWaveform({
  isListening,
  audioLevel,
  onToggle,
  barCount = 40,
  className,
}: SpeechWaveformProps) {
  const [bars, setBars] = useState<number[]>(() =>
    Array.from({ length: barCount }, () => Math.random() * 0.3)
  );
  const animFrameRef = useRef<number>(0);
  const phaseRef = useRef(0);

  const animateBars = useCallback(() => {
    phaseRef.current += 0.03;
    const baseLevel = isListening ? Math.max(0.1, audioLevel) : 0.05;

    setBars(
      Array.from({ length: barCount }, (_, i) => {
        const wave = Math.sin(phaseRef.current + i * 0.3) * 0.3 + 0.5;
        const noise = Math.random() * 0.2;
        return Math.max(0.02, (wave + noise) * baseLevel);
      })
    );

    animFrameRef.current = requestAnimationFrame(animateBars);
  }, [isListening, audioLevel, barCount]);

  useEffect(() => {
    animFrameRef.current = requestAnimationFrame(animateBars);
    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [animateBars]);

  return (
    <div className={cn('flex flex-col gap-3', className)}>
      {/* Waveform Display */}
      <div className="flex items-center justify-center gap-[2px] h-20 px-2">
        {bars.map((height, index) => (
          <WaveformBar
            key={index}
            height={height}
            index={index}
            isListening={isListening}
          />
        ))}
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-3">
        <button
          onClick={onToggle}
          className={cn(
            'relative flex items-center justify-center w-14 h-14 rounded-full transition-all duration-300',
            isListening
              ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30 shadow-lg shadow-red-500/20'
              : 'bg-neuro-500/20 text-neuro-400 hover:bg-neuro-500/30 shadow-lg shadow-neuro-500/20'
          )}
          aria-label={isListening ? 'Stop recording' : 'Start recording'}
          aria-pressed={isListening}
        >
          {isListening ? (
            <motion.div
              animate={{ scale: [1, 1.1, 1] }}
              transition={{ duration: 1.5, repeat: Infinity }}
            >
              <MicOff className="w-6 h-6" />
            </motion.div>
          ) : (
            <Mic className="w-6 h-6" />
          )}
          {isListening && (
            <motion.div
              className="absolute inset-0 rounded-full border-2 border-red-400/30"
              animate={{ scale: [1, 1.3, 1], opacity: [0.3, 0, 0.3] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
          )}
        </button>

        {/* Status Text */}
        <div className="text-center">
          <p className="text-xs font-medium text-gray-400">
            {isListening ? 'Listening...' : 'Click to speak'}
          </p>
          {isListening && (
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-[10px] text-gray-600 mt-0.5"
            >
              Audio level: {Math.round(audioLevel * 100)}%
            </motion.p>
          )}
        </div>
      </div>
    </div>
  );
}
