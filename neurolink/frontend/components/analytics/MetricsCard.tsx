'use client';

import { useRef, useCallback, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import {
  TrendingUp,
  TrendingDown,
  Minus,
} from 'lucide-react';
import GlassCard from '@/components/ui/GlassCard';
import { cn } from '@/lib/cn';

interface MetricsCardProps {
  icon: React.ReactNode;
  label: string;
  value: number;
  suffix?: string;
  trend?: 'up' | 'down' | 'stable';
  trendValue?: string;
  color?: 'neuro' | 'emerald' | 'amber' | 'rose' | 'blue';
  className?: string;
  animationDuration?: number;
}

const colorClasses: Record<string, { bg: string; text: string; border: string; glow: string }> = {
  neuro: {
    bg: 'bg-neuro-500/10',
    text: 'text-neuro-400',
    border: 'border-neuro-500/20',
    glow: 'shadow-neuro-500/20',
  },
  emerald: {
    bg: 'bg-emerald-500/10',
    text: 'text-emerald-400',
    border: 'border-emerald-500/20',
    glow: 'shadow-emerald-500/20',
  },
  amber: {
    bg: 'bg-amber-500/10',
    text: 'text-amber-400',
    border: 'border-amber-500/20',
    glow: 'shadow-amber-500/20',
  },
  rose: {
    bg: 'bg-rose-500/10',
    text: 'text-rose-400',
    border: 'border-rose-500/20',
    glow: 'shadow-rose-500/20',
  },
  blue: {
    bg: 'bg-blue-500/10',
    text: 'text-blue-400',
    border: 'border-blue-500/20',
    glow: 'shadow-blue-500/20',
  },
};

function AnimatedNumber({
  value,
  suffix = '',
  duration = 1500,
  decimals = 0,
}: {
  value: number;
  suffix?: string;
  duration?: number;
  decimals?: number;
}) {
  const [display, setDisplay] = useState(0);
  const startTime = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);

  const animate = useCallback(
    (timestamp: number) => {
      if (!startTime.current) startTime.current = timestamp;
      const progress = Math.min((timestamp - startTime.current) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(eased * value);
      if (progress < 1) {
        rafRef.current = requestAnimationFrame(animate);
      }
    },
    [value, duration]
  );

  useEffect(() => {
    startTime.current = null;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(animate);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [animate]);

  return (
    <span>
      {display.toFixed(decimals)}
      {suffix}
    </span>
  );
}

function TrendIndicator({
  trend,
  trendValue,
}: {
  trend: 'up' | 'down' | 'stable';
  trendValue?: string;
}) {
  if (!trendValue) return null;

  return (
    <div
      className={cn(
        'flex items-center gap-1 text-xs font-medium',
        trend === 'up' && 'text-emerald-400',
        trend === 'down' && 'text-rose-400',
        trend === 'stable' && 'text-gray-400'
      )}
    >
      {trend === 'up' && <TrendingUp className="w-3 h-3" />}
      {trend === 'down' && <TrendingDown className="w-3 h-3" />}
      {trend === 'stable' && <Minus className="w-3 h-3" />}
      <span>{trendValue}</span>
    </div>
  );
}

function Sparkline({ value, color }: { value: number; color: string }) {
  const data = useRef(
    Array.from({ length: 20 }, () => Math.random() * value * 0.5 + value * 0.5)
  );

  const max = Math.max(...data.current, value);
  const height = 32;
  const width = 60;
  const points = data.current
    .concat(value)
    .map((v, i) => {
      const x = (i / 20) * width;
      const y = height - (v / max) * height;
      return `${x},${y}`;
    })
    .join(' ');

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className="flex-shrink-0"
      aria-hidden="true"
    >
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity={0.4}
      />
    </svg>
  );
}

export default function MetricsCard({
  icon,
  label,
  value,
  suffix = '',
  trend,
  trendValue,
  color = 'neuro',
  className,
  animationDuration = 1500,
}: MetricsCardProps) {
  const colors = colorClasses[color];

  return (
    <GlassCard className={cn('p-5', className)} hover glow>
      <div className="flex items-start justify-between mb-3">
        <div
          className={cn(
            'w-10 h-10 rounded-xl flex items-center justify-center border',
            colors.bg,
            colors.text,
            colors.border
          )}
        >
          {icon}
        </div>
        <Sparkline value={value} color={colors.text.replace('text-', '#')} />
      </div>

      <div className="space-y-1">
        <p className="text-xs text-gray-500 uppercase tracking-wider">{label}</p>
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold text-white">
            <AnimatedNumber
              value={value}
              suffix={suffix}
              duration={animationDuration}
              decimals={suffix === '%' ? 1 : 0}
            />
          </span>

          {trend && trendValue && (
            <TrendIndicator trend={trend} trendValue={trendValue} />
          )}
        </div>
      </div>
    </GlassCard>
  );
}
