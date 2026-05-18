'use client';

import { useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import GlassCard from '@/components/ui/GlassCard';
import { cn } from '@/lib/cn';

interface AccuracyDataPoint {
  date: string;
  gesture: number;
  speech: number;
  emotion: number;
  multimodal: number;
}

interface AccuracyChartProps {
  data: AccuracyDataPoint[];
  className?: string;
  showLegend?: boolean;
  height?: number;
}

const SERIES = [
  { key: 'gesture', name: 'Gesture', color: '#6366f1' },
  { key: 'speech', name: 'Speech', color: '#34d399' },
  { key: 'emotion', name: 'Emotion', color: '#f59e0b' },
  { key: 'multimodal', name: 'Multimodal', color: '#a78bfa' },
] as const;

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload) return null;

  return (
    <div className="glass-strong p-3 rounded-xl text-sm min-w-[140px]">
      <p className="text-gray-400 text-xs mb-2">{label}</p>
      <div className="space-y-1">
        {payload.map((entry: any) => (
          <div key={entry.name} className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <div
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: entry.color }}
              />
              <span className="text-gray-300 text-xs">{entry.name}</span>
            </div>
            <span className="text-white font-medium text-xs">
              {entry.value.toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function AccuracyChart({
  data,
  className,
  showLegend = true,
  height = 300,
}: AccuracyChartProps) {
  const [hiddenSeries, setHiddenSeries] = useState<Set<string>>(new Set());

  const toggleSeries = (key: string) => {
    setHiddenSeries((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  if (!data || data.length === 0) {
    return (
      <GlassCard className={cn('p-6 flex items-center justify-center', className)}>
        <p className="text-sm text-gray-500">No accuracy data available</p>
      </GlassCard>
    );
  }

  return (
    <GlassCard className={cn('p-6', className)}>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-white">Accuracy Over Time</h3>
          <p className="text-sm text-gray-500">Recognition accuracy trends across modalities</p>
        </div>
        {showLegend && (
          <div className="flex items-center gap-3">
            {SERIES.map((series) => (
              <button
                key={series.key}
                onClick={() => toggleSeries(series.key)}
                className={cn(
                  'flex items-center gap-1.5 text-xs transition-all',
                  hiddenSeries.has(series.key)
                    ? 'text-gray-600 line-through'
                    : 'text-gray-400 hover:text-gray-300'
                )}
                aria-label={`Toggle ${series.name} series`}
                aria-pressed={!hiddenSeries.has(series.key)}
              >
                <div
                  className="w-2.5 h-2.5 rounded-full"
                  style={{
                    backgroundColor: series.color,
                    opacity: hiddenSeries.has(series.key) ? 0.3 : 1,
                  }}
                />
                {series.name}
              </button>
            ))}
          </div>
        )}
      </div>

      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="hsl(0, 0%, 12%)"
              vertical={false}
            />
            <XAxis
              dataKey="date"
              stroke="hsl(0, 0%, 40%)"
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              domain={[50, 100]}
              stroke="hsl(0, 0%, 40%)"
              fontSize={12}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v) => `${v}%`}
            />
            <Tooltip content={<CustomTooltip />} />

            {SERIES.map((series) => (
              <Line
                key={series.key}
                type="monotone"
                dataKey={series.key}
                name={series.name}
                stroke={series.color}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, fill: series.color, strokeWidth: 0 }}
                animationDuration={1500}
                hide={hiddenSeries.has(series.key)}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </GlassCard>
  );
}
