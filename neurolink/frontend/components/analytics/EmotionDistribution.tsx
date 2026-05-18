'use client';

import { useState } from 'react';
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { motion } from 'framer-motion';
import GlassCard from '@/components/ui/GlassCard';
import { cn } from '@/lib/cn';

const EMOTION_COLORS: Record<string, string> = {
  happy: '#22c55e',
  sad: '#3b82f6',
  angry: '#ef4444',
  surprise: '#f59e0b',
  neutral: '#8b5cf6',
  fear: '#ec4899',
  disgust: '#14b8a6',
  contempt: '#f97316',
};

interface EmotionEntry {
  emotion: string;
  count: number;
  percentage: number;
}

interface EmotionDistributionProps {
  data: EmotionEntry[];
  className?: string;
  innerRadius?: number;
  outerRadius?: number;
  height?: number;
}

function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const entry = payload[0].payload;
  return (
    <div className="glass-strong p-3 rounded-xl text-sm min-w-[120px]">
      <div className="flex items-center gap-2 mb-1">
        <div
          className="w-3 h-3 rounded-full"
          style={{ backgroundColor: EMOTION_COLORS[entry.emotion] || '#6366f1' }}
        />
        <span className="text-white font-medium capitalize">{entry.emotion}</span>
      </div>
      <p className="text-gray-400 text-xs">
        {entry.count.toLocaleString()} occurrences
      </p>
      <p className="text-gray-400 text-xs">
        {entry.percentage.toFixed(1)}% of total
      </p>
    </div>
  );
}

function CustomLegend({ payload }: any) {
  if (!payload) return null;
  return (
    <div className="flex flex-wrap justify-center gap-3 mt-4">
      {payload.map((entry: any) => (
        <div key={entry.value} className="flex items-center gap-1.5">
          <div
            className="w-2.5 h-2.5 rounded-full"
            style={{ backgroundColor: entry.color }}
          />
          <span className="text-xs text-gray-400 capitalize">{entry.value}</span>
        </div>
      ))}
    </div>
  );
}

export default function EmotionDistribution({
  data,
  className,
  innerRadius = 60,
  outerRadius = 100,
  height = 320,
}: EmotionDistributionProps) {
  const [activeIndex, setActiveIndex] = useState<number | null>(null);

  if (!data || data.length === 0) {
    return (
      <GlassCard className={cn('p-6 flex items-center justify-center', className)}>
        <p className="text-sm text-gray-500">No emotion data available</p>
      </GlassCard>
    );
  }

  const onPieEnter = (_: any, index: number) => setActiveIndex(index);
  const onPieLeave = () => setActiveIndex(null);

  const total = data.reduce((sum, entry) => sum + entry.count, 0);

  return (
    <GlassCard className={cn('p-6', className)}>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-white">Emotion Distribution</h3>
          <p className="text-sm text-gray-500">
            {total.toLocaleString()} total detections
          </p>
        </div>
      </div>

      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={innerRadius}
              outerRadius={outerRadius}
              paddingAngle={3}
              dataKey="count"
              nameKey="emotion"
              animationDuration={1500}
              animationBegin={200}
              onMouseEnter={onPieEnter}
              onMouseLeave={onPieLeave}
            >
              {data.map((entry) => (
                <Cell
                  key={entry.emotion}
                  fill={EMOTION_COLORS[entry.emotion] || '#6366f1'}
                  opacity={
                    activeIndex !== null
                      ? data[activeIndex].emotion === entry.emotion
                        ? 1
                        : 0.5
                      : 1
                  }
                  stroke={
                    activeIndex !== null &&
                    data[activeIndex].emotion === entry.emotion
                      ? EMOTION_COLORS[entry.emotion] || '#6366f1'
                      : 'transparent'
                  }
                  strokeWidth={2}
                />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
            <Legend content={<CustomLegend />} />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Top emotions list */}
      <div className="mt-4 space-y-1.5">
        {data
          .sort((a, b) => b.count - a.count)
          .slice(0, 3)
          .map((entry, i) => (
            <motion.div
              key={entry.emotion}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.1 }}
              className="flex items-center justify-between p-2 rounded-lg bg-white/[0.02]"
            >
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-600 w-4">{i + 1}</span>
                <div
                  className="w-2 h-2 rounded-full"
                  style={{
                    backgroundColor: EMOTION_COLORS[entry.emotion] || '#6366f1',
                  }}
                />
                <span className="text-sm text-gray-300 capitalize">{entry.emotion}</span>
              </div>
              <span className="text-sm text-white font-medium">
                {entry.percentage.toFixed(1)}%
              </span>
            </motion.div>
          ))}
      </div>
    </GlassCard>
  );
}
