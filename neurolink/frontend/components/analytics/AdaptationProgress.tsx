'use client';

import { motion } from 'framer-motion';
import {
  TrendingUp,
  TrendingDown,
  Target,
  Brain,
  Move,
  Mic,
  Eye,
  Zap,
} from 'lucide-react';
import GlassCard from '@/components/ui/GlassCard';
import { cn } from '@/lib/cn';

interface AdaptationMetric {
  category: string;
  current: number;
  previous: number;
  target: number;
  unit: string;
  trend: 'up' | 'down' | 'stable';
}

interface AdaptationProgressProps {
  data: AdaptationMetric[];
  className?: string;
}

const CATEGORY_ICONS: Record<string, React.ReactNode> = {
  'Gesture Recognition': <Move className="w-4 h-4" />,
  'Speech Accuracy': <Mic className="w-4 h-4" />,
  'Emotion Detection': <Brain className="w-4 h-4" />,
  'Response Time': <Zap className="w-4 h-4" />,
  'Personalization': <Target className="w-4 h-4" />,
  'Context Awareness': <Eye className="w-4 h-4" />,
};

const CATEGORY_COLORS: Record<string, string> = {
  'Gesture Recognition': 'from-neuro-500 to-neuro-400',
  'Speech Accuracy': 'from-emerald-500 to-emerald-400',
  'Emotion Detection': 'from-amber-500 to-amber-400',
  'Response Time': 'from-blue-500 to-blue-400',
  'Personalization': 'from-purple-500 to-purple-400',
  'Context Awareness': 'from-rose-500 to-rose-400',
};

function ProgressBar({
  current,
  target,
  color,
  trend,
  previous,
}: {
  current: number;
  target: number;
  color: string;
  trend: 'up' | 'down' | 'stable';
  previous: number;
}) {
  const progress = Math.min((current / target) * 100, 100);
  const improvement = trend === 'up'
    ? ((current - previous) / previous) * 100
    : ((previous - current) / previous) * 100;

  return (
    <div className="space-y-1.5">
      <div className="relative h-3 rounded-full bg-white/5 overflow-hidden">
        {/* Previous level marker */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-gray-500/50 z-10"
          style={{ left: `${(previous / target) * 100}%` }}
          title={`Previous: ${previous}${'%'}`}
        />

        {/* Current progress */}
        <motion.div
          className={cn('h-full rounded-full bg-gradient-to-r', color)}
          initial={{ width: 0 }}
          whileInView={{ width: `${progress}%` }}
          viewport={{ once: true }}
          transition={{ duration: 1.2, ease: 'easeOut' }}
        />

        {/* Glow effect */}
        <motion.div
          className={cn(
            'absolute top-0 bottom-0 w-8 rounded-full blur-md opacity-50',
            color.replace('from-', 'bg-').split(' ')[0]
          )}
          initial={{ left: '0%' }}
          whileInView={{ left: `${progress}%` }}
          viewport={{ once: true }}
          transition={{ duration: 1.2, ease: 'easeOut' }}
          style={{ transform: 'translateX(-50%)' }}
        />
      </div>

      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-1.5">
          <span className="text-white font-medium">
            {current}{'%'}
          </span>
          <span className="text-gray-600">/ {target}{'%'}</span>

          {improvement > 0 && (
            <span className="text-emerald-500 text-[10px] font-medium flex items-center gap-0.5">
              <TrendingUp className="w-2.5 h-2.5" />
              +{improvement.toFixed(0)}%
            </span>
          )}
          {improvement < 0 && (
            <span className="text-rose-500 text-[10px] font-medium flex items-center gap-0.5">
              <TrendingDown className="w-2.5 h-2.5" />
              {improvement.toFixed(0)}%
            </span>
          )}
        </div>
        <span className="text-gray-600">{progress.toFixed(0)}%</span>
      </div>
    </div>
  );
}

function OverallScore({ data }: { data: AdaptationMetric[] }) {
  const totalCurrent = data.reduce((sum, m) => sum + m.current, 0);
  const totalTarget = data.reduce((sum, m) => sum + m.target, 0);
  const overall = totalTarget > 0 ? (totalCurrent / totalTarget) * 100 : 0;

  return (
    <div className="flex flex-col items-center justify-center p-6 rounded-xl bg-white/[0.02] border border-white/5">
      <div className="relative w-24 h-24 mb-3">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 96 96">
          <circle
            cx="48"
            cy="48"
            r="42"
            fill="none"
            stroke="hsl(0, 0%, 12%)"
            strokeWidth="6"
          />
          <motion.circle
            cx="48"
            cy="48"
            r="42"
            fill="none"
            stroke="url(#scoreGradient)"
            strokeWidth="6"
            strokeLinecap="round"
            strokeDasharray={`${2 * Math.PI * 42}`}
            initial={{ strokeDashoffset: 2 * Math.PI * 42 }}
            whileInView={{
              strokeDashoffset: 2 * Math.PI * 42 * (1 - overall / 100),
            }}
            viewport={{ once: true }}
            transition={{ duration: 1.5, ease: 'easeOut' }}
          />
          <defs>
            <linearGradient id="scoreGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#6366f1" />
              <stop offset="100%" stopColor="#a78bfa" />
            </linearGradient>
          </defs>
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <motion.span
            className="text-2xl font-bold text-white"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.5 }}
          >
            {overall.toFixed(0)}%
          </motion.span>
        </div>
      </div>
      <p className="text-sm text-gray-400 font-medium">Overall Adaptation Score</p>
      <p className="text-xs text-gray-600 mt-1">
        {data.filter((m) => m.current >= m.target).length} of {data.length} metrics at target
      </p>
    </div>
  );
}

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.08 },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 15 },
  visible: { opacity: 1, y: 0 },
};

export default function AdaptationProgress({
  data,
  className,
}: AdaptationProgressProps) {
  if (!data || data.length === 0) {
    return (
      <GlassCard className={cn('p-6 flex items-center justify-center', className)}>
        <p className="text-sm text-gray-500">No adaptation data available</p>
      </GlassCard>
    );
  }

  return (
    <GlassCard className={cn('p-6', className)}>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-white">Adaptation Progress</h3>
          <p className="text-sm text-gray-500">
            Personalization improvement over time
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Progress bars */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="lg:col-span-2 space-y-5"
        >
          {data.map((metric) => (
            <motion.div
              key={metric.category}
              variants={itemVariants}
              className="p-4 rounded-xl bg-white/[0.02] border border-white/5"
            >
              <div className="flex items-center gap-3 mb-3">
                <div
                  className={cn(
                    'w-8 h-8 rounded-lg flex items-center justify-center',
                    'bg-white/5 text-gray-400'
                  )}
                >
                  {CATEGORY_ICONS[metric.category] || <Target className="w-4 h-4" />}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-white">{metric.category}</p>
                  {metric.trend && (
                    <span
                      className={cn(
                        'text-[10px] font-medium flex items-center gap-1',
                        metric.trend === 'up'
                          ? 'text-emerald-500'
                          : metric.trend === 'down'
                          ? 'text-rose-500'
                          : 'text-gray-500'
                      )}
                    >
                      {metric.trend === 'up' && <TrendingUp className="w-2.5 h-2.5" />}
                      {metric.trend === 'down' && <TrendingDown className="w-2.5 h-2.5" />}
                      {metric.trend.charAt(0).toUpperCase() + metric.trend.slice(1)} from previous
                    </span>
                  )}
                </div>
              </div>

              <ProgressBar
                current={metric.current}
                target={metric.target}
                color={CATEGORY_COLORS[metric.category] || 'from-neuro-500 to-neuro-400'}
                trend={metric.trend}
                previous={metric.previous}
              />
            </motion.div>
          ))}
        </motion.div>

        {/* Overall Score */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <OverallScore data={data} />
        </motion.div>
      </div>
    </GlassCard>
  );
}
