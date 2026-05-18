'use client';

import { useEffect, useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
  Legend,
} from 'recharts';
import {
  Download,
  Calendar,
  TrendingUp,
  Brain,
  Move,
  Mic,
  Activity,
} from 'lucide-react';
import GlassCard from '@/components/ui/GlassCard';
import NeuroBackground from '@/components/ui/NeuroBackground';
import MetricsCard from '@/components/analytics/MetricsCard';
import { cn } from '@/lib/cn';
import { useAnalyticsStore } from '@/stores/analytics-store';
import toast from 'react-hot-toast';

const emotionColors: Record<string, string> = {
  happy: '#22c55e',
  sad: '#3b82f6',
  angry: '#ef4444',
  surprise: '#f59e0b',
  neutral: '#8b5cf6',
  fear: '#ec4899',
  disgust: '#14b8a6',
  contempt: '#f97316',
};

const gestureAccuracyData = [
  { date: 'Week 1', accuracy: 82, target: 90 },
  { date: 'Week 2', accuracy: 86, target: 90 },
  { date: 'Week 3', accuracy: 88, target: 92 },
  { date: 'Week 4', accuracy: 91, target: 92 },
  { date: 'Week 5', accuracy: 93, target: 94 },
  { date: 'Week 6', accuracy: 95, target: 94 },
  { date: 'Week 7', accuracy: 96, target: 96 },
  { date: 'Week 8', accuracy: 97, target: 96 },
];

const emotionDistribution = [
  { emotion: 'happy', count: 340, percentage: 34 },
  { emotion: 'neutral', count: 280, percentage: 28 },
  { emotion: 'sad', count: 120, percentage: 12 },
  { emotion: 'surprise', count: 100, percentage: 10 },
  { emotion: 'angry', count: 80, percentage: 8 },
  { emotion: 'fear', count: 50, percentage: 5 },
  { emotion: 'disgust', count: 30, percentage: 3 },
];

const efficiencyData = [
  { date: 'Week 1', accuracy: 78, speed: 65, understanding: 72 },
  { date: 'Week 2', accuracy: 82, speed: 70, understanding: 76 },
  { date: 'Week 3', accuracy: 85, speed: 74, understanding: 80 },
  { date: 'Week 4', accuracy: 88, speed: 78, understanding: 83 },
  { date: 'Week 5', accuracy: 91, speed: 82, understanding: 87 },
  { date: 'Week 6', accuracy: 93, speed: 85, understanding: 90 },
  { date: 'Week 7', accuracy: 95, speed: 88, understanding: 92 },
  { date: 'Week 8', accuracy: 97, speed: 91, understanding: 94 },
];

const learningCurveData = [
  { date: 'Week 1', gesture: 72, speech: 68, emotion: 65, multimodal: 60 },
  { date: 'Week 2', gesture: 78, speech: 73, emotion: 70, multimodal: 67 },
  { date: 'Week 3', gesture: 82, speech: 78, emotion: 75, multimodal: 73 },
  { date: 'Week 4', gesture: 86, speech: 82, emotion: 79, multimodal: 78 },
  { date: 'Week 5', gesture: 90, speech: 86, emotion: 83, multimodal: 82 },
  { date: 'Week 6', gesture: 93, speech: 89, emotion: 87, multimodal: 86 },
  { date: 'Week 7', gesture: 95, speech: 92, emotion: 90, multimodal: 90 },
  { date: 'Week 8', gesture: 97, speech: 94, emotion: 92, multimodal: 93 },
];

const adaptationProgressData = [
  { category: 'Gesture Recognition', current: 97, previous: 82, target: 98, unit: '%', trend: 'up' as const },
  { category: 'Speech Accuracy', current: 94, previous: 78, target: 96, unit: '%', trend: 'up' as const },
  { category: 'Emotion Detection', current: 92, previous: 75, target: 95, unit: '%', trend: 'up' as const },
  { category: 'Response Time', current: 12, previous: 28, target: 8, unit: 'ms', trend: 'down' as const },
  { category: 'Personalization', current: 85, previous: 62, target: 100, unit: '%', trend: 'up' as const },
  { category: 'Context Awareness', current: 78, previous: 55, target: 95, unit: '%', trend: 'up' as const },
];

const dateRangeOptions = [
  { label: '7 Days', value: '7d' },
  { label: '30 Days', value: '30d' },
  { label: '90 Days', value: '90d' },
  { label: 'Custom', value: 'custom' },
];

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload) return null;
  return (
    <div className="glass-strong p-3 rounded-xl text-sm">
      <p className="text-gray-400 mb-2">{label}</p>
      {payload.map((entry: any, index: number) => (
        <div key={index} className="flex items-center gap-2">
          <div
            className="w-2 h-2 rounded-full"
            style={{ backgroundColor: entry.color }}
          />
          <span className="text-gray-300">{entry.name}: </span>
          <span className="text-white font-medium">{entry.value}%</span>
        </div>
      ))}
    </div>
  );
}

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.06 },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
};

export default function AnalyticsPage() {
  const [dateRange, setDateRange] = useState('30d');
  const [mounted, setMounted] = useState(false);
  const analyticsStore = useAnalyticsStore();

  useEffect(() => {
    setMounted(true);
    analyticsStore.fetchAllMetrics();
  }, []);

  const handleExport = useCallback(async () => {
    try {
      const blob = await analyticsStore.exportData('csv');
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `neurolink-analytics-${new Date().toISOString().split('T')[0]}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      toast.success('Analytics data exported successfully');
    } catch {
      toast.error('Failed to export analytics data');
    }
  }, [analyticsStore]);

  if (!mounted) return null;

  return (
    <main className="relative min-h-screen bg-[hsl(var(--background))]">
      <NeuroBackground />

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="space-y-8"
        >
          {/* Header */}
          <motion.div
            variants={itemVariants}
            className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4"
          >
            <div>
              <h1 className="text-3xl font-bold text-white">Analytics</h1>
              <p className="text-gray-400 mt-1">Comprehensive system performance and user adaptation metrics</p>
            </div>
            <div className="flex items-center gap-3">
              {/* Date Range Selector */}
              <div className="flex items-center gap-1 p-1 rounded-xl glass">
                {dateRangeOptions.map((option) => (
                  <button
                    key={option.value}
                    onClick={() => setDateRange(option.value)}
                    className={cn(
                      'px-3 py-1.5 rounded-lg text-xs font-medium transition-all',
                      dateRange === option.value
                        ? 'bg-neuro-500 text-white'
                        : 'text-gray-400 hover:text-white hover:bg-white/5'
                    )}
                  >
                    {option.label}
                  </button>
                ))}
              </div>

              {/* Export Button */}
              <button
                onClick={handleExport}
                className="flex items-center gap-2 px-4 py-2 rounded-xl glass text-sm text-gray-300 hover:text-white hover:bg-white/5 transition-all border border-white/10"
                aria-label="Export analytics data"
              >
                <Download className="w-4 h-4" />
                Export
              </button>
            </div>
          </motion.div>

          {/* Summary Metrics */}
          <motion.div
            variants={itemVariants}
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"
          >
            <MetricsCard
              icon={<Move className="w-5 h-5" />}
              label="Gesture Accuracy"
              value={97}
              suffix="%"
              trend="up"
              trendValue="+15%"
              color="neuro"
            />
            <MetricsCard
              icon={<Mic className="w-5 h-5" />}
              label="Speech Accuracy"
              value={94}
              suffix="%"
              trend="up"
              trendValue="+16%"
              color="neuro"
            />
            <MetricsCard
              icon={<Brain className="w-5 h-5" />}
              label="Emotion Detection"
              value={92}
              suffix="%"
              trend="up"
              trendValue="+17%"
              color="neuro"
            />
            <MetricsCard
              icon={<Activity className="w-5 h-5" />}
              label="Overall Efficiency"
              value={94}
              suffix="%"
              trend="up"
              trendValue="+12%"
              color="neuro"
            />
          </motion.div>

          {/* Charts Row 1 */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Gesture Accuracy */}
            <motion.div variants={itemVariants}>
              <GlassCard className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h2 className="text-lg font-semibold text-white">Gesture Recognition Accuracy</h2>
                    <p className="text-sm text-gray-500">Actual vs target performance</p>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="flex items-center gap-1.5">
                      <div className="w-3 h-3 rounded-full bg-neuro-400" />
                      <span className="text-xs text-gray-400">Actual</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <div className="w-3 h-3 rounded-full bg-emerald-400" />
                      <span className="text-xs text-gray-400">Target</span>
                    </div>
                  </div>
                </div>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={gestureAccuracyData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(0, 0%, 12%)" />
                      <XAxis dataKey="date" stroke="hsl(0, 0%, 40%)" fontSize={12} tickLine={false} />
                      <YAxis domain={[70, 100]} stroke="hsl(0, 0%, 40%)" fontSize={12} tickLine={false} />
                      <Tooltip content={<CustomTooltip />} />
                      <Line
                        type="monotone"
                        dataKey="accuracy"
                        name="Actual"
                        stroke="hsl(239, 84%, 67%)"
                        strokeWidth={2}
                        dot={{ r: 4, fill: 'hsl(239, 84%, 67%)' }}
                        animationDuration={1500}
                      />
                      <Line
                        type="monotone"
                        dataKey="target"
                        name="Target"
                        stroke="#34d399"
                        strokeWidth={2}
                        strokeDasharray="5 5"
                        dot={false}
                        animationDuration={1500}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </GlassCard>
            </motion.div>

            {/* Emotion Distribution */}
            <motion.div variants={itemVariants}>
              <GlassCard className="p-6">
                <h2 className="text-lg font-semibold text-white mb-6">Emotion Distribution</h2>
                <div className="h-72 flex items-center">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={emotionDistribution}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        paddingAngle={3}
                        dataKey="count"
                        animationDuration={1500}
                      >
                        {emotionDistribution.map((entry) => (
                          <Cell
                            key={entry.emotion}
                            fill={emotionColors[entry.emotion] || '#6366f1'}
                          />
                        ))}
                      </Pie>
                      <Tooltip
                        content={({ active, payload }) => {
                          if (!active || !payload?.length) return null;
                          const data = payload[0].payload;
                          return (
                            <div className="glass-strong p-3 rounded-xl text-sm">
                              <p className="text-white font-medium capitalize mb-1">{data.emotion}</p>
                              <p className="text-gray-400">{data.count} occurrences</p>
                              <p className="text-gray-400">{data.percentage}% of total</p>
                            </div>
                          );
                        }}
                      />
                      <Legend
                        wrapperStyle={{ fontSize: '11px', color: '#9ca3af' }}
                        formatter={(value: string) => (
                          <span className="text-gray-400 capitalize">{value}</span>
                        )}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </GlassCard>
            </motion.div>
          </div>

          {/* Charts Row 2 */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Communication Efficiency */}
            <motion.div variants={itemVariants}>
              <GlassCard className="p-6">
                <h2 className="text-lg font-semibold text-white mb-6">Communication Efficiency</h2>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={efficiencyData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(0, 0%, 12%)" />
                      <XAxis dataKey="date" stroke="hsl(0, 0%, 40%)" fontSize={12} tickLine={false} />
                      <YAxis domain={[50, 100]} stroke="hsl(0, 0%, 40%)" fontSize={12} tickLine={false} />
                      <Tooltip content={<CustomTooltip />} />
                      <Bar
                        dataKey="accuracy"
                        name="Accuracy"
                        fill="hsl(239, 84%, 67%)"
                        radius={[4, 4, 0, 0]}
                        animationDuration={1500}
                      />
                      <Bar
                        dataKey="speed"
                        name="Speed"
                        fill="#34d399"
                        radius={[4, 4, 0, 0]}
                        animationDuration={1500}
                      />
                      <Bar
                        dataKey="understanding"
                        name="Understanding"
                        fill="#a78bfa"
                        radius={[4, 4, 0, 0]}
                        animationDuration={1500}
                      />
                      <Legend
                        wrapperStyle={{ fontSize: '11px', color: '#9ca3af' }}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </GlassCard>
            </motion.div>

            {/* Learning Curve */}
            <motion.div variants={itemVariants}>
              <GlassCard className="p-6">
                <h2 className="text-lg font-semibold text-white mb-6">Learning Curve</h2>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={learningCurveData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(0, 0%, 12%)" />
                      <XAxis dataKey="date" stroke="hsl(0, 0%, 40%)" fontSize={12} tickLine={false} />
                      <YAxis domain={[50, 100]} stroke="hsl(0, 0%, 40%)" fontSize={12} tickLine={false} />
                      <Tooltip content={<CustomTooltip />} />
                      <Area
                        type="monotone"
                        dataKey="gesture"
                        name="Gesture"
                        stroke="hsl(239, 84%, 67%)"
                        fill="hsl(239, 84%, 67%)"
                        fillOpacity={0.1}
                        strokeWidth={2}
                        animationDuration={1500}
                      />
                      <Area
                        type="monotone"
                        dataKey="speech"
                        name="Speech"
                        stroke="#34d399"
                        fill="#34d399"
                        fillOpacity={0.1}
                        strokeWidth={2}
                        animationDuration={1500}
                      />
                      <Area
                        type="monotone"
                        dataKey="emotion"
                        name="Emotion"
                        stroke="#f59e0b"
                        fill="#f59e0b"
                        fillOpacity={0.1}
                        strokeWidth={2}
                        animationDuration={1500}
                      />
                      <Area
                        type="monotone"
                        dataKey="multimodal"
                        name="Multimodal"
                        stroke="#a78bfa"
                        fill="#a78bfa"
                        fillOpacity={0.1}
                        strokeWidth={2}
                        animationDuration={1500}
                      />
                      <Legend
                        wrapperStyle={{ fontSize: '11px', color: '#9ca3af' }}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </GlassCard>
            </motion.div>
          </div>

          {/* Adaptation Progress */}
          <motion.div variants={itemVariants}>
            <GlassCard className="p-6">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-lg font-semibold text-white">Adaptation Progress</h2>
                  <p className="text-sm text-gray-500">Personalization improvement metrics</p>
                </div>
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded-full bg-neuro-400" />
                    <span className="text-xs text-gray-400">Current</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded-full bg-gray-600" />
                    <span className="text-xs text-gray-400">Previous</span>
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {adaptationProgressData.map((metric) => {
                  const improvement = metric.trend === 'up'
                    ? ((metric.current - metric.previous) / metric.previous) * 100
                    : ((metric.previous - metric.current) / metric.previous) * 100;

                  return (
                    <div key={metric.category} className="p-4 rounded-xl bg-white/[0.02] border border-white/5">
                      <div className="flex items-center justify-between mb-3">
                        <span className="text-sm text-gray-400">{metric.category}</span>
                        <span className={cn(
                          'text-xs font-medium',
                          metric.trend === 'up' ? 'text-emerald-400' : 'text-rose-400'
                        )}>
                          {metric.trend === 'up' ? '+' : '-'}{improvement.toFixed(0)}%
                        </span>
                      </div>
                      <div className="flex items-baseline gap-2 mb-3">
                        <span className="text-2xl font-bold text-white">
                          {metric.current}{metric.unit}
                        </span>
                        <span className="text-sm text-gray-600 line-through">
                          {metric.previous}{metric.unit}
                        </span>
                      </div>
                      <div className="h-2 rounded-full bg-white/5 overflow-hidden">
                        <div className="flex h-full gap-0.5">
                          <motion.div
                            initial={{ width: 0 }}
                            whileInView={{ width: `${(metric.current / metric.target) * 100}%` }}
                            viewport={{ once: true }}
                            transition={{ duration: 1, ease: 'easeOut' }}
                            className="h-full rounded-full bg-gradient-to-r from-neuro-500 to-neuro-400"
                          />
                        </div>
                      </div>
                      <div className="flex items-center justify-between mt-1.5">
                        <span className="text-xs text-gray-600">Target: {metric.target}{metric.unit}</span>
                        <span className="text-xs text-gray-600">
                          {Math.round((metric.current / metric.target) * 100)}%
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </GlassCard>
          </motion.div>
        </motion.div>
      </div>
    </main>
  );
}
