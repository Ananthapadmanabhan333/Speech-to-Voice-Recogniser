'use client';

import { useEffect, useState, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import {
  Activity,
  Brain,
  Mic,
  Move,
  MessageCircle,
  Clock,
  TrendingUp,
  Users,
  Zap,
  ArrowUpRight,
  ArrowDownRight,
} from 'lucide-react';
import GlassCard from '@/components/ui/GlassCard';
import NeuroBackground from '@/components/ui/NeuroBackground';
import MetricsCard from '@/components/analytics/MetricsCard';
import { cn } from '@/lib/cn';
import type { DashboardMetrics, CommunicationSession } from '@/types';
import api from '@/lib/api';

const chartData = [
  { day: 'Mon', accuracy: 92, speech: 88, emotion: 85 },
  { day: 'Tue', accuracy: 94, speech: 91, emotion: 87 },
  { day: 'Wed', accuracy: 91, speech: 89, emotion: 86 },
  { day: 'Thu', accuracy: 95, speech: 93, emotion: 90 },
  { day: 'Fri', accuracy: 97, speech: 94, emotion: 92 },
  { day: 'Sat', accuracy: 96, speech: 92, emotion: 91 },
  { day: 'Sun', accuracy: 98, speech: 96, emotion: 94 },
];

const recentCommunications: CommunicationSession[] = [
  {
    id: '1',
    type: 'multimodal',
    messages: [],
    metrics: {
      totalMessages: 12,
      averageConfidence: 0.95,
      duration: 180,
      modalityDistribution: { gesture: 4, speech: 5, text: 3, emotion: 0, multimodal: 0 },
      emotionProgression: [],
      gestureAccuracy: 96,
      speechAccuracy: 94,
    },
    status: 'completed',
    startedAt: Date.now() - 3600000,
    participants: ['user', 'ai'],
  },
  {
    id: '2',
    type: 'gesture',
    messages: [],
    metrics: {
      totalMessages: 8,
      averageConfidence: 0.92,
      duration: 120,
      modalityDistribution: { gesture: 8, speech: 0, text: 0, emotion: 0, multimodal: 0 },
      emotionProgression: [],
      gestureAccuracy: 93,
      speechAccuracy: 0,
    },
    status: 'completed',
    startedAt: Date.now() - 7200000,
    participants: ['user', 'ai'],
  },
  {
    id: '3',
    type: 'speech',
    messages: [],
    metrics: {
      totalMessages: 15,
      averageConfidence: 0.97,
      duration: 240,
      modalityDistribution: { gesture: 0, speech: 15, text: 0, emotion: 0, multimodal: 0 },
      emotionProgression: [],
      gestureAccuracy: 0,
      speechAccuracy: 97,
    },
    status: 'active',
    startedAt: Date.now() - 600000,
    participants: ['user', 'ai'],
  },
  {
    id: '4',
    type: 'text',
    messages: [],
    metrics: {
      totalMessages: 6,
      averageConfidence: 0.99,
      duration: 90,
      modalityDistribution: { gesture: 0, speech: 0, text: 6, emotion: 0, multimodal: 0 },
      emotionProgression: [],
      gestureAccuracy: 0,
      speechAccuracy: 0,
    },
    status: 'completed',
    startedAt: Date.now() - 14400000,
    participants: ['user', 'ai'],
  },
];

const personalizationData = [
  { category: 'Gesture Recognition', current: 85, target: 98 },
  { category: 'Speech Understanding', current: 78, target: 95 },
  { category: 'Emotion Detection', current: 82, target: 94 },
  { category: 'Context Awareness', current: 70, target: 90 },
  { category: 'Response Relevance', current: 88, target: 96 },
];

function AnimatedCounter({
  value,
  suffix = '',
  duration = 2000,
}: {
  value: number;
  suffix?: string;
  duration?: number;
}) {
  const [display, setDisplay] = useState(0);
  const startTime = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);

  const animate = useCallback(
    (timestamp: number) => {
      if (!startTime.current) startTime.current = timestamp;
      const progress = Math.min((timestamp - startTime.current) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(Math.floor(eased * value));
      if (progress < 1) {
        rafRef.current = requestAnimationFrame(animate);
      }
    },
    [value, duration]
  );

  useEffect(() => {
    startTime.current = null;
    rafRef.current = requestAnimationFrame(animate);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [animate]);

  return (
    <span>
      {display}
      {suffix}
    </span>
  );
}

function PersonalizationProgress({
  label,
  current,
  target,
}: {
  label: string;
  current: number;
  target: number;
}) {
  const progress = (current / target) * 100;
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm text-gray-400">{label}</span>
        <span className="text-xs text-gray-500">
          {current}% / {target}%
        </span>
      </div>
      <div className="h-2 rounded-full bg-white/5 overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          whileInView={{ width: `${progress}%` }}
          viewport={{ once: true }}
          transition={{ duration: 1, ease: 'easeOut' }}
          className="h-full rounded-full bg-gradient-to-r from-neuro-500 to-purple-500"
        />
      </div>
    </div>
  );
}

function PerformanceGauge({
  label,
  value,
  max = 100,
  color = 'neuro',
}: {
  label: string;
  value: number;
  max?: number;
  color?: 'neuro' | 'emerald' | 'amber' | 'rose';
}) {
  const percentage = (value / max) * 100;
  const colors = {
    neuro: 'stroke-neuro-400',
    emerald: 'stroke-emerald-400',
    amber: 'stroke-amber-400',
    rose: 'stroke-rose-400',
  };
  return (
    <div className="flex flex-col items-center gap-2">
      <svg className="w-16 h-16 -rotate-90" viewBox="0 0 64 64">
        <circle
          cx="32"
          cy="32"
          r="28"
          fill="none"
          stroke="hsl(0, 0%, 12%)"
          strokeWidth="4"
        />
        <motion.circle
          cx="32"
          cy="32"
          r="28"
          fill="none"
          className={colors[color]}
          strokeWidth="4"
          strokeLinecap="round"
          strokeDasharray={`${2 * Math.PI * 28}`}
          initial={{ strokeDashoffset: 2 * Math.PI * 28 }}
          whileInView={{ strokeDashoffset: 2 * Math.PI * 28 * (1 - percentage / 100) }}
          viewport={{ once: true }}
          transition={{ duration: 1.5, ease: 'easeOut' }}
        />
      </svg>
      <span className="text-lg font-bold text-white">{value}%</span>
      <span className="text-xs text-gray-500">{label}</span>
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
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
};

export default function DashboardPage() {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const fetchMetrics = async () => {
      try {
        const response = await api.getDashboardMetrics();
        if (response.success) {
          setMetrics(response.data);
        }
      } catch {
        // Use fallback data
      }
    };
    fetchMetrics();
  }, []);

  if (!mounted) return null;

  const displayMetrics = metrics || {
    gestureAccuracy: 97,
    speechConfidence: 94,
    emotionDetectionRate: 91,
    sessionsToday: 24,
    activeUsers: 156,
    systemUptime: 99.9,
    adaptationScore: 82,
  };

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
          <motion.div variants={itemVariants} className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white">Dashboard</h1>
              <p className="text-gray-400 mt-1">Real-time system overview and performance metrics</p>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg glass text-sm">
                <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-emerald-400 font-medium">Live</span>
              </div>
              <span className="text-sm text-gray-500">
                Updated {new Date().toLocaleTimeString()}
              </span>
            </div>
          </motion.div>

          {/* Metrics Cards */}
          <motion.div
            variants={itemVariants}
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"
          >
            <MetricsCard
              icon={<Move className="w-5 h-5" />}
              label="Gesture Accuracy"
              value={displayMetrics.gestureAccuracy}
              suffix="%"
              trend="up"
              trendValue="+2.4%"
              color="neuro"
            />
            <MetricsCard
              icon={<Mic className="w-5 h-5" />}
              label="Speech Confidence"
              value={displayMetrics.speechConfidence}
              suffix="%"
              trend="up"
              trendValue="+1.8%"
              color="neuro"
            />
            <MetricsCard
              icon={<Brain className="w-5 h-5" />}
              label="Emotion Detection"
              value={displayMetrics.emotionDetectionRate}
              suffix="%"
              trend="up"
              trendValue="+3.2%"
              color="neuro"
            />
            <MetricsCard
              icon={<MessageCircle className="w-5 h-5" />}
              label="Sessions Today"
              value={displayMetrics.sessionsToday}
              trend="up"
              trendValue="+5"
              color="neuro"
            />
          </motion.div>

          {/* Charts and Data Row */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Activity Chart */}
            <motion.div variants={itemVariants} className="lg:col-span-2">
              <GlassCard className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h2 className="text-lg font-semibold text-white">Weekly Performance</h2>
                    <p className="text-sm text-gray-500">Accuracy trends across modalities</p>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-neuro-400" />
                      <span className="text-xs text-gray-400">Gesture</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-emerald-400" />
                      <span className="text-xs text-gray-400">Speech</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-purple-400" />
                      <span className="text-xs text-gray-400">Emotion</span>
                    </div>
                  </div>
                </div>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(0, 0%, 12%)" />
                      <XAxis
                        dataKey="day"
                        stroke="hsl(0, 0%, 40%)"
                        fontSize={12}
                        tickLine={false}
                      />
                      <YAxis
                        stroke="hsl(0, 0%, 40%)"
                        fontSize={12}
                        tickLine={false}
                        domain={[75, 100]}
                      />
                      <Tooltip
                        contentStyle={{
                          background: 'hsl(0, 0%, 6%)',
                          border: '1px solid hsla(0, 0%, 100%, 0.06)',
                          borderRadius: '8px',
                          backdropFilter: 'blur(16px)',
                        }}
                        labelStyle={{ color: 'hsl(0, 0%, 95%)' }}
                      />
                      <Line
                        type="monotone"
                        dataKey="accuracy"
                        stroke="hsl(239, 84%, 67%)"
                        strokeWidth={2}
                        dot={false}
                        animationDuration={1500}
                      />
                      <Line
                        type="monotone"
                        dataKey="speech"
                        stroke="#34d399"
                        strokeWidth={2}
                        dot={false}
                        animationDuration={1500}
                      />
                      <Line
                        type="monotone"
                        dataKey="emotion"
                        stroke="#a78bfa"
                        strokeWidth={2}
                        dot={false}
                        animationDuration={1500}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </GlassCard>
            </motion.div>

            {/* System Performance */}
            <motion.div variants={itemVariants}>
              <GlassCard className="p-6 h-full">
                <h2 className="text-lg font-semibold text-white mb-6">System Performance</h2>
                <div className="grid grid-cols-2 gap-4">
                  <PerformanceGauge
                    label="CPU"
                    value={34}
                    color="emerald"
                  />
                  <PerformanceGauge
                    label="Memory"
                    value={52}
                    color="amber"
                  />
                  <PerformanceGauge
                    label="Latency"
                    value={12}
                    max={50}
                    color="neuro"
                  />
                  <PerformanceGauge
                    label="Uptime"
                    value={99.9}
                    color="emerald"
                  />
                </div>
              </GlassCard>
            </motion.div>
          </div>

          {/* Bottom Row */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Recent Communications */}
            <motion.div variants={itemVariants} className="lg:col-span-2">
              <GlassCard className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold text-white">Recent Communications</h2>
                  <button className="text-sm text-neuro-400 hover:text-neuro-300 transition-colors">
                    View All
                  </button>
                </div>
                <div className="space-y-3">
                  {recentCommunications.map((session) => (
                    <div
                      key={session.id}
                      className="flex items-center justify-between p-3 rounded-xl hover:bg-white/5 transition-colors cursor-pointer group"
                    >
                      <div className="flex items-center gap-3">
                        <div
                          className={cn(
                            'w-10 h-10 rounded-lg flex items-center justify-center',
                            session.type === 'multimodal'
                              ? 'bg-neuro-500/10 text-neuro-400'
                              : session.type === 'gesture'
                              ? 'bg-emerald-500/10 text-emerald-400'
                              : session.type === 'speech'
                              ? 'bg-blue-500/10 text-blue-400'
                              : 'bg-purple-500/10 text-purple-400'
                          )}
                        >
                          {session.type === 'multimodal' ? (
                            <Brain className="w-5 h-5" />
                          ) : session.type === 'gesture' ? (
                            <Move className="w-5 h-5" />
                          ) : session.type === 'speech' ? (
                            <Mic className="w-5 h-5" />
                          ) : (
                            <MessageCircle className="w-5 h-5" />
                          )}
                        </div>
                        <div>
                          <p className="text-sm font-medium text-white capitalize">
                            {session.type} Session
                          </p>
                          <p className="text-xs text-gray-500">
                            {session.metrics.totalMessages} messages ·{' '}
                            {Math.round(session.metrics.duration / 60)} min
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="text-right">
                          <p className="text-sm font-medium text-white">
                            {Math.round(session.metrics.averageConfidence * 100)}%
                          </p>
                          <p className="text-xs text-gray-500">confidence</p>
                        </div>
                        <div
                          className={cn(
                            'w-2 h-2 rounded-full',
                            session.status === 'active'
                              ? 'bg-emerald-400 animate-pulse'
                              : 'bg-gray-600'
                          )}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </GlassCard>
            </motion.div>

            {/* Personalization Progress */}
            <motion.div variants={itemVariants}>
              <GlassCard className="p-6 h-full">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-lg font-semibold text-white">Personalization</h2>
                  <span className="text-sm text-neuro-400 font-medium">
                    {displayMetrics.adaptationScore}%
                  </span>
                </div>
                <div className="space-y-5">
                  {personalizationData.map((item) => (
                    <PersonalizationProgress
                      key={item.category}
                      label={item.category}
                      current={item.current}
                      target={item.target}
                    />
                  ))}
                </div>
              </GlassCard>
            </motion.div>
          </div>
        </motion.div>
      </div>
    </main>
  );
}
