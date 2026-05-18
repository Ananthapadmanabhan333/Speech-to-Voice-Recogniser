'use client';

import { useEffect, useState } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import {
  Move,
  Mic,
  Brain,
 MessageCircle,
  Activity,
  Zap,
  ArrowRight,
  Sparkles,
  Fingerprint,
} from 'lucide-react';
import NeuroBackground from '@/components/ui/NeuroBackground';
import GlassCard from '@/components/ui/GlassCard';
import { cn } from '@/lib/cn';

interface FeatureCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  index: number;
}

function FeatureCard({ icon, title, description, index }: FeatureCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-50px' }}
      transition={{ delay: index * 0.1, duration: 0.5 }}
    >
      <GlassCard className="group p-6 h-full cursor-default">
        <div className="flex flex-col gap-4">
          <div className="w-12 h-12 rounded-xl bg-neuro-500/10 flex items-center justify-center text-neuro-400 group-hover:bg-neuro-500/20 group-hover:scale-110 transition-all duration-300">
            {icon}
          </div>
          <h3 className="text-lg font-semibold text-white">{title}</h3>
          <p className="text-sm text-gray-400 leading-relaxed">{description}</p>
        </div>
      </GlassCard>
    </motion.div>
  );
}

function MetricCard({
  label,
  value,
  unit,
  icon,
  trend,
}: {
  label: string;
  value: string;
  unit?: string;
  icon: React.ReactNode;
  trend?: 'up' | 'down';
}) {
  return (
    <GlassCard className="p-4">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg bg-neuro-500/10 flex items-center justify-center text-neuro-400">
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-xs text-gray-500 uppercase tracking-wider">{label}</p>
          <div className="flex items-baseline gap-1">
            <span className="text-xl font-bold text-white">{value}</span>
            {unit && <span className="text-sm text-gray-400">{unit}</span>}
          </div>
        </div>
        {trend && (
          <span
            className={cn(
              'text-xs font-medium',
              trend === 'up' ? 'text-emerald-400' : 'text-red-400'
            )}
          >
            {trend === 'up' ? '↑' : '↓'}
          </span>
        )}
      </div>
    </GlassCard>
  );
}

function QuickActionButton({
  icon,
  label,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  color: string;
}) {
  return (
    <motion.button
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      className={cn(
        'flex items-center gap-3 px-5 py-3 rounded-xl text-sm font-medium transition-all duration-200',
        'border border-white/10 hover:border-white/20 backdrop-blur-sm',
        color
      )}
    >
      {icon}
      <span>{label}</span>
    </motion.button>
  );
}

function SystemStatusBar() {
  const statuses = [
    { label: 'Gesture Engine', status: 'active', latency: '12ms' },
    { label: 'Speech Processor', status: 'active', latency: '8ms' },
    { label: 'Emotion AI', status: 'active', latency: '15ms' },
    { label: 'Fusion Layer', status: 'active', latency: '5ms' },
  ];

  return (
    <GlassCard className="p-4" aria-label="System status">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-sm font-medium text-emerald-400">All Systems Active</span>
        </div>
        <div className="flex items-center gap-4 flex-wrap">
          {statuses.map((s) => (
            <div key={s.label} className="flex items-center gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              <span className="text-xs text-gray-400">{s.label}</span>
              <span className="text-xs text-gray-500">{s.latency}</span>
            </div>
          ))}
        </div>
      </div>
    </GlassCard>
  );
}

function HeroSection() {
  const { scrollY } = useScroll();
  const heroY = useTransform(scrollY, [0, 500], [0, 150]);
  const heroOpacity = useTransform(scrollY, [0, 400], [1, 0]);

  return (
    <motion.section
      style={{ y: heroY, opacity: heroOpacity }}
      className="relative min-h-screen flex items-center justify-center overflow-hidden"
    >
      <NeuroBackground />

      <div className="relative z-10 max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass mb-8">
            <Sparkles className="w-4 h-4 text-neuro-400" />
            <span className="text-sm text-neuro-300 font-medium">
              Next-Gen Communication Intelligence
            </span>
          </div>

          <h1 className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-bold tracking-tight mb-6">
            <span className="text-white">Think. </span>
            <span className="text-gradient">Gesture. </span>
            <br />
            <span className="text-white">Speak. </span>
            <span className="text-gradient">Connect.</span>
          </h1>

          <p className="text-lg sm:text-xl text-gray-400 max-w-2xl mx-auto mb-10 leading-relaxed">
            Neurolink bridges the gap between human intent and machine understanding
            through adaptive multimodal AI — combining gestures, speech, and emotions
            in real-time.
          </p>

          <div className="flex flex-wrap items-center justify-center gap-4">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="inline-flex items-center gap-2 px-8 py-4 rounded-xl bg-neuro-500 text-white font-semibold text-lg hover:bg-neuro-600 transition-colors shadow-lg shadow-neuro-500/20"
            >
              Start Communicating
              <ArrowRight className="w-5 h-5" />
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="inline-flex items-center gap-2 px-8 py-4 rounded-xl border border-white/10 text-white font-semibold text-lg hover:bg-white/5 transition-colors"
            >
              Watch Demo
            </motion.button>
          </div>
        </motion.div>
      </div>

      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-[hsl(var(--background))] to-transparent pointer-events-none" />
    </motion.section>
  );
}

function FeaturesSection() {
  const features = [
    {
      icon: <Move className="w-6 h-6" />,
      title: 'Gesture Recognition',
      description:
        'Real-time hand tracking and gesture interpretation with 21-point landmark detection and adaptive learning.',
    },
    {
      icon: <Mic className="w-6 h-6" />,
      title: 'Speech Analysis',
      description:
        'Multi-language speech recognition with sentiment analysis, speaker diarization, and adaptive noise filtering.',
    },
    {
      icon: <Brain className="w-6 h-6" />,
      title: 'Emotion Detection',
      description:
        'Facial expression and vocal tone analysis for comprehensive emotion understanding with arousal-valence mapping.',
    },
    {
      icon: <MessageCircle className="w-6 h-6" />,
      title: 'Multimodal Fusion',
      description:
        'Intelligent fusion of gesture, speech, and emotion data for accurate intent recognition and contextual responses.',
    },
    {
      icon: <Activity className="w-6 h-6" />,
      title: 'Adaptive Learning',
      description:
        'Continuous personalization that adapts to your unique communication patterns, improving accuracy over time.',
    },
    {
      icon: <Zap className="w-6 h-6" />,
      title: 'Real-time Processing',
      description:
        'Sub-20ms latency for gesture processing, with parallel neural network inference for instantaneous responses.',
    },
  ];

  return (
    <section className="relative py-32 px-4 sm:px-6 lg:px-8">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">
            Powered by{' '}
            <span className="text-gradient">Advanced AI</span>
          </h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Six integrated neural networks working in parallel to understand your every gesture,
            word, and emotion.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <FeatureCard key={feature.title} {...feature} index={index} />
          ))}
        </div>
      </div>
    </section>
  );
}

function MetricsSection() {
  const metrics = [
    { label: 'Gesture Accuracy', value: '98.7', unit: '%', icon: <Fingerprint className="w-5 h-5" />, trend: 'up' as const },
    { label: 'Speech Confidence', value: '96.3', unit: '%', icon: <Mic className="w-5 h-5" />, trend: 'up' as const },
    { label: 'Emotion Detection', value: '94.1', unit: '%', icon: <Brain className="w-5 h-5" />, trend: 'up' as const },
    { label: 'Processing Latency', value: '12', unit: 'ms', icon: <Zap className="w-5 h-5" />, trend: 'down' as const },
  ];

  return (
    <section className="relative py-20 px-4 sm:px-6 lg:px-8">
      <div className="max-w-5xl mx-auto">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {metrics.map((metric) => (
            <motion.div
              key={metric.label}
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4 }}
            >
              <MetricCard {...metric} />
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

function CTASection() {
  return (
    <section className="relative py-32 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <GlassCard className="p-12 sm:p-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">
              Ready to Transform Your{' '}
              <span className="text-gradient">Communication?</span>
            </h2>
            <p className="text-gray-400 max-w-xl mx-auto mb-8">
              Join the next generation of human-computer interaction. Start with gesture
              recognition and unlock the full multimodal experience.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-4">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="inline-flex items-center gap-2 px-8 py-4 rounded-xl bg-neuro-500 text-white font-semibold hover:bg-neuro-600 transition-colors shadow-lg shadow-neuro-500/20"
              >
                Get Started Free
                <ArrowRight className="w-5 h-5" />
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="inline-flex items-center gap-2 px-8 py-4 rounded-xl border border-white/10 text-gray-300 font-semibold hover:bg-white/5 transition-colors"
              >
                View Documentation
              </motion.button>
            </div>
          </GlassCard>
        </motion.div>
      </div>
    </section>
  );
}

export default function HomePage() {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  if (!mounted) return null;

  return (
    <main className="relative min-h-screen bg-[hsl(var(--background))] overflow-x-hidden">
      <HeroSection />
      <MetricsSection />
      <FeaturesSection />
      <CTASection />

      <footer className="relative border-t border-white/5 py-8 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-neuro-400" />
            <span className="text-sm font-semibold text-white">Neurolink</span>
          </div>
          <p className="text-xs text-gray-600">
            &copy; {new Date().getFullYear()} Neurolink. All rights reserved.
          </p>
          <div className="flex items-center gap-4">
            <button className="text-xs text-gray-500 hover:text-gray-300 transition-colors">
              Privacy
            </button>
            <button className="text-xs text-gray-500 hover:text-gray-300 transition-colors">
              Terms
            </button>
            <button className="text-xs text-gray-500 hover:text-gray-300 transition-colors">
              Documentation
            </button>
          </div>
        </div>
      </footer>
    </main>
  );
}
