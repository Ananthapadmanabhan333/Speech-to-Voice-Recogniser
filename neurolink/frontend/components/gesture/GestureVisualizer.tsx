'use client';

import { useMemo } from 'react';
import { motion } from 'framer-motion';
import type { Gesture, Point3D } from '@/types';
import { cn } from '@/lib/cn';

const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

const FINGER_TIPS = [4, 8, 12, 16, 20];

const FINGER_NAMES: Record<number, string> = {
  4: 'Thumb',
  8: 'Index',
  12: 'Middle',
  16: 'Ring',
  20: 'Pinky',
};

const LANDMARK_COLORS = [
  '#6366f1', '#818cf8', '#a5b4fc', '#c7d2fe',
  '#818cf8', '#6366f1', '#818cf8', '#a5b4fc',
  '#6366f1', '#818cf8', '#a5b4fc', '#c7d2fe',
  '#6366f1', '#818cf8', '#a5b4fc', '#c7d2fe',
  '#6366f1', '#818cf8', '#a5b4fc', '#c7d2fe',
  '#a5b4fc',
];

function Landmark({ point, index, confidence }: { point: Point3D; index: number; color: string; confidence: number }) {
  const size = FINGER_TIPS.includes(index) ? 4 : 3;
  const opacity = confidence > 0.8 ? 1 : confidence > 0.5 ? 0.6 : 0.3;

  return (
    <motion.circle
      cx={point.x}
      cy={point.y}
      r={size}
      fill={LANDMARK_COLORS[index] || '#6366f1'}
      fillOpacity={opacity}
      initial={{ scale: 0 }}
      animate={{ scale: 1 }}
      transition={{ duration: 0.2 }}
      className="drop-shadow-lg"
    />
  );
}

function Connection({
  from,
  to,
  confidence,
}: {
  from: Point3D;
  to: Point3D;
  confidence: number;
}) {
  const opacity = confidence > 0.8 ? 0.6 : confidence > 0.5 ? 0.3 : 0.1;

  return (
    <line
      x1={from.x}
      y1={from.y}
      x2={to.x}
      y2={to.y}
      stroke="hsl(239, 84%, 67%)"
      strokeOpacity={opacity}
      strokeWidth={1.5}
      strokeLinecap="round"
    />
  );
}

function GestureLabel({
  type,
  confidence,
  handedness,
  boundingBox,
}: {
  type: string;
  confidence: number;
  handedness: string;
  boundingBox?: { x: number; y: number; width: number; height: number };
}) {
  const labelX = boundingBox ? boundingBox.x + boundingBox.width / 2 : 120;
  const labelY = boundingBox ? boundingBox.y - 10 : 20;

  return (
    <motion.g
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <rect
        x={labelX - 60}
        y={labelY - 14}
        width={120}
        height={28}
        rx={14}
        fill="hsl(239, 84%, 67%)"
        fillOpacity={0.15}
        stroke="hsl(239, 84%, 67%)"
        strokeOpacity={0.3}
        strokeWidth={1}
      />
      <text
        x={labelX}
        y={labelY + 1}
        textAnchor="middle"
        fill="hsl(239, 84%, 67%)"
        fontSize={11}
        fontWeight={600}
        fontFamily="var(--font-inter), sans-serif"
      >
        {type.replace('_', ' ')} · {(confidence * 100).toFixed(0)}%
      </text>
    </motion.g>
  );
}

interface GestureVisualizerProps {
  gesture: Gesture | null;
  width?: number;
  height?: number;
  className?: string;
}

export default function GestureVisualizer({
  gesture,
  width = 400,
  height = 300,
  className,
}: GestureVisualizerProps) {
  const landmarks = gesture?.landmarks ?? [];
  const confidence = gesture?.confidence ?? 0;

  const scalePoints = (points: Point3D[]): Point3D[] => {
    if (points.length === 0) return [];
    const padding = 20;
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    points.forEach((p) => {
      minX = Math.min(minX, p.x);
      maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y);
      maxY = Math.max(maxY, p.y);
    });

    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const scale = Math.min((width - padding * 2) / rangeX, (height - padding * 2) / rangeY);
    const offsetX = (width - rangeX * scale) / 2 - minX * scale;
    const offsetY = (height - rangeY * scale) / 2 - minY * scale;

    return points.map((p) => ({
      x: p.x * scale + offsetX,
      y: p.y * scale + offsetY,
      z: p.z,
    }));
  };

  const scaledLandmarks = useMemo(() => scalePoints(landmarks), [landmarks, width, height]);

  if (!gesture || landmarks.length === 0) {
    return (
      <div
        className={cn(
          'flex items-center justify-center h-full bg-gray-900/50 rounded-xl',
          className
        )}
      >
        <p className="text-sm text-gray-600">No gesture detected</p>
      </div>
    );
  }

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className={cn('w-full h-full', className)}
      role="img"
      aria-label={`Gesture visualization: ${gesture.type.replace('_', ' ')} with ${(confidence * 100).toFixed(0)}% confidence`}
    >
      <rect width={width} height={height} rx={12} fill="hsl(0, 0%, 3%)" fillOpacity={0.5} />

      {/* Connections */}
      {HAND_CONNECTIONS.map(([fromIdx, toIdx]) => {
        if (fromIdx >= scaledLandmarks.length || toIdx >= scaledLandmarks.length) return null;
        return (
          <Connection
            key={`conn-${fromIdx}-${toIdx}`}
            from={scaledLandmarks[fromIdx]}
            to={scaledLandmarks[toIdx]}
            confidence={confidence}
          />
        );
      })}

      {/* Landmarks */}
      {scaledLandmarks.map((point, index) => (
        <Landmark
          key={`lm-${index}`}
          point={point}
          index={index}
          color={LANDMARK_COLORS[index] || '#6366f1'}
          confidence={confidence}
        />
      ))}

      {/* Gesture Label */}
      <GestureLabel
        type={gesture.type}
        confidence={confidence}
        handedness={gesture.handedness}
        boundingBox={gesture.boundingBox}
      />
    </svg>
  );
}
