'use client';

import { forwardRef } from 'react';
import { motion, type HTMLMotionProps } from 'framer-motion';
import { cn } from '@/lib/cn';

interface GlassCardProps extends Omit<HTMLMotionProps<'div'>, 'children'> {
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
  glow?: boolean;
  as?: 'div' | 'section' | 'article' | 'aside';
}

const GlassCard = forwardRef<HTMLDivElement, GlassCardProps>(
  ({ children, className, hover = true, glow = false, as: Component = 'div', ...props }, ref) => {
    const MotionComponent = motion[Component as keyof typeof motion];

    const motionProps: HTMLMotionProps<'div'> = {
      ref,
      initial: { opacity: 0, y: 10 },
      whileInView: { opacity: 1, y: 0 },
      viewport: { once: true, margin: '-30px' },
      transition: { duration: 0.4, ease: 'easeOut' },
      ...props,
    };

    return (
      <MotionComponent
        {...motionProps}
        className={cn(
          'glass rounded-2xl transition-all duration-300',
          hover && 'glass-hover',
          glow && 'glow-sm',
          className
        )}
        role="region"
        aria-label={props['aria-label']}
      >
        {children}
      </MotionComponent>
    );
  }
);

GlassCard.displayName = 'GlassCard';

export default GlassCard;
