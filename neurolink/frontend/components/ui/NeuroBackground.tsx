'use client';

import { useEffect, useRef, useCallback } from 'react';

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  opacity: number;
  hue: number;
}

interface Orb {
  x: number;
  y: number;
  radius: number;
  hue: number;
  opacity: number;
  speed: number;
  phase: number;
}

export default function NeuroBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const orbsRef = useRef<Orb[]>([]);
  const mouseRef = useRef({ x: 0, y: 0 });
  const animationFrameRef = useRef<number>(0);
  const timeRef = useRef(0);

  const PARTICLE_COUNT = 80;
  const ORB_COUNT = 3;
  const CONNECTION_DISTANCE = 120;
  const MOUSE_INFLUENCE = 40;

  const initParticles = useCallback((width: number, height: number): Particle[] => {
    return Array.from({ length: PARTICLE_COUNT }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5,
      size: Math.random() * 2 + 1,
      opacity: Math.random() * 0.5 + 0.2,
      hue: Math.random() * 60 + 240,
    }));
  }, []);

  const initOrbs = useCallback((width: number, height: number): Orb[] => {
    return [
      {
        x: width * 0.3,
        y: height * 0.4,
        radius: Math.min(width, height) * 0.25,
        hue: 239,
        opacity: 0.12,
        speed: 0.3,
        phase: 0,
      },
      {
        x: width * 0.7,
        y: height * 0.6,
        radius: Math.min(width, height) * 0.2,
        hue: 262,
        opacity: 0.1,
        speed: 0.4,
        phase: 2.1,
      },
      {
        x: width * 0.5,
        y: height * 0.3,
        radius: Math.min(width, height) * 0.15,
        hue: 188,
        opacity: 0.08,
        speed: 0.25,
        phase: 4.2,
      },
    ];
  }, []);

  const drawConnections = useCallback(
    (ctx: CanvasRenderingContext2D, particles: Particle[], width: number, height: number) => {
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < CONNECTION_DISTANCE) {
            const alpha = (1 - distance / CONNECTION_DISTANCE) * 0.15;
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `hsla(239, 84%, 67%, ${alpha})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }
    },
    []
  );

  const animate = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const particles = particlesRef.current;
    const orbs = orbsRef.current;

    timeRef.current += 0.005;

    ctx.clearRect(0, 0, width, height);

    // Draw gradient orbs
    orbs.forEach((orb) => {
      const pulse = Math.sin(timeRef.current * orb.speed + orb.phase) * 0.3 + 0.7;
      const gradient = ctx.createRadialGradient(
        orb.x + Math.sin(timeRef.current * 0.5 + orb.phase) * 20,
        orb.y + Math.cos(timeRef.current * 0.3 + orb.phase) * 20,
        0,
        orb.x,
        orb.y,
        orb.radius * pulse
      );
      gradient.addColorStop(0, `hsla(${orb.hue}, 84%, 67%, ${orb.opacity * pulse})`);
      gradient.addColorStop(0.5, `hsla(${orb.hue}, 84%, 67%, ${orb.opacity * pulse * 0.5})`);
      gradient.addColorStop(1, `hsla(${orb.hue}, 84%, 67%, 0)`);
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);
    });

    // Update and draw particles
    particles.forEach((particle) => {
      const dx = mouseRef.current.x - particle.x;
      const dy = mouseRef.current.y - particle.y;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist < MOUSE_INFLUENCE * 3) {
        const force = (MOUSE_INFLUENCE * 3 - dist) / (MOUSE_INFLUENCE * 3);
        particle.vx -= (dx / dist) * force * 0.2;
        particle.vy -= (dy / dist) * force * 0.2;
      }

      particle.x += particle.vx;
      particle.y += particle.vy;
      particle.vx += (Math.random() - 0.5) * 0.01;
      particle.vy += (Math.random() - 0.5) * 0.01;

      // Damping
      particle.vx *= 0.99;
      particle.vy *= 0.99;

      // Wrap around edges
      if (particle.x < 0) particle.x = width;
      if (particle.x > width) particle.x = 0;
      if (particle.y < 0) particle.y = height;
      if (particle.y > height) particle.y = 0;

      ctx.beginPath();
      ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${particle.hue}, 84%, 67%, ${particle.opacity})`;
      ctx.fill();
    });

    drawConnections(ctx, particles, width, height);

    animationFrameRef.current = requestAnimationFrame(animate);
  }, [drawConnections]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
      canvas.style.width = `${window.innerWidth}px`;
      canvas.style.height = `${window.innerHeight}px`;

      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.scale(dpr, dpr);
      }

      particlesRef.current = initParticles(window.innerWidth, window.innerHeight);
      orbsRef.current = initOrbs(window.innerWidth, window.innerHeight);
    };

    const handleMouseMove = (e: MouseEvent) => {
      mouseRef.current = { x: e.clientX, y: e.clientY };
    };

    const handleTouchMove = (e: TouchEvent) => {
      const touch = e.touches[0];
      mouseRef.current = { x: touch.clientX, y: touch.clientY };
    };

    resize();
    animate();

    window.addEventListener('resize', resize);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('touchmove', handleTouchMove, { passive: true });

    return () => {
      window.removeEventListener('resize', resize);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('touchmove', handleTouchMove);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [animate, initParticles, initOrbs]);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none"
      style={{ zIndex: 0 }}
      aria-hidden="true"
    />
  );
}
