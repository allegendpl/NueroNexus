import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

interface NeuralVisualizationProps {
  data: number[];
  width?: number;
  height?: number;
  theme?: 'neural' | 'quantum' | 'nexus' | 'cyber' | 'matrix';
  animated?: boolean;
}

export default function NeuralVisualization({
  data,
  width = 400,
  height = 200,
  theme = 'neural',
  animated = true
}: NeuralVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  const themeColors = {
    neural: {
      primary: '#0ea5e9',
      secondary: '#38bdf8',
      accent: '#7dd3fc',
      background: 'rgba(14, 165, 233, 0.1)'
    },
    quantum: {
      primary: '#d946ef',
      secondary: '#e879f9',
      accent: '#f0abfc',
      background: 'rgba(217, 70, 239, 0.1)'
    },
    nexus: {
      primary: '#10b981',
      secondary: '#34d399',
      accent: '#6ee7b7',
      background: 'rgba(16, 185, 129, 0.1)'
    },
    cyber: {
      primary: '#f97316',
      secondary: '#fb923c',
      accent: '#fdba74',
      background: 'rgba(249, 115, 22, 0.1)'
    },
    matrix: {
      primary: '#22c55e',
      secondary: '#4ade80',
      accent: '#86efac',
      background: 'rgba(34, 197, 94, 0.1)'
    }
  };

  const colors = themeColors[theme];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = width;
    canvas.height = height;

    let time = 0;

    const animate = () => {
      ctx.clearRect(0, 0, width, height);
      
      // Background gradient
      const gradient = ctx.createLinearGradient(0, 0, width, height);
      gradient.addColorStop(0, colors.background);
      gradient.addColorStop(1, 'transparent');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);

      // Neural network visualization
      const nodeCount = Math.min(data.length, 50);
      const nodes: { x: number; y: number; value: number; connections: number[] }[] = [];
      
      // Create nodes
      for (let i = 0; i < nodeCount; i++) {
        const angle = (i / nodeCount) * Math.PI * 2;
        const radius = Math.min(width, height) * 0.3;
        const centerX = width / 2;
        const centerY = height / 2;
        
        nodes.push({
          x: centerX + Math.cos(angle + time * 0.01) * radius,
          y: centerY + Math.sin(angle + time * 0.01) * radius,
          value: data[i] || 0,
          connections: []
        });
      }

      // Create connections
      nodes.forEach((node, i) => {
        const connectionCount = Math.floor(Math.random() * 3) + 1;
        for (let j = 0; j < connectionCount; j++) {
          const targetIndex = Math.floor(Math.random() * nodes.length);
          if (targetIndex !== i && !node.connections.includes(targetIndex)) {
            node.connections.push(targetIndex);
          }
        }
      });

      // Draw connections
      ctx.strokeStyle = colors.secondary;
      ctx.lineWidth = 1;
      nodes.forEach((node, i) => {
        node.connections.forEach(targetIndex => {
          const target = nodes[targetIndex];
          const strength = (node.value + target.value) / 2;
          
          ctx.globalAlpha = strength * 0.5 + 0.1;
          ctx.beginPath();
          ctx.moveTo(node.x, node.y);
          ctx.lineTo(target.x, target.y);
          ctx.stroke();
          
          // Animated data flow
          if (animated) {
            const progress = (time * 0.02) % 1;
            const flowX = node.x + (target.x - node.x) * progress;
            const flowY = node.y + (target.y - node.y) * progress;
            
            ctx.fillStyle = colors.accent;
            ctx.globalAlpha = 0.8;
            ctx.beginPath();
            ctx.arc(flowX, flowY, 2, 0, Math.PI * 2);
            ctx.fill();
          }
        });
      });

      // Draw nodes
      nodes.forEach((node, i) => {
        const size = 3 + node.value * 10;
        const pulse = animated ? Math.sin(time * 0.05 + i) * 0.5 + 0.5 : 1;
        
        // Node glow
        const glowGradient = ctx.createRadialGradient(
          node.x, node.y, 0,
          node.x, node.y, size * 2
        );
        glowGradient.addColorStop(0, colors.primary);
        glowGradient.addColorStop(1, 'transparent');
        
        ctx.fillStyle = glowGradient;
        ctx.globalAlpha = 0.3 * pulse;
        ctx.beginPath();
        ctx.arc(node.x, node.y, size * 2, 0, Math.PI * 2);
        ctx.fill();
        
        // Node core
        ctx.fillStyle = colors.primary;
        ctx.globalAlpha = 0.8 + 0.2 * pulse;
        ctx.beginPath();
        ctx.arc(node.x, node.y, size, 0, Math.PI * 2);
        ctx.fill();
        
        // Node highlight
        ctx.fillStyle = colors.accent;
        ctx.globalAlpha = 0.6;
        ctx.beginPath();
        ctx.arc(node.x - size * 0.3, node.y - size * 0.3, size * 0.3, 0, Math.PI * 2);
        ctx.fill();
      });

      // Neural activity waves
      if (animated) {
        ctx.strokeStyle = colors.primary;
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.3;
        
        for (let wave = 0; wave < 3; wave++) {
          ctx.beginPath();
          for (let x = 0; x < width; x += 2) {
            const y = height / 2 + 
              Math.sin((x + time * 2 + wave * 100) * 0.02) * 20 * 
              Math.sin(time * 0.01 + wave) +
              Math.sin((x + time * 3 + wave * 150) * 0.01) * 10;
            
            if (x === 0) {
              ctx.moveTo(x, y);
            } else {
              ctx.lineTo(x, y);
            }
          }
          ctx.stroke();
        }
      }

      ctx.globalAlpha = 1;
      time++;
      
      if (animated) {
        animationRef.current = requestAnimationFrame(animate);
      }
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [data, width, height, theme, animated, colors]);

  return (
    <motion.div
      className="relative rounded-xl overflow-hidden backdrop-blur-sm bg-black/20 border border-white/10"
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
    >
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ width, height }}
      />
      
      {/* Overlay effects */}
      <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent pointer-events-none" />
      
      {/* Stats overlay */}
      <div className="absolute top-2 right-2 text-xs text-white/70 font-mono">
        <div>Nodes: {Math.min(data.length, 50)}</div>
        <div>Activity: {(data.reduce((a, b) => a + b, 0) / data.length * 100).toFixed(1)}%</div>
      </div>
    </motion.div>
  );
}