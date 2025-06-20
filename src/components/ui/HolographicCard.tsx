import React from 'react';
import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';

interface HolographicCardProps {
  children: React.ReactNode;
  className?: string;
  glowColor?: string;
  intensity?: number;
  animated?: boolean;
}

export default function HolographicCard({ 
  children, 
  className = '', 
  glowColor = '#0ea5e9',
  intensity = 1,
  animated = true 
}: HolographicCardProps) {
  const [ref, inView] = useInView({
    threshold: 0.1,
    triggerOnce: false,
  });

  const cardVariants = {
    hidden: { 
      opacity: 0, 
      y: 50, 
      rotateX: -15,
      scale: 0.9 
    },
    visible: { 
      opacity: 1, 
      y: 0, 
      rotateX: 0,
      scale: 1,
      transition: {
        duration: 0.8,
        ease: [0.25, 0.46, 0.45, 0.94],
      }
    },
    hover: {
      y: -10,
      rotateX: 5,
      scale: 1.02,
      transition: {
        duration: 0.3,
        ease: "easeOut"
      }
    }
  };

  const glowVariants = {
    idle: {
      boxShadow: `0 0 20px ${glowColor}40, 0 0 40px ${glowColor}20, 0 0 60px ${glowColor}10`,
    },
    hover: {
      boxShadow: `0 0 30px ${glowColor}60, 0 0 60px ${glowColor}40, 0 0 90px ${glowColor}20`,
    }
  };

  return (
    <motion.div
      ref={ref}
      className={`
        relative backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl
        overflow-hidden group cursor-pointer perspective-1000
        ${className}
      `}
      variants={animated ? cardVariants : {}}
      initial={animated ? "hidden" : "visible"}
      animate={inView && animated ? "visible" : animated ? "hidden" : "visible"}
      whileHover={animated ? "hover" : {}}
      style={{
        background: `linear-gradient(135deg, 
          rgba(255,255,255,0.1) 0%, 
          rgba(255,255,255,0.05) 50%, 
          rgba(255,255,255,0.02) 100%)`,
      }}
    >
      {/* Holographic shimmer effect */}
      <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -skew-x-12 animate-cyber-scan" />
      </div>
      
      {/* Neural network pattern overlay */}
      <div className="absolute inset-0 opacity-20">
        <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
          <defs>
            <pattern id="neural-pattern" x="0" y="0" width="20" height="20" patternUnits="userSpaceOnUse">
              <circle cx="10" cy="10" r="1" fill={glowColor} opacity="0.3">
                <animate attributeName="r" values="1;2;1" dur="3s" repeatCount="indefinite" />
              </circle>
              <line x1="10" y1="10" x2="20" y2="10" stroke={glowColor} strokeWidth="0.5" opacity="0.2" />
              <line x1="10" y1="10" x2="10" y2="20" stroke={glowColor} strokeWidth="0.5" opacity="0.2" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#neural-pattern)" />
        </svg>
      </div>
      
      {/* Quantum field distortion */}
      <div className="absolute inset-0 opacity-30 group-hover:opacity-50 transition-opacity duration-300">
        <div 
          className="w-full h-full animate-hologram"
          style={{
            background: `radial-gradient(circle at 50% 50%, ${glowColor}20 0%, transparent 70%)`,
            filter: 'blur(1px)',
          }}
        />
      </div>
      
      {/* Glow effect */}
      <motion.div
        className="absolute inset-0 rounded-2xl"
        variants={glowVariants}
        initial="idle"
        whileHover="hover"
        style={{
          background: 'transparent',
        }}
      />
      
      {/* Content */}
      <div className="relative z-10 p-6">
        {children}
      </div>
      
      {/* Corner accents */}
      <div className="absolute top-0 left-0 w-8 h-8 border-l-2 border-t-2 border-white/30 rounded-tl-2xl" />
      <div className="absolute top-0 right-0 w-8 h-8 border-r-2 border-t-2 border-white/30 rounded-tr-2xl" />
      <div className="absolute bottom-0 left-0 w-8 h-8 border-l-2 border-b-2 border-white/30 rounded-bl-2xl" />
      <div className="absolute bottom-0 right-0 w-8 h-8 border-r-2 border-b-2 border-white/30 rounded-br-2xl" />
      
      {/* Data stream effect */}
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-white/50 to-transparent animate-data-stream opacity-0 group-hover:opacity-100" />
    </motion.div>
  );
}