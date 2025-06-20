import React from 'react';
import { motion } from 'framer-motion';

interface QuantumButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'quantum' | 'neural' | 'cyber';
  size?: 'sm' | 'md' | 'lg' | 'xl';
  disabled?: boolean;
  loading?: boolean;
  className?: string;
}

export default function QuantumButton({
  children,
  onClick,
  variant = 'primary',
  size = 'md',
  disabled = false,
  loading = false,
  className = '',
}: QuantumButtonProps) {
  const variants = {
    primary: 'bg-gradient-to-r from-neural-500 to-neural-600 hover:from-neural-600 hover:to-neural-700 text-white',
    secondary: 'bg-gradient-to-r from-gray-600 to-gray-700 hover:from-gray-700 hover:to-gray-800 text-white',
    quantum: 'bg-gradient-to-r from-nexus-500 to-nexus-600 hover:from-nexus-600 hover:to-nexus-700 text-white',
    neural: 'bg-gradient-to-r from-quantum-500 to-quantum-600 hover:from-quantum-600 hover:to-quantum-700 text-white',
    cyber: 'bg-gradient-to-r from-cyber-500 to-cyber-600 hover:from-cyber-600 hover:to-cyber-700 text-white',
  };

  const sizes = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-base',
    lg: 'px-6 py-3 text-lg',
    xl: 'px-8 py-4 text-xl',
  };

  const buttonVariants = {
    idle: { 
      scale: 1,
      boxShadow: '0 0 20px rgba(14, 165, 233, 0.3)',
    },
    hover: { 
      scale: 1.05,
      boxShadow: '0 0 30px rgba(14, 165, 233, 0.6)',
      transition: {
        duration: 0.2,
        ease: "easeOut"
      }
    },
    tap: { 
      scale: 0.95,
      transition: {
        duration: 0.1,
        ease: "easeIn"
      }
    },
    disabled: {
      scale: 1,
      opacity: 0.5,
      boxShadow: 'none',
    }
  };

  const quantumRipple = {
    initial: { scale: 0, opacity: 1 },
    animate: { 
      scale: 4, 
      opacity: 0,
      transition: {
        duration: 0.6,
        ease: "easeOut"
      }
    }
  };

  return (
    <motion.button
      className={`
        relative overflow-hidden rounded-xl font-semibold
        backdrop-blur-sm border border-white/20
        transition-all duration-300 ease-out
        disabled:cursor-not-allowed
        ${variants[variant]}
        ${sizes[size]}
        ${className}
      `}
      variants={buttonVariants}
      initial="idle"
      whileHover={!disabled ? "hover" : "disabled"}
      whileTap={!disabled ? "tap" : "disabled"}
      animate={disabled ? "disabled" : "idle"}
      onClick={onClick}
      disabled={disabled || loading}
    >
      {/* Quantum field background */}
      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -skew-x-12 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
      
      {/* Neural network pattern */}
      <div className="absolute inset-0 opacity-20">
        <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
          <defs>
            <pattern id="button-neural" x="0" y="0" width="10" height="10" patternUnits="userSpaceOnUse">
              <circle cx="5" cy="5" r="0.5" fill="currentColor" opacity="0.3">
                <animate attributeName="opacity" values="0.3;0.8;0.3" dur="2s" repeatCount="indefinite" />
              </circle>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#button-neural)" />
        </svg>
      </div>
      
      {/* Content */}
      <span className="relative z-10 flex items-center justify-center gap-2">
        {loading && (
          <motion.div
            className="w-4 h-4 border-2 border-current border-t-transparent rounded-full"
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          />
        )}
        {children}
      </span>
      
      {/* Quantum ripple effect on click */}
      <motion.div
        className="absolute inset-0 bg-white/20 rounded-xl"
        variants={quantumRipple}
        initial="initial"
        whileTap="animate"
      />
      
      {/* Holographic edges */}
      <div className="absolute inset-0 rounded-xl border border-white/30 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
      
      {/* Energy particles */}
      {!disabled && (
        <div className="absolute inset-0 pointer-events-none">
          {Array.from({ length: 3 }).map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 bg-white rounded-full"
              style={{
                left: `${20 + i * 30}%`,
                top: `${20 + i * 20}%`,
              }}
              animate={{
                y: [-10, 10, -10],
                opacity: [0, 1, 0],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                delay: i * 0.5,
                ease: "easeInOut"
              }}
            />
          ))}
        </div>
      )}
    </motion.button>
  );
}