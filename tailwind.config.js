/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'neural': {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
        },
        'nexus': {
          50: '#fdf4ff',
          100: '#fae8ff',
          200: '#f5d0fe',
          300: '#f0abfc',
          400: '#e879f9',
          500: '#d946ef',
          600: '#c026d3',
          700: '#a21caf',
          800: '#86198f',
          900: '#701a75',
        },
        'quantum': {
          50: '#ecfdf5',
          100: '#d1fae5',
          200: '#a7f3d0',
          300: '#6ee7b7',
          400: '#34d399',
          500: '#10b981',
          600: '#059669',
          700: '#047857',
          800: '#065f46',
          900: '#064e3b',
        },
        'cyber': {
          50: '#fff7ed',
          100: '#ffedd5',
          200: '#fed7aa',
          300: '#fdba74',
          400: '#fb923c',
          500: '#f97316',
          600: '#ea580c',
          700: '#c2410c',
          800: '#9a3412',
          900: '#7c2d12',
        }
      },
      fontFamily: {
        'neural': ['Inter', 'system-ui', 'sans-serif'],
        'quantum': ['JetBrains Mono', 'monospace'],
        'nexus': ['Poppins', 'sans-serif'],
      },
      animation: {
        'neural-pulse': 'neural-pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'quantum-spin': 'quantum-spin 3s linear infinite',
        'nexus-float': 'nexus-float 6s ease-in-out infinite',
        'cyber-glow': 'cyber-glow 2s ease-in-out infinite alternate',
        'matrix-rain': 'matrix-rain 20s linear infinite',
        'hologram': 'hologram 4s ease-in-out infinite',
        'data-stream': 'data-stream 15s linear infinite',
        'neural-network': 'neural-network 8s ease-in-out infinite',
        'quantum-entangle': 'quantum-entangle 5s ease-in-out infinite',
        'cyber-scan': 'cyber-scan 3s linear infinite',
      },
      keyframes: {
        'neural-pulse': {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: 0.5 },
        },
        'quantum-spin': {
          '0%': { transform: 'rotate(0deg) scale(1)' },
          '50%': { transform: 'rotate(180deg) scale(1.1)' },
          '100%': { transform: 'rotate(360deg) scale(1)' },
        },
        'nexus-float': {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-20px)' },
        },
        'cyber-glow': {
          '0%': { boxShadow: '0 0 5px #0ea5e9, 0 0 10px #0ea5e9, 0 0 15px #0ea5e9' },
          '100%': { boxShadow: '0 0 10px #d946ef, 0 0 20px #d946ef, 0 0 30px #d946ef' },
        },
        'matrix-rain': {
          '0%': { transform: 'translateY(-100vh)' },
          '100%': { transform: 'translateY(100vh)' },
        },
        'hologram': {
          '0%, 100%': { opacity: 0.8, transform: 'translateZ(0)' },
          '50%': { opacity: 1, transform: 'translateZ(10px)' },
        },
        'data-stream': {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100vw)' },
        },
        'neural-network': {
          '0%, 100%': { strokeDashoffset: 0 },
          '50%': { strokeDashoffset: 100 },
        },
        'quantum-entangle': {
          '0%, 100%': { transform: 'rotate(0deg) translateX(0)' },
          '25%': { transform: 'rotate(90deg) translateX(10px)' },
          '50%': { transform: 'rotate(180deg) translateX(0)' },
          '75%': { transform: 'rotate(270deg) translateX(-10px)' },
        },
        'cyber-scan': {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' },
        },
      },
      backdropBlur: {
        'neural': '20px',
      },
      backgroundImage: {
        'neural-gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'quantum-gradient': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
        'nexus-gradient': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
        'cyber-gradient': 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
        'matrix-gradient': 'linear-gradient(180deg, #000000 0%, #1a1a2e 50%, #16213e 100%)',
      },
    },
  },
  plugins: [],
}