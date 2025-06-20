import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Brain, 
  Zap, 
  Settings, 
  User, 
  MessageSquare, 
  BarChart3, 
  Cpu, 
  Network,
  Atom,
  Eye,
  Shield,
  Rocket,
  Database,
  Code,
  Layers,
  GitBranch,
  Activity
} from 'lucide-react';
import { useAppStore } from '../../store';
import HolographicCard from '../ui/HolographicCard';

const menuItems = [
  { 
    id: 'dashboard', 
    label: 'Neural Dashboard', 
    icon: Brain, 
    color: '#0ea5e9',
    description: 'Central command center'
  },
  { 
    id: 'agents', 
    label: 'AI Agents', 
    icon: Cpu, 
    color: '#d946ef',
    description: 'Manage AI entities'
  },
  { 
    id: 'conversations', 
    label: 'Quantum Chat', 
    icon: MessageSquare, 
    color: '#10b981',
    description: 'Advanced conversations'
  },
  { 
    id: 'neural-network', 
    label: 'Neural Networks', 
    icon: Network, 
    color: '#f97316',
    description: 'Deep learning models'
  },
  { 
    id: 'quantum-lab', 
    label: 'Quantum Lab', 
    icon: Atom, 
    color: '#8b5cf6',
    description: 'Quantum computing'
  },
  { 
    id: 'analytics', 
    label: 'Cyber Analytics', 
    icon: BarChart3, 
    color: '#06b6d4',
    description: 'Performance metrics'
  },
  { 
    id: 'vision', 
    label: 'Computer Vision', 
    icon: Eye, 
    color: '#84cc16',
    description: 'Visual AI processing'
  },
  { 
    id: 'security', 
    label: 'AI Security', 
    icon: Shield, 
    color: '#ef4444',
    description: 'Security protocols'
  },
  { 
    id: 'deployment', 
    label: 'Model Deploy', 
    icon: Rocket, 
    color: '#f59e0b',
    description: 'Deploy AI models'
  },
  { 
    id: 'data-lake', 
    label: 'Data Lake', 
    icon: Database, 
    color: '#3b82f6',
    description: 'Massive data storage'
  },
  { 
    id: 'code-gen', 
    label: 'Code Genesis', 
    icon: Code, 
    color: '#ec4899',
    description: 'AI code generation'
  },
  { 
    id: 'model-zoo', 
    label: 'Model Zoo', 
    icon: Layers, 
    color: '#14b8a6',
    description: 'Pre-trained models'
  },
  { 
    id: 'pipeline', 
    label: 'ML Pipeline', 
    icon: GitBranch, 
    color: '#a855f7',
    description: 'Machine learning ops'
  },
  { 
    id: 'monitoring', 
    label: 'System Monitor', 
    icon: Activity, 
    color: '#22c55e',
    description: 'Real-time monitoring'
  },
];

export default function Sidebar() {
  const { sidebarOpen, setSidebarOpen, theme } = useAppStore();
  const [activeItem, setActiveItem] = React.useState('dashboard');

  const sidebarVariants = {
    open: {
      x: 0,
      transition: {
        type: "spring",
        stiffness: 300,
        damping: 30
      }
    },
    closed: {
      x: -320,
      transition: {
        type: "spring",
        stiffness: 300,
        damping: 30
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, x: -20 },
    visible: (i: number) => ({
      opacity: 1,
      x: 0,
      transition: {
        delay: i * 0.05,
        duration: 0.3,
        ease: "easeOut"
      }
    })
  };

  return (
    <>
      {/* Backdrop */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 lg:hidden"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSidebarOpen(false)}
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <motion.div
        className="fixed left-0 top-0 h-full w-80 bg-black/20 backdrop-blur-xl border-r border-white/10 z-50 overflow-y-auto"
        variants={sidebarVariants}
        initial="closed"
        animate={sidebarOpen ? "open" : "closed"}
      >
        {/* Header */}
        <div className="p-6 border-b border-white/10">
          <motion.div
            className="flex items-center gap-3"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <div className="relative">
              <Brain className="w-8 h-8 text-neural-400" />
              <motion.div
                className="absolute inset-0 bg-neural-400 rounded-full blur-lg opacity-30"
                animate={{
                  scale: [1, 1.2, 1],
                  opacity: [0.3, 0.6, 0.3]
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
              />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">NeuroNexus</h1>
              <p className="text-xs text-white/60">Ultimate AI Platform</p>
            </div>
          </motion.div>
        </div>

        {/* Navigation */}
        <div className="p-4 space-y-2">
          {menuItems.map((item, index) => {
            const Icon = item.icon;
            const isActive = activeItem === item.id;
            
            return (
              <motion.div
                key={item.id}
                custom={index}
                variants={itemVariants}
                initial="hidden"
                animate="visible"
              >
                <motion.button
                  className={`
                    w-full p-3 rounded-xl text-left transition-all duration-300
                    flex items-center gap-3 group relative overflow-hidden
                    ${isActive 
                      ? 'bg-white/10 text-white border border-white/20' 
                      : 'text-white/70 hover:text-white hover:bg-white/5'
                    }
                  `}
                  onClick={() => setActiveItem(item.id)}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  {/* Background glow */}
                  {isActive && (
                    <motion.div
                      className="absolute inset-0 rounded-xl opacity-20"
                      style={{
                        background: `linear-gradient(135deg, ${item.color}40, transparent)`
                      }}
                      layoutId="activeBackground"
                      transition={{ type: "spring", stiffness: 300, damping: 30 }}
                    />
                  )}
                  
                  {/* Icon */}
                  <div className="relative">
                    <Icon 
                      className="w-5 h-5 transition-colors duration-300" 
                      style={{ color: isActive ? item.color : undefined }}
                    />
                    {isActive && (
                      <motion.div
                        className="absolute inset-0 blur-md opacity-50"
                        style={{ backgroundColor: item.color }}
                        animate={{
                          scale: [1, 1.5, 1],
                          opacity: [0.5, 0.8, 0.5]
                        }}
                        transition={{
                          duration: 2,
                          repeat: Infinity,
                          ease: "easeInOut"
                        }}
                      />
                    )}
                  </div>
                  
                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="font-medium truncate">{item.label}</div>
                    <div className="text-xs text-white/50 truncate">{item.description}</div>
                  </div>
                  
                  {/* Activity indicator */}
                  {isActive && (
                    <motion.div
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: item.color }}
                      animate={{
                        scale: [1, 1.2, 1],
                        opacity: [0.7, 1, 0.7]
                      }}
                      transition={{
                        duration: 1.5,
                        repeat: Infinity,
                        ease: "easeInOut"
                      }}
                    />
                  )}
                  
                  {/* Hover effect */}
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -skew-x-12 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
                </motion.button>
              </motion.div>
            );
          })}
        </div>

        {/* System Status */}
        <div className="p-4 mt-auto">
          <HolographicCard className="p-4" glowColor="#10b981">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-white/70">System Status</span>
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                  <span className="text-xs text-green-400">Online</span>
                </div>
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-white/60">Neural Load</span>
                  <span className="text-white">73%</span>
                </div>
                <div className="w-full bg-white/10 rounded-full h-1">
                  <motion.div
                    className="bg-gradient-to-r from-neural-400 to-neural-500 h-1 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: '73%' }}
                    transition={{ duration: 1, ease: "easeOut" }}
                  />
                </div>
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-white/60">Quantum Coherence</span>
                  <span className="text-white">91%</span>
                </div>
                <div className="w-full bg-white/10 rounded-full h-1">
                  <motion.div
                    className="bg-gradient-to-r from-nexus-400 to-nexus-500 h-1 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: '91%' }}
                    transition={{ duration: 1, delay: 0.2, ease: "easeOut" }}
                  />
                </div>
              </div>
            </div>
          </HolographicCard>
        </div>

        {/* Toggle Button */}
        <motion.button
          className="absolute -right-12 top-6 w-10 h-10 bg-black/20 backdrop-blur-xl border border-white/10 rounded-r-xl flex items-center justify-center text-white/70 hover:text-white transition-colors"
          onClick={() => setSidebarOpen(!sidebarOpen)}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <motion.div
            animate={{ rotate: sidebarOpen ? 180 : 0 }}
            transition={{ duration: 0.3 }}
          >
            <Zap className="w-5 h-5" />
          </motion.div>
        </motion.button>
      </motion.div>
    </>
  );
}