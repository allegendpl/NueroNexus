import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Brain, 
  Cpu, 
  Zap, 
  Activity, 
  Network, 
  Eye, 
  MessageSquare,
  BarChart3,
  Atom,
  Shield,
  Rocket,
  Database
} from 'lucide-react';
import HolographicCard from '../ui/HolographicCard';
import QuantumButton from '../ui/QuantumButton';
import NeuralVisualization from '../ui/NeuralVisualization';
import { useAppStore } from '../../store';

interface MetricCardProps {
  title: string;
  value: string;
  change: string;
  icon: React.ComponentType<any>;
  color: string;
  trend: number[];
}

function MetricCard({ title, value, change, icon: Icon, color, trend }: MetricCardProps) {
  return (
    <HolographicCard glowColor={color} className="p-6">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-white/60 text-sm font-medium">{title}</p>
          <p className="text-2xl font-bold text-white mt-1">{value}</p>
          <p className={`text-sm mt-1 ${change.startsWith('+') ? 'text-green-400' : 'text-red-400'}`}>
            {change}
          </p>
        </div>
        <div className="relative">
          <Icon className="w-8 h-8" style={{ color }} />
          <motion.div
            className="absolute inset-0 blur-lg opacity-30"
            style={{ backgroundColor: color }}
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
      </div>
      
      {/* Mini trend chart */}
      <div className="mt-4 h-8 flex items-end gap-1">
        {trend.map((value, index) => (
          <motion.div
            key={index}
            className="flex-1 rounded-t"
            style={{ backgroundColor: color }}
            initial={{ height: 0 }}
            animate={{ height: `${value}%` }}
            transition={{ delay: index * 0.1, duration: 0.5 }}
          />
        ))}
      </div>
    </HolographicCard>
  );
}

export default function NeuralDashboard() {
  const { theme, neuralActivity, quantumCoherence } = useAppStore();
  const [realTimeData, setRealTimeData] = useState<number[]>([]);
  const [systemMetrics, setSystemMetrics] = useState({
    cpuUsage: 0,
    memoryUsage: 0,
    networkActivity: 0,
    activeAgents: 0,
    tasksCompleted: 0,
    quantumEntanglements: 0
  });

  // Simulate real-time data updates
  useEffect(() => {
    const interval = setInterval(() => {
      setRealTimeData(prev => {
        const newData = [...prev];
        newData.push(Math.random());
        if (newData.length > 100) newData.shift();
        return newData;
      });

      setSystemMetrics(prev => ({
        cpuUsage: Math.max(0, Math.min(100, prev.cpuUsage + (Math.random() - 0.5) * 10)),
        memoryUsage: Math.max(0, Math.min(100, prev.memoryUsage + (Math.random() - 0.5) * 5)),
        networkActivity: Math.max(0, Math.min(100, prev.networkActivity + (Math.random() - 0.5) * 20)),
        activeAgents: Math.max(0, prev.activeAgents + Math.floor((Math.random() - 0.5) * 3)),
        tasksCompleted: prev.tasksCompleted + Math.floor(Math.random() * 3),
        quantumEntanglements: Math.max(0, prev.quantumEntanglements + Math.floor((Math.random() - 0.5) * 2))
      }));
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const metrics = [
    {
      title: 'Neural Processing',
      value: `${systemMetrics.cpuUsage.toFixed(1)}%`,
      change: '+12.5%',
      icon: Brain,
      color: '#0ea5e9',
      trend: [65, 70, 68, 75, 80, 78, 85, 82, 88, 85]
    },
    {
      title: 'Active AI Agents',
      value: systemMetrics.activeAgents.toString(),
      change: '+3',
      icon: Cpu,
      color: '#d946ef',
      trend: [45, 50, 48, 55, 60, 58, 65, 62, 68, 65]
    },
    {
      title: 'Quantum Coherence',
      value: `${(quantumCoherence * 100).toFixed(1)}%`,
      change: '+8.2%',
      icon: Atom,
      color: '#10b981',
      trend: [70, 75, 73, 80, 85, 83, 90, 87, 93, 90]
    },
    {
      title: 'Tasks Completed',
      value: systemMetrics.tasksCompleted.toString(),
      change: '+156',
      icon: Activity,
      color: '#f97316',
      trend: [30, 35, 33, 40, 45, 43, 50, 47, 53, 50]
    },
    {
      title: 'Network Activity',
      value: `${systemMetrics.networkActivity.toFixed(1)} Gb/s`,
      change: '+24.1%',
      icon: Network,
      color: '#8b5cf6',
      trend: [55, 60, 58, 65, 70, 68, 75, 72, 78, 75]
    },
    {
      title: 'Memory Usage',
      value: `${systemMetrics.memoryUsage.toFixed(1)}%`,
      change: '-2.3%',
      icon: Database,
      color: '#06b6d4',
      trend: [80, 75, 77, 70, 65, 67, 60, 63, 57, 60]
    }
  ];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <motion.div
        className="flex items-center justify-between"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Neural Command Center</h1>
          <p className="text-white/60">Real-time AI system monitoring and control</p>
        </div>
        
        <div className="flex gap-3">
          <QuantumButton variant="quantum" size="md">
            <Rocket className="w-4 h-4" />
            Deploy Model
          </QuantumButton>
          <QuantumButton variant="neural" size="md">
            <Shield className="w-4 h-4" />
            Security Scan
          </QuantumButton>
        </div>
      </motion.div>

      {/* Metrics Grid */}
      <motion.div
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        {metrics.map((metric, index) => (
          <motion.div
            key={metric.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
          >
            <MetricCard {...metric} />
          </motion.div>
        ))}
      </motion.div>

      {/* Neural Visualization Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <HolographicCard className="p-6" glowColor="#0ea5e9">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold text-white">Neural Network Activity</h3>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-neural-400 rounded-full animate-pulse" />
                <span className="text-sm text-white/60">Live</span>
              </div>
            </div>
            <NeuralVisualization 
              data={realTimeData} 
              theme={theme}
              width={400}
              height={250}
            />
          </HolographicCard>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.6 }}
        >
          <HolographicCard className="p-6" glowColor="#d946ef">
            <h3 className="text-xl font-semibold text-white mb-4">Quantum Entanglement Matrix</h3>
            <div className="space-y-4">
              {/* Quantum state visualization */}
              <div className="grid grid-cols-8 gap-1">
                {Array.from({ length: 64 }).map((_, i) => (
                  <motion.div
                    key={i}
                    className="aspect-square rounded-sm"
                    style={{
                      backgroundColor: `hsl(${(i * 5.625 + Date.now() / 100) % 360}, 70%, 50%)`
                    }}
                    animate={{
                      opacity: [0.3, 1, 0.3],
                      scale: [0.8, 1.2, 0.8]
                    }}
                    transition={{
                      duration: 2,
                      delay: i * 0.02,
                      repeat: Infinity,
                      ease: "easeInOut"
                    }}
                  />
                ))}
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-white/60">Entanglement Strength</span>
                  <span className="text-white">94.7%</span>
                </div>
                <div className="w-full bg-white/10 rounded-full h-2">
                  <motion.div
                    className="bg-gradient-to-r from-nexus-400 to-nexus-500 h-2 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: '94.7%' }}
                    transition={{ duration: 1.5, ease: "easeOut" }}
                  />
                </div>
              </div>
            </div>
          </HolographicCard>
        </motion.div>
      </div>

      {/* System Alerts */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.8 }}
      >
        <HolographicCard className="p-6" glowColor="#10b981">
          <h3 className="text-xl font-semibold text-white mb-4">System Intelligence Feed</h3>
          <div className="space-y-3">
            {[
              { type: 'success', message: 'Neural network training completed successfully', time: '2 min ago' },
              { type: 'info', message: 'Quantum entanglement established with Agent-7', time: '5 min ago' },
              { type: 'warning', message: 'High memory usage detected in Vision module', time: '8 min ago' },
              { type: 'success', message: 'New AI model deployed to production', time: '12 min ago' },
            ].map((alert, index) => (
              <motion.div
                key={index}
                className="flex items-start gap-3 p-3 rounded-lg bg-white/5 border border-white/10"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <div className={`w-2 h-2 rounded-full mt-2 ${
                  alert.type === 'success' ? 'bg-green-400' :
                  alert.type === 'warning' ? 'bg-yellow-400' :
                  'bg-blue-400'
                } animate-pulse`} />
                <div className="flex-1">
                  <p className="text-white text-sm">{alert.message}</p>
                  <p className="text-white/50 text-xs mt-1">{alert.time}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </HolographicCard>
      </motion.div>
    </div>
  );
}