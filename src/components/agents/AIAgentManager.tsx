import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Plus, 
  Brain, 
  Cpu, 
  Eye, 
  MessageSquare, 
  Zap, 
  Settings,
  Play,
  Pause,
  Trash2,
  Edit,
  Activity,
  Network,
  Shield,
  Rocket
} from 'lucide-react';
import HolographicCard from '../ui/HolographicCard';
import QuantumButton from '../ui/QuantumButton';
import { useAppStore } from '../../store';
import type { AIAgent, AgentType } from '../../types';

const agentTypes: { type: AgentType; icon: React.ComponentType<any>; color: string; description: string }[] = [
  { type: 'conversational', icon: MessageSquare, color: '#0ea5e9', description: 'Advanced dialogue and communication' },
  { type: 'analytical', icon: Brain, color: '#d946ef', description: 'Data analysis and insights' },
  { type: 'creative', icon: Zap, color: '#10b981', description: 'Content creation and ideation' },
  { type: 'research', icon: Eye, color: '#f97316', description: 'Information gathering and synthesis' },
  { type: 'technical', icon: Cpu, color: '#8b5cf6', description: 'Code generation and debugging' },
  { type: 'neural', icon: Network, color: '#06b6d4', description: 'Deep learning and pattern recognition' },
  { type: 'quantum', icon: Activity, color: '#84cc16', description: 'Quantum computing and optimization' },
  { type: 'security', icon: Shield, color: '#ef4444', description: 'Cybersecurity and threat detection' },
];

interface AgentCardProps {
  agent: AIAgent;
  onEdit: (agent: AIAgent) => void;
  onDelete: (agentId: string) => void;
  onToggleStatus: (agentId: string) => void;
}

function AgentCard({ agent, onEdit, onDelete, onToggleStatus }: AgentCardProps) {
  const agentType = agentTypes.find(t => t.type === agent.type);
  const Icon = agentType?.icon || Cpu;
  const color = agentType?.color || '#0ea5e9';

  return (
    <HolographicCard glowColor={color} className="p-6">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
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
          <div>
            <h3 className="text-lg font-semibold text-white">{agent.name}</h3>
            <p className="text-sm text-white/60 capitalize">{agent.type} Agent</p>
          </div>
        </div>
        
        <div className={`px-2 py-1 rounded-full text-xs font-medium ${
          agent.status === 'active' ? 'bg-green-500/20 text-green-400' :
          agent.status === 'learning' ? 'bg-blue-500/20 text-blue-400' :
          agent.status === 'idle' ? 'bg-yellow-500/20 text-yellow-400' :
          'bg-red-500/20 text-red-400'
        }`}>
          {agent.status}
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <p className="text-xs text-white/50">Response Time</p>
          <p className="text-sm font-medium text-white">{agent.performance.responseTime}ms</p>
        </div>
        <div>
          <p className="text-xs text-white/50">Accuracy</p>
          <p className="text-sm font-medium text-white">{(agent.performance.accuracy * 100).toFixed(1)}%</p>
        </div>
        <div>
          <p className="text-xs text-white/50">Tasks Completed</p>
          <p className="text-sm font-medium text-white">{agent.performance.tasksCompleted}</p>
        </div>
        <div>
          <p className="text-xs text-white/50">User Satisfaction</p>
          <p className="text-sm font-medium text-white">{(agent.performance.userSatisfaction * 100).toFixed(1)}%</p>
        </div>
      </div>

      {/* Capabilities */}
      <div className="mb-4">
        <p className="text-xs text-white/50 mb-2">Capabilities</p>
        <div className="flex flex-wrap gap-1">
          {agent.capabilities.slice(0, 3).map((capability, index) => (
            <span
              key={index}
              className="px-2 py-1 bg-white/10 rounded-full text-xs text-white/70"
            >
              {capability.name}
            </span>
          ))}
          {agent.capabilities.length > 3 && (
            <span className="px-2 py-1 bg-white/10 rounded-full text-xs text-white/70">
              +{agent.capabilities.length - 3} more
            </span>
          )}
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        <QuantumButton
          size="sm"
          variant="primary"
          onClick={() => onToggleStatus(agent.id)}
          className="flex-1"
        >
          {agent.status === 'active' ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
          {agent.status === 'active' ? 'Pause' : 'Activate'}
        </QuantumButton>
        <QuantumButton
          size="sm"
          variant="secondary"
          onClick={() => onEdit(agent)}
        >
          <Edit className="w-3 h-3" />
        </QuantumButton>
        <QuantumButton
          size="sm"
          variant="secondary"
          onClick={() => onDelete(agent.id)}
        >
          <Trash2 className="w-3 h-3" />
        </QuantumButton>
      </div>
    </HolographicCard>
  );
}

export default function AIAgentManager() {
  const { agents, addAgent } = useAppStore();
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedType, setSelectedType] = useState<AgentType>('conversational');
  const [agentName, setAgentName] = useState('');

  const handleCreateAgent = () => {
    if (!agentName.trim()) return;

    const newAgent: AIAgent = {
      id: `agent-${Date.now()}`,
      name: agentName,
      type: selectedType,
      capabilities: [
        { name: 'Natural Language Processing', level: 85, description: 'Advanced text understanding', prerequisites: [], applications: [] },
        { name: 'Machine Learning', level: 78, description: 'Pattern recognition and learning', prerequisites: [], applications: [] },
        { name: 'Data Analysis', level: 92, description: 'Statistical analysis and insights', prerequisites: [], applications: [] },
      ],
      personality: {
        name: 'Professional Assistant',
        traits: ['analytical', 'helpful', 'precise'],
        expertise: ['technology', 'business', 'science'],
        communicationStyle: 'professional',
        knowledgeDepth: 85,
        creativity: 70,
        analyticalThinking: 90,
        emotionalIntelligence: 75,
      },
      knowledgeBase: {
        domains: [],
        totalFacts: 0,
        lastUpdated: new Date(),
        accuracy: 0.95,
        coverage: 0.80,
      },
      neuralNetwork: {
        layers: [
          { type: 'input', neurons: 512, activation: 'relu' },
          { type: 'hidden', neurons: 256, activation: 'relu', dropout: 0.2 },
          { type: 'hidden', neurons: 128, activation: 'relu', dropout: 0.2 },
          { type: 'output', neurons: 64, activation: 'softmax' },
        ],
        weights: [],
        biases: [],
        activationFunction: 'relu',
        learningRate: 0.001,
        epochs: 100,
        accuracy: 0.92,
        loss: 0.08,
      },
      status: 'idle',
      performance: {
        responseTime: Math.floor(Math.random() * 500) + 100,
        accuracy: 0.85 + Math.random() * 0.15,
        userSatisfaction: 0.80 + Math.random() * 0.20,
        tasksCompleted: Math.floor(Math.random() * 1000),
        errorsEncountered: Math.floor(Math.random() * 10),
        learningRate: 0.001 + Math.random() * 0.009,
        adaptabilityScore: 0.75 + Math.random() * 0.25,
      },
    };

    addAgent(newAgent);
    setAgentName('');
    setShowCreateModal(false);
  };

  const handleToggleStatus = (agentId: string) => {
    // Implementation would update agent status
    console.log('Toggle status for agent:', agentId);
  };

  const handleEditAgent = (agent: AIAgent) => {
    // Implementation would open edit modal
    console.log('Edit agent:', agent);
  };

  const handleDeleteAgent = (agentId: string) => {
    // Implementation would remove agent
    console.log('Delete agent:', agentId);
  };

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
          <h1 className="text-3xl font-bold text-white mb-2">AI Agent Command Center</h1>
          <p className="text-white/60">Manage and deploy intelligent AI agents</p>
        </div>
        
        <QuantumButton
          variant="quantum"
          size="md"
          onClick={() => setShowCreateModal(true)}
        >
          <Plus className="w-4 h-4" />
          Create Agent
        </QuantumButton>
      </motion.div>

      {/* Stats */}
      <motion.div
        className="grid grid-cols-1 md:grid-cols-4 gap-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        {[
          { label: 'Total Agents', value: agents.length, color: '#0ea5e9', icon: Cpu },
          { label: 'Active Agents', value: agents.filter(a => a.status === 'active').length, color: '#10b981', icon: Activity },
          { label: 'Learning Agents', value: agents.filter(a => a.status === 'learning').length, color: '#f97316', icon: Brain },
          { label: 'Avg Performance', value: '94.2%', color: '#d946ef', icon: Zap },
        ].map((stat, index) => {
          const Icon = stat.icon;
          return (
            <HolographicCard key={stat.label} glowColor={stat.color} className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-white/60 text-sm">{stat.label}</p>
                  <p className="text-2xl font-bold text-white">{stat.value}</p>
                </div>
                <Icon className="w-8 h-8" style={{ color: stat.color }} />
              </div>
            </HolographicCard>
          );
        })}
      </motion.div>

      {/* Agents Grid */}
      <motion.div
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        {agents.map((agent, index) => (
          <motion.div
            key={agent.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
          >
            <AgentCard
              agent={agent}
              onEdit={handleEditAgent}
              onDelete={handleDeleteAgent}
              onToggleStatus={handleToggleStatus}
            />
          </motion.div>
        ))}
      </motion.div>

      {/* Create Agent Modal */}
      <AnimatePresence>
        {showCreateModal && (
          <motion.div
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setShowCreateModal(false)}
          >
            <motion.div
              className="bg-black/20 backdrop-blur-xl border border-white/10 rounded-2xl p-6 w-full max-w-2xl"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
            >
              <h2 className="text-2xl font-bold text-white mb-6">Create New AI Agent</h2>
              
              <div className="space-y-6">
                <div>
                  <label className="block text-white/70 text-sm font-medium mb-2">Agent Name</label>
                  <input
                    type="text"
                    value={agentName}
                    onChange={(e) => setAgentName(e.target.value)}
                    className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-xl text-white placeholder-white/50 focus:outline-none focus:border-neural-400"
                    placeholder="Enter agent name..."
                  />
                </div>

                <div>
                  <label className="block text-white/70 text-sm font-medium mb-4">Agent Type</label>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {agentTypes.map((type) => {
                      const Icon = type.icon;
                      return (
                        <motion.button
                          key={type.type}
                          className={`p-4 rounded-xl border transition-all ${
                            selectedType === type.type
                              ? 'border-white/30 bg-white/10'
                              : 'border-white/10 bg-white/5 hover:bg-white/10'
                          }`}
                          onClick={() => setSelectedType(type.type)}
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                        >
                          <Icon className="w-6 h-6 mx-auto mb-2" style={{ color: type.color }} />
                          <p className="text-sm font-medium text-white capitalize">{type.type}</p>
                          <p className="text-xs text-white/50 mt-1">{type.description}</p>
                        </motion.button>
                      );
                    })}
                  </div>
                </div>

                <div className="flex gap-3 pt-4">
                  <QuantumButton
                    variant="secondary"
                    size="md"
                    onClick={() => setShowCreateModal(false)}
                    className="flex-1"
                  >
                    Cancel
                  </QuantumButton>
                  <QuantumButton
                    variant="quantum"
                    size="md"
                    onClick={handleCreateAgent}
                    className="flex-1"
                    disabled={!agentName.trim()}
                  >
                    <Rocket className="w-4 h-4" />
                    Create Agent
                  </QuantumButton>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}