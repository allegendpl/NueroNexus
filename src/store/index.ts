import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { 
  User, 
  AIAgent, 
  Conversation, 
  Task, 
  SystemMetrics, 
  Plugin,
  NeuralProfile,
  QuantumState 
} from '../types';

interface AppState {
  // User Management
  user: User | null;
  isAuthenticated: boolean;
  
  // AI Agents
  agents: AIAgent[];
  activeAgent: AIAgent | null;
  
  // Conversations
  conversations: Conversation[];
  activeConversation: Conversation | null;
  
  // Tasks
  tasks: Task[];
  activeTasks: Task[];
  
  // System
  systemMetrics: SystemMetrics | null;
  plugins: Plugin[];
  
  // UI State
  theme: 'neural' | 'quantum' | 'nexus' | 'cyber' | 'matrix';
  sidebarOpen: boolean;
  loading: boolean;
  error: string | null;
  
  // Neural Network State
  neuralActivity: number[];
  quantumCoherence: number;
  
  // Actions
  setUser: (user: User | null) => void;
  setAuthenticated: (authenticated: boolean) => void;
  addAgent: (agent: AIAgent) => void;
  setActiveAgent: (agent: AIAgent | null) => void;
  addConversation: (conversation: Conversation) => void;
  setActiveConversation: (conversation: Conversation | null) => void;
  addTask: (task: Task) => void;
  updateTask: (taskId: string, updates: Partial<Task>) => void;
  setSystemMetrics: (metrics: SystemMetrics) => void;
  addPlugin: (plugin: Plugin) => void;
  setTheme: (theme: 'neural' | 'quantum' | 'nexus' | 'cyber' | 'matrix') => void;
  setSidebarOpen: (open: boolean) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  updateNeuralActivity: (activity: number[]) => void;
  updateQuantumCoherence: (coherence: number) => void;
}

export const useAppStore = create<AppState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial State
        user: null,
        isAuthenticated: false,
        agents: [],
        activeAgent: null,
        conversations: [],
        activeConversation: null,
        tasks: [],
        activeTasks: [],
        systemMetrics: null,
        plugins: [],
        theme: 'neural',
        sidebarOpen: true,
        loading: false,
        error: null,
        neuralActivity: Array(100).fill(0),
        quantumCoherence: 0,

        // Actions
        setUser: (user) => set({ user }),
        setAuthenticated: (authenticated) => set({ isAuthenticated: authenticated }),
        
        addAgent: (agent) => set((state) => ({ 
          agents: [...state.agents, agent] 
        })),
        
        setActiveAgent: (agent) => set({ activeAgent: agent }),
        
        addConversation: (conversation) => set((state) => ({ 
          conversations: [...state.conversations, conversation] 
        })),
        
        setActiveConversation: (conversation) => set({ activeConversation: conversation }),
        
        addTask: (task) => set((state) => ({ 
          tasks: [...state.tasks, task],
          activeTasks: task.status === 'in-progress' 
            ? [...state.activeTasks, task] 
            : state.activeTasks
        })),
        
        updateTask: (taskId, updates) => set((state) => ({
          tasks: state.tasks.map(task => 
            task.id === taskId ? { ...task, ...updates } : task
          ),
          activeTasks: state.activeTasks.map(task => 
            task.id === taskId ? { ...task, ...updates } : task
          )
        })),
        
        setSystemMetrics: (metrics) => set({ systemMetrics: metrics }),
        
        addPlugin: (plugin) => set((state) => ({ 
          plugins: [...state.plugins, plugin] 
        })),
        
        setTheme: (theme) => set({ theme }),
        setSidebarOpen: (open) => set({ sidebarOpen: open }),
        setLoading: (loading) => set({ loading }),
        setError: (error) => set({ error }),
        
        updateNeuralActivity: (activity) => set({ neuralActivity: activity }),
        updateQuantumCoherence: (coherence) => set({ quantumCoherence: coherence }),
      }),
      {
        name: 'neuronexus-storage',
        partialize: (state) => ({
          user: state.user,
          isAuthenticated: state.isAuthenticated,
          theme: state.theme,
          sidebarOpen: state.sidebarOpen,
        }),
      }
    ),
    {
      name: 'NeuroNexus Store',
    }
  )
);

// Quantum State Store
interface QuantumStore {
  entanglements: Map<string, number>;
  superpositions: Map<string, any[]>;
  coherenceMatrix: number[][];
  quantumGates: string[];
  
  createEntanglement: (id1: string, id2: string, strength: number) => void;
  addSuperposition: (id: string, states: any[]) => void;
  updateCoherence: (matrix: number[][]) => void;
  applyQuantumGate: (gate: string) => void;
  measureQuantumState: (id: string) => any;
}

export const useQuantumStore = create<QuantumStore>((set, get) => ({
  entanglements: new Map(),
  superpositions: new Map(),
  coherenceMatrix: [],
  quantumGates: [],
  
  createEntanglement: (id1, id2, strength) => set((state) => {
    const newEntanglements = new Map(state.entanglements);
    newEntanglements.set(`${id1}-${id2}`, strength);
    return { entanglements: newEntanglements };
  }),
  
  addSuperposition: (id, states) => set((state) => {
    const newSuperpositions = new Map(state.superpositions);
    newSuperpositions.set(id, states);
    return { superpositions: newSuperpositions };
  }),
  
  updateCoherence: (matrix) => set({ coherenceMatrix: matrix }),
  
  applyQuantumGate: (gate) => set((state) => ({
    quantumGates: [...state.quantumGates, gate]
  })),
  
  measureQuantumState: (id) => {
    const state = get();
    const superposition = state.superpositions.get(id);
    if (!superposition) return null;
    
    // Quantum measurement collapses superposition
    const randomIndex = Math.floor(Math.random() * superposition.length);
    return superposition[randomIndex];
  },
}));

// Neural Network Store
interface NeuralStore {
  networks: Map<string, any>;
  activations: Map<string, number[]>;
  weights: Map<string, number[][]>;
  learningRates: Map<string, number>;
  
  createNetwork: (id: string, config: any) => void;
  updateActivations: (id: string, activations: number[]) => void;
  updateWeights: (id: string, weights: number[][]) => void;
  trainNetwork: (id: string, data: any[]) => void;
  predict: (id: string, input: any) => any;
}

export const useNeuralStore = create<NeuralStore>((set, get) => ({
  networks: new Map(),
  activations: new Map(),
  weights: new Map(),
  learningRates: new Map(),
  
  createNetwork: (id, config) => set((state) => {
    const newNetworks = new Map(state.networks);
    newNetworks.set(id, config);
    return { networks: newNetworks };
  }),
  
  updateActivations: (id, activations) => set((state) => {
    const newActivations = new Map(state.activations);
    newActivations.set(id, activations);
    return { activations: newActivations };
  }),
  
  updateWeights: (id, weights) => set((state) => {
    const newWeights = new Map(state.weights);
    newWeights.set(id, weights);
    return { weights: newWeights };
  }),
  
  trainNetwork: (id, data) => {
    // Implement neural network training logic
    console.log(`Training network ${id} with data:`, data);
  },
  
  predict: (id, input) => {
    const state = get();
    const network = state.networks.get(id);
    if (!network) return null;
    
    // Implement prediction logic
    return Math.random(); // Placeholder
  },
}));