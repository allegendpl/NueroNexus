import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Toaster } from 'react-hot-toast';
import { useAppStore } from './store';
import NeuralBackground from './components/ui/NeuralBackground';
import Sidebar from './components/layout/Sidebar';
import NeuralDashboard from './components/dashboard/NeuralDashboard';
import AIAgentManager from './components/agents/AIAgentManager';
import './index.css';

function App() {
  const { theme, sidebarOpen, updateNeuralActivity, updateQuantumCoherence } = useAppStore();

  // Simulate neural activity and quantum coherence updates
  useEffect(() => {
    const interval = setInterval(() => {
      const newActivity = Array.from({ length: 100 }, () => Math.random());
      updateNeuralActivity(newActivity);
      updateQuantumCoherence(Math.random());
    }, 1000);

    return () => clearInterval(interval);
  }, [updateNeuralActivity, updateQuantumCoherence]);

  return (
    <Router>
      <div className={`min-h-screen bg-gradient-to-br ${
        theme === 'neural' ? 'from-slate-900 via-blue-900 to-slate-900' :
        theme === 'quantum' ? 'from-purple-900 via-pink-900 to-purple-900' :
        theme === 'nexus' ? 'from-emerald-900 via-teal-900 to-emerald-900' :
        theme === 'cyber' ? 'from-orange-900 via-red-900 to-orange-900' :
        'from-black via-green-900 to-black'
      } relative overflow-hidden`}>
        
        {/* Neural Background */}
        <NeuralBackground theme={theme} />
        
        {/* Sidebar */}
        <Sidebar />
        
        {/* Main Content */}
        <motion.div
          className={`transition-all duration-300 ${
            sidebarOpen ? 'lg:ml-80' : 'ml-0'
          }`}
          layout
        >
          <AnimatePresence mode="wait">
            <Routes>
              <Route path="/" element={<NeuralDashboard />} />
              <Route path="/agents" element={<AIAgentManager />} />
              <Route path="/conversations" element={<div className="p-6"><h1 className="text-white text-2xl">Quantum Chat Coming Soon</h1></div>} />
              <Route path="/neural-network" element={<div className="p-6"><h1 className="text-white text-2xl">Neural Networks Coming Soon</h1></div>} />
              <Route path="/quantum-lab" element={<div className="p-6"><h1 className="text-white text-2xl">Quantum Lab Coming Soon</h1></div>} />
              <Route path="/analytics" element={<div className="p-6"><h1 className="text-white text-2xl">Cyber Analytics Coming Soon</h1></div>} />
              <Route path="/vision" element={<div className="p-6"><h1 className="text-white text-2xl">Computer Vision Coming Soon</h1></div>} />
              <Route path="/security" element={<div className="p-6"><h1 className="text-white text-2xl">AI Security Coming Soon</h1></div>} />
              <Route path="/deployment" element={<div className="p-6"><h1 className="text-white text-2xl">Model Deploy Coming Soon</h1></div>} />
              <Route path="/data-lake" element={<div className="p-6"><h1 className="text-white text-2xl">Data Lake Coming Soon</h1></div>} />
              <Route path="/code-gen" element={<div className="p-6"><h1 className="text-white text-2xl">Code Genesis Coming Soon</h1></div>} />
              <Route path="/model-zoo" element={<div className="p-6"><h1 className="text-white text-2xl">Model Zoo Coming Soon</h1></div>} />
              <Route path="/pipeline" element={<div className="p-6"><h1 className="text-white text-2xl">ML Pipeline Coming Soon</h1></div>} />
              <Route path="/monitoring" element={<div className="p-6"><h1 className="text-white text-2xl">System Monitor Coming Soon</h1></div>} />
            </Routes>
          </AnimatePresence>
        </motion.div>
        
        {/* Toast Notifications */}
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: 'rgba(0, 0, 0, 0.8)',
              color: '#fff',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              backdropFilter: 'blur(10px)',
            },
          }}
        />
      </div>
    </Router>
  );
}

export default App;