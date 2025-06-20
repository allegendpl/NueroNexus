import React, { useEffect, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Points, PointMaterial } from '@react-three/drei';
import * as THREE from 'three';

interface NeuralNetworkProps {
  count?: number;
  theme: 'neural' | 'quantum' | 'nexus' | 'cyber' | 'matrix';
}

function NeuralNetwork({ count = 5000, theme }: NeuralNetworkProps) {
  const ref = useRef<THREE.Points>(null);
  const [sphere] = React.useState(() => {
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    
    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      
      // Create neural network-like distribution
      const radius = Math.random() * 10 + 5;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.random() * Math.PI;
      
      positions[i3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i3 + 2] = radius * Math.cos(phi);
      
      // Theme-based colors
      const colorMap = {
        neural: [0.2, 0.6, 1.0],
        quantum: [0.8, 0.2, 0.9],
        nexus: [0.0, 0.8, 0.6],
        cyber: [1.0, 0.4, 0.0],
        matrix: [0.0, 1.0, 0.0]
      };
      
      const [r, g, b] = colorMap[theme];
      colors[i3] = r + Math.random() * 0.3;
      colors[i3 + 1] = g + Math.random() * 0.3;
      colors[i3 + 2] = b + Math.random() * 0.3;
    }
    
    return { positions, colors };
  });

  useFrame((state, delta) => {
    if (ref.current) {
      ref.current.rotation.x -= delta / 10;
      ref.current.rotation.y -= delta / 15;
      
      // Animate neural activity
      const positions = ref.current.geometry.attributes.position.array as Float32Array;
      const time = state.clock.getElapsedTime();
      
      for (let i = 0; i < positions.length; i += 3) {
        const x = positions[i];
        const y = positions[i + 1];
        const z = positions[i + 2];
        
        // Add neural pulse effect
        const pulse = Math.sin(time * 2 + (x + y + z) * 0.1) * 0.1;
        positions[i] = x + pulse;
        positions[i + 1] = y + pulse;
        positions[i + 2] = z + pulse;
      }
      
      ref.current.geometry.attributes.position.needsUpdate = true;
    }
  });

  return (
    <group rotation={[0, 0, Math.PI / 4]}>
      <Points ref={ref} positions={sphere.positions} colors={sphere.colors}>
        <PointMaterial
          transparent
          vertexColors
          size={0.05}
          sizeAttenuation={true}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </Points>
    </group>
  );
}

function QuantumField({ theme }: { theme: string }) {
  const meshRef = useRef<THREE.Mesh>(null);
  
  useFrame((state) => {
    if (meshRef.current) {
      const time = state.clock.getElapsedTime();
      meshRef.current.rotation.x = time * 0.1;
      meshRef.current.rotation.y = time * 0.05;
      
      // Quantum field distortion
      const geometry = meshRef.current.geometry as THREE.PlaneGeometry;
      const positions = geometry.attributes.position.array as Float32Array;
      
      for (let i = 0; i < positions.length; i += 3) {
        const x = positions[i];
        const y = positions[i + 1];
        positions[i + 2] = Math.sin(x * 0.1 + time) * Math.cos(y * 0.1 + time) * 2;
      }
      
      geometry.attributes.position.needsUpdate = true;
    }
  });

  return (
    <mesh ref={meshRef} position={[0, 0, -10]}>
      <planeGeometry args={[50, 50, 100, 100]} />
      <meshBasicMaterial
        color={theme === 'quantum' ? '#d946ef' : '#0ea5e9'}
        wireframe
        transparent
        opacity={0.1}
      />
    </mesh>
  );
}

export default function NeuralBackground({ theme }: { theme: 'neural' | 'quantum' | 'nexus' | 'cyber' | 'matrix' }) {
  return (
    <div className="fixed inset-0 -z-10">
      <Canvas
        camera={{ position: [0, 0, 30], fov: 60 }}
        style={{ background: 'transparent' }}
      >
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} />
        <NeuralNetwork theme={theme} />
        <QuantumField theme={theme} />
      </Canvas>
      
      {/* Matrix rain effect for matrix theme */}
      {theme === 'matrix' && (
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          {Array.from({ length: 50 }).map((_, i) => (
            <div
              key={i}
              className="absolute text-green-400 text-xs font-mono animate-matrix-rain opacity-20"
              style={{
                left: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 20}s`,
                animationDuration: `${15 + Math.random() * 10}s`,
              }}
            >
              {Array.from({ length: 20 }).map((_, j) => (
                <div key={j} className="block">
                  {String.fromCharCode(0x30A0 + Math.random() * 96)}
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
      
      {/* Cyber grid for cyber theme */}
      {theme === 'cyber' && (
        <div className="absolute inset-0 opacity-10">
          <div className="w-full h-full bg-gradient-to-br from-cyber-500 to-cyber-700">
            <div className="absolute inset-0 bg-[linear-gradient(90deg,transparent_24px,rgba(255,255,255,0.05)_25px,rgba(255,255,255,0.05)_26px,transparent_27px,transparent_74px,rgba(255,255,255,0.05)_75px,rgba(255,255,255,0.05)_76px,transparent_77px),linear-gradient(rgba(255,255,255,0.05)_24px,transparent_25px,transparent_26px,rgba(255,255,255,0.05)_27px,rgba(255,255,255,0.05)_74px,transparent_75px,transparent_76px,rgba(255,255,255,0.05)_77px)] bg-[length:100px_100px]" />
          </div>
        </div>
      )}
    </div>
  );
}