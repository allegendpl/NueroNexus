export interface User {
  id: string;
  username: string;
  email: string;
  avatar?: string;
  level: number;
  experience: number;
  badges: Badge[];
  preferences: UserPreferences;
  neuralProfile: NeuralProfile;
  quantumState: QuantumState;
}

export interface Badge {
  id: string;
  name: string;
  description: string;
  icon: string;
  rarity: 'common' | 'rare' | 'epic' | 'legendary' | 'mythic';
  unlockedAt: Date;
  nftTokenId?: string;
}

export interface UserPreferences {
  theme: 'neural' | 'quantum' | 'nexus' | 'cyber' | 'matrix';
  language: string;
  voiceSettings: VoiceSettings;
  aiPersonality: AIPersonality;
  privacyLevel: number;
  notifications: NotificationSettings;
}

export interface VoiceSettings {
  enabled: boolean;
  voice: string;
  speed: number;
  pitch: number;
  volume: number;
  language: string;
}

export interface AIPersonality {
  name: string;
  traits: string[];
  expertise: string[];
  communicationStyle: 'formal' | 'casual' | 'technical' | 'creative' | 'empathetic';
  knowledgeDepth: number;
  creativity: number;
  analyticalThinking: number;
  emotionalIntelligence: number;
}

export interface NotificationSettings {
  email: boolean;
  push: boolean;
  inApp: boolean;
  frequency: 'immediate' | 'hourly' | 'daily' | 'weekly';
}

export interface NeuralProfile {
  cognitiveMap: CognitiveMap;
  learningPatterns: LearningPattern[];
  knowledgeGraph: KnowledgeNode[];
  memoryPalace: MemoryPalace;
  thoughtVectors: ThoughtVector[];
}

export interface CognitiveMap {
  strengths: string[];
  weaknesses: string[];
  learningStyle: 'visual' | 'auditory' | 'kinesthetic' | 'reading' | 'multimodal';
  processingSpeed: number;
  attentionSpan: number;
  creativityIndex: number;
  logicalReasoningScore: number;
}

export interface LearningPattern {
  id: string;
  pattern: string;
  frequency: number;
  effectiveness: number;
  contexts: string[];
  timestamp: Date;
}

export interface KnowledgeNode {
  id: string;
  concept: string;
  connections: string[];
  strength: number;
  lastAccessed: Date;
  category: string;
  metadata: Record<string, any>;
}

export interface MemoryPalace {
  rooms: MemoryRoom[];
  totalCapacity: number;
  usedCapacity: number;
  efficiency: number;
}

export interface MemoryRoom {
  id: string;
  name: string;
  theme: string;
  memories: Memory[];
  capacity: number;
  accessFrequency: number;
}

export interface Memory {
  id: string;
  content: string;
  type: 'fact' | 'experience' | 'skill' | 'emotion' | 'association';
  importance: number;
  vividness: number;
  lastRecalled: Date;
  associations: string[];
}

export interface ThoughtVector {
  id: string;
  vector: number[];
  concept: string;
  context: string;
  timestamp: Date;
  similarity: number;
}

export interface QuantumState {
  entanglements: QuantumEntanglement[];
  superpositions: Superposition[];
  coherenceLevel: number;
  quantumAdvantage: number;
  parallelProcessing: ParallelProcess[];
}

export interface QuantumEntanglement {
  id: string;
  partnerId: string;
  strength: number;
  type: 'knowledge' | 'emotion' | 'goal' | 'experience';
  established: Date;
  lastInteraction: Date;
}

export interface Superposition {
  id: string;
  states: QuantumStateVector[];
  probability: number;
  collapsed: boolean;
  observationCount: number;
}

export interface QuantumStateVector {
  state: string;
  amplitude: number;
  phase: number;
}

export interface ParallelProcess {
  id: string;
  task: string;
  threads: ProcessThread[];
  efficiency: number;
  startTime: Date;
  estimatedCompletion: Date;
}

export interface ProcessThread {
  id: string;
  status: 'running' | 'waiting' | 'completed' | 'error';
  progress: number;
  result?: any;
}

export interface AIAgent {
  id: string;
  name: string;
  type: AgentType;
  capabilities: Capability[];
  personality: AIPersonality;
  knowledgeBase: KnowledgeBase;
  neuralNetwork: NeuralNetwork;
  status: 'active' | 'learning' | 'idle' | 'maintenance';
  performance: PerformanceMetrics;
}

export type AgentType = 
  | 'conversational'
  | 'analytical'
  | 'creative'
  | 'research'
  | 'coaching'
  | 'technical'
  | 'emotional'
  | 'strategic'
  | 'quantum'
  | 'neural';

export interface Capability {
  name: string;
  level: number;
  description: string;
  prerequisites: string[];
  applications: string[];
}

export interface KnowledgeBase {
  domains: KnowledgeDomain[];
  totalFacts: number;
  lastUpdated: Date;
  accuracy: number;
  coverage: number;
}

export interface KnowledgeDomain {
  name: string;
  facts: Fact[];
  confidence: number;
  sources: Source[];
  lastVerified: Date;
}

export interface Fact {
  id: string;
  statement: string;
  confidence: number;
  sources: string[];
  verified: boolean;
  category: string;
  tags: string[];
}

export interface Source {
  id: string;
  url: string;
  title: string;
  credibility: number;
  lastChecked: Date;
  type: 'academic' | 'news' | 'blog' | 'official' | 'community';
}

export interface NeuralNetwork {
  layers: NetworkLayer[];
  weights: number[][][];
  biases: number[][];
  activationFunction: string;
  learningRate: number;
  epochs: number;
  accuracy: number;
  loss: number;
}

export interface NetworkLayer {
  type: 'input' | 'hidden' | 'output' | 'convolutional' | 'recurrent' | 'attention';
  neurons: number;
  activation: string;
  dropout?: number;
}

export interface PerformanceMetrics {
  responseTime: number;
  accuracy: number;
  userSatisfaction: number;
  tasksCompleted: number;
  errorsEncountered: number;
  learningRate: number;
  adaptabilityScore: number;
}

export interface Conversation {
  id: string;
  userId: string;
  agentId: string;
  messages: Message[];
  context: ConversationContext;
  metadata: ConversationMetadata;
  status: 'active' | 'paused' | 'completed' | 'archived';
}

export interface Message {
  id: string;
  content: string;
  type: 'text' | 'voice' | 'image' | 'video' | 'file' | 'code' | 'data';
  sender: 'user' | 'agent' | 'system';
  timestamp: Date;
  metadata: MessageMetadata;
  reactions: Reaction[];
  attachments: Attachment[];
}

export interface MessageMetadata {
  sentiment: number;
  confidence: number;
  intent: string;
  entities: Entity[];
  keywords: string[];
  language: string;
  processingTime: number;
}

export interface Entity {
  text: string;
  type: string;
  confidence: number;
  startIndex: number;
  endIndex: number;
}

export interface Reaction {
  type: string;
  userId: string;
  timestamp: Date;
}

export interface Attachment {
  id: string;
  name: string;
  type: string;
  size: number;
  url: string;
  metadata: Record<string, any>;
}

export interface ConversationContext {
  topic: string;
  goals: string[];
  currentFocus: string;
  emotionalState: EmotionalState;
  knowledgeLevel: number;
  preferences: Record<string, any>;
}

export interface EmotionalState {
  primary: string;
  secondary: string[];
  intensity: number;
  valence: number;
  arousal: number;
  confidence: number;
}

export interface ConversationMetadata {
  startTime: Date;
  lastActivity: Date;
  messageCount: number;
  averageResponseTime: number;
  satisfactionScore: number;
  tags: string[];
  category: string;
}

export interface Task {
  id: string;
  title: string;
  description: string;
  type: TaskType;
  priority: 'low' | 'medium' | 'high' | 'critical';
  status: 'pending' | 'in-progress' | 'completed' | 'failed' | 'cancelled';
  assignedAgent: string;
  requiredCapabilities: string[];
  dependencies: string[];
  subtasks: SubTask[];
  progress: number;
  estimatedDuration: number;
  actualDuration?: number;
  createdAt: Date;
  updatedAt: Date;
  completedAt?: Date;
  result?: TaskResult;
}

export type TaskType = 
  | 'analysis'
  | 'research'
  | 'creation'
  | 'optimization'
  | 'learning'
  | 'communication'
  | 'problem-solving'
  | 'decision-making'
  | 'planning'
  | 'execution';

export interface SubTask {
  id: string;
  title: string;
  status: 'pending' | 'in-progress' | 'completed' | 'failed';
  progress: number;
  assignedTo?: string;
}

export interface TaskResult {
  success: boolean;
  data: any;
  confidence: number;
  alternatives: Alternative[];
  recommendations: string[];
  metadata: Record<string, any>;
}

export interface Alternative {
  option: string;
  confidence: number;
  pros: string[];
  cons: string[];
  impact: number;
}

export interface AnalyticsEvent {
  id: string;
  type: string;
  userId: string;
  sessionId: string;
  timestamp: Date;
  data: Record<string, any>;
  context: EventContext;
}

export interface EventContext {
  page: string;
  userAgent: string;
  location: string;
  referrer: string;
  sessionDuration: number;
  previousEvents: string[];
}

export interface SystemMetrics {
  cpu: number;
  memory: number;
  network: number;
  storage: number;
  activeUsers: number;
  activeAgents: number;
  tasksInQueue: number;
  averageResponseTime: number;
  errorRate: number;
  uptime: number;
}

export interface Plugin {
  id: string;
  name: string;
  version: string;
  description: string;
  author: string;
  capabilities: string[];
  dependencies: string[];
  configuration: PluginConfig;
  status: 'active' | 'inactive' | 'error' | 'updating';
  permissions: Permission[];
}

export interface PluginConfig {
  settings: Record<string, any>;
  apiKeys: Record<string, string>;
  endpoints: Record<string, string>;
  features: Record<string, boolean>;
}

export interface Permission {
  type: string;
  scope: string;
  granted: boolean;
  grantedAt?: Date;
  grantedBy?: string;
}

export interface WebhookEvent {
  id: string;
  type: string;
  payload: any;
  timestamp: Date;
  source: string;
  processed: boolean;
  retryCount: number;
  maxRetries: number;
}

export interface APIResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: Date;
  requestId: string;
  metadata?: Record<string, any>;
}

export interface PaginatedResponse<T> extends APIResponse<T[]> {
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
    hasNext: boolean;
    hasPrev: boolean;
  };
}

export interface SearchQuery {
  query: string;
  filters: Record<string, any>;
  sort: SortOption[];
  pagination: PaginationOptions;
  facets: string[];
}

export interface SortOption {
  field: string;
  direction: 'asc' | 'desc';
}

export interface PaginationOptions {
  page: number;
  limit: number;
  offset?: number;
}

export interface SearchResult<T> {
  items: T[];
  total: number;
  facets: Record<string, FacetValue[]>;
  suggestions: string[];
  queryTime: number;
}

export interface FacetValue {
  value: string;
  count: number;
  selected: boolean;
}