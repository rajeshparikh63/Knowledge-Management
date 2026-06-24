// Auth Types
export interface User {
  id: string;
  email: string;
  firstName?: string;
  lastName?: string;
  organization_id: string;
}

// Document Types
export interface Document {
  id: string;  // Changed from _id (MongoDB) to id (PostgreSQL)
  file_name: string;
  folder_name?: string;
  user_id: string;
  organization_id: string;
  created_at: string;
  updated_at: string;
  // Present in the list response to signal "a downloadable file exists". The
  // actual presigned download URL is fetched on demand (GET /documents/{id})
  // when the user clicks the filename — not pre-generated for every doc.
  file_key?: string;
  file_url?: string;
  // Processing status fields
  status?: 'processing' | 'completed' | 'failed';
  processing_stage?: string;
  processing_stage_description?: string;
  processing_progress?: {
    current: number;
    total: number;
    percentage: number;
  };
  error?: string;
  completed_at?: string;
  failed_at?: string;
  // Arbitrary metadata stamped at ingest time (e.g. { source: 'google_drive',
  // drive_file_id, drive_file_name }).
  metadata?: Record<string, any>;
}

// Knowledge Base Types
export interface KnowledgeBase {
  name: string;
  folder_name: string;
  document_count: number;
  display_name: string;
}

// Chat Types
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  isStreaming?: boolean;
  sources?: any[]; // Sources specific to this message
  graph?: KnowledgeGraph; // GraphRAG retrieval graph (entities + triples + chunks)
}

// Knowledge-graph payload returned by the GraphRAG search tool
export interface KnowledgeGraphAnchor {
  name: string;
  type?: string | null;
  chunk_count: number;
}

export interface KnowledgeGraphTriple {
  subject: string;
  subject_type?: string | null;
  predicate: string;
  object: string;
  object_type?: string | null;
  confidence?: number | null;
  source_chunk?: string | null;
}

export interface KnowledgeGraphChunk {
  chunk_id: string;
  document_id: string;
  text: string;
  score: number | null;
  shared_entities: number | null;
  via: 'vector' | 'graph';
}

export interface KnowledgeGraph {
  anchors: KnowledgeGraphAnchor[];
  triples: KnowledgeGraphTriple[];
  chunks: KnowledgeGraphChunk[];
  query?: string;
}

export interface SourceReference {
  document_id: string;
  filename: string;
  text?: string;
  file_key?: string;
  page_number?: number;
  snippet?: string;
  folder_name?: string;
}
