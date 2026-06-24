import apiClient, { fetchWithRefresh } from './client';
import { Document, KnowledgeBase } from '@/types';

// Helper to get user info from localStorage (assuming it's stored there after login)
const getUserParams = () => {
  const userStr = localStorage.getItem('user');
  if (userStr) {
    try {
      const user = JSON.parse(userStr);
      if (user.id && user.organization_id) {
        // Map user object fields to API parameter names
        return {
          user_id: user.id,
          organization_id: user.organization_id
        };
      }
    } catch (e) {
      // Failed to parse user from localStorage
    }
  }
  return null;
};

// API Response wrapper type
interface ApiResponse<T> {
  success: boolean;
  data: T;
  count?: number;
}

export const documentsApi = {
  // Get all documents.
  // Note: backend default limit is 100, but a single Google Drive ingest can
  // produce thousands of docs — we explicitly request a larger ceiling so the
  // sidebar shows everything. If you ever hit this cap, switch to proper
  // pagination instead of just bumping the number.
  listDocuments: async (folderName?: string): Promise<Document[]> => {
    const userParams = getUserParams();
    if (!userParams) {
      throw new Error('User not authenticated');
    }

    const params = new URLSearchParams({
      organization_id: userParams.organization_id,
      limit: '5000',
    });

    if (folderName && folderName !== 'All Knowledge Bases') {
      params.append('folder_name', folderName);
    }

    const response = await apiClient.get<ApiResponse<Document[]>>(`/upload/documents?${params.toString()}`);
    return response.data.data; // Extract data from wrapper
  },

  // Upload documents (documents are created immediately with status="processing")
  uploadDocuments: async (files: File[], folderName: string): Promise<{ task_id: string; status: string; total_files: number }> => {
    if (!folderName || !folderName.trim()) {
      throw new Error('Folder name is required');
    }

    const formData = new FormData();
    // Append multiple files
    files.forEach(file => {
      formData.append('files', file);
    });
    formData.append('folder_name', folderName.trim());
    // user_id and organization_id are extracted from JWT token by backend

    const response = await apiClient.post<ApiResponse<{ task_id: string; status: string; total_files: number; folder_name: string }>>('/upload/documents', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data.data;
  },

  // Upload YouTube video by URL
  uploadYouTubeVideo: async (youtubeUrl: string, folderName: string): Promise<{ document_id: string; youtube_metadata: any }> => {
    if (!folderName || !folderName.trim()) {
      throw new Error('Folder name is required');
    }

    if (!youtubeUrl || !youtubeUrl.trim()) {
      throw new Error('YouTube URL is required');
    }

    const response = await apiClient.post<ApiResponse<{ document_id: string; youtube_metadata: any }>>('/upload/youtube', {
      youtube_url: youtubeUrl.trim(),
      folder_name: folderName.trim(),
      // user_id and organization_id are extracted from JWT token by backend
    });
    return response.data.data;
  },

  // Get specific document by ID (for polling status)
  getDocument: async (docId: string): Promise<Document> => {
    const response = await apiClient.get<ApiResponse<Document>>(`/upload/documents/${encodeURIComponent(docId)}`);
    return response.data.data;
  },

  // Delete document
  deleteDocument: async (docId: string): Promise<void> => {
    await apiClient.delete(`/upload/documents/${encodeURIComponent(docId)}`);
  },

  // Get folders (knowledge bases)
  listFolders: async (): Promise<KnowledgeBase[]> => {
    const userParams = getUserParams();
    if (!userParams) {
      throw new Error('User not authenticated');
    }

    const params = new URLSearchParams({
      organization_id: userParams.organization_id,
    });

    const response = await apiClient.get<ApiResponse<KnowledgeBase[]>>(`/upload/folders?${params.toString()}`);
    return response.data.data; // Extract data from wrapper
  },

  // Delete folder
  deleteFolder: async (folderName: string): Promise<void> => {
    const userParams = getUserParams();
    if (!userParams) {
      throw new Error('User not authenticated');
    }

    const params = new URLSearchParams({
      organization_id: userParams.organization_id,
    });

    await apiClient.delete(`/upload/folders/${encodeURIComponent(folderName)}?${params.toString()}`);
  },
};

// TAK Credentials type (matches backend TAKCredentials)
export interface TAKCredentials {
  tak_host: string;
  tak_port: number;
  tak_username: string;
  tak_password: string;
  agent_callsign: string;
}

export const chatApi = {
  // Send chat message with SSE streaming.
  // Note: filenames are NOT sent — the backend resolves doc context from
  // document_ids via search_knowledge_base. Filenames were noisy/optional
  // and the backend prompt explicitly ignored them.
  sendMessage: async (
    message: string,
    documentIds: string[],
    sessionId: string,
    model: string,
    takCredentials?: TAKCredentials
  ): Promise<ReadableStream> => {
    // Use fetchWithRefresh for automatic token refresh on 401
    const response = await fetchWithRefresh(`${process.env.NEXT_PUBLIC_API_URL}/api/chat`, {
      method: 'POST',
      body: JSON.stringify({
        message,
        document_ids: documentIds,
        session_id: sessionId,
        model,
        tak_credentials: takCredentials,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Chat request failed');
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    return response.body;
  },
};

// Chat Session Types
export interface ChatSession {
  session_id: string;
  name: string;
  message_preview: string;
  created_at: number;
  updated_at: number;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at: number;
  model?: string;
  sources?: any[];
}

export interface SessionHistory {
  success: boolean;
  session_id: string;
  session_name: string | null;
  messages: ChatMessage[];
  created_at: number;
  updated_at: number;
}

interface SessionsResponse {
  success: boolean;
  sessions: ChatSession[];
  count: number;
}

// Chat Session Management API
export const chatSessionsApi = {
  // List all chat sessions for current user
  listSessions: async (): Promise<ChatSession[]> => {
    const response = await apiClient.get<SessionsResponse>('/chat/sessions');
    return response.data.sessions;
  },

  // Get full history for a specific session
  getSession: async (sessionId: string): Promise<SessionHistory> => {
    const response = await apiClient.get<SessionHistory>(`/chat/sessions/${sessionId}`);
    return response.data;
  },

  // Delete a chat session
  deleteSession: async (sessionId: string): Promise<void> => {
    await apiClient.delete(`/chat/sessions/${sessionId}`);
  },

  // Rename a chat session
  renameSession: async (sessionId: string, newName: string): Promise<void> => {
    await apiClient.put(`/chat/sessions/${sessionId}/name`, null, {
      params: {
        new_name: newName
      }
    });
  },
};
