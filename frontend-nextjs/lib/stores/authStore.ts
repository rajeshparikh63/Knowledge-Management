import { create } from 'zustand';
import { User, LoginCredentials, SignupData, authApi } from '../api/auth';

// Helper function to format error messages from API responses
function formatErrorMessage(error: any): string {
  const detail = error.response?.data?.detail;

  // If detail is an array (FastAPI validation errors)
  if (Array.isArray(detail)) {
    return detail
      .map((err: any) => {
        // Extract the error message and location
        const location = err.loc?.slice(1).join(' > ') || 'field';
        const message = err.msg || 'Invalid value';
        return `${location}: ${message}`;
      })
      .join(', ');
  }

  // If detail is a string
  if (typeof detail === 'string') {
    return detail;
  }

  // If detail is an object with a message
  if (detail && typeof detail === 'object' && detail.msg) {
    return detail.msg;
  }

  // Fallback error messages
  if (error.message) {
    return error.message;
  }

  return 'An unexpected error occurred';
}

interface AuthState {
  user: User | null;
  isLoading: boolean;
  error: string | null;
  isInitializing: boolean;

  // Actions
  login: (credentials: LoginCredentials) => Promise<void>;
  signup: (data: SignupData) => Promise<{ email: string; message: string }>;
  logout: () => Promise<void>;
  fetchUser: () => Promise<void>;
  clearError: () => void;
  initializeAuth: () => Promise<void>;
}

export const useAuthStore = create<AuthState>((set, get) => ({
  user: null,
  isLoading: false,
  error: null,
  isInitializing: true,

  login: async (credentials) => {
    try {
      set({ isLoading: true, error: null });

      const response = await authApi.login(credentials);

      // Store tokens in localStorage
      localStorage.setItem('access_token', response.access_token);
      localStorage.setItem('refresh_token', response.refresh_token);

      // Fetch user info from /auth/me endpoint
      await get().fetchUser();

      set({ isLoading: false });
    } catch (error: any) {
      const errorMessage = formatErrorMessage(error);
      set({
        error: errorMessage,
        isLoading: false,
      });
      throw error;
    }
  },

  signup: async (data) => {
    try {
      set({ isLoading: true, error: null });
      const response = await authApi.signup(data);
      set({ isLoading: false });
      return response;
    } catch (error: any) {
      const errorMessage = formatErrorMessage(error);
      set({
        error: errorMessage,
        isLoading: false,
      });
      throw error;
    }
  },

  logout: async () => {
    try {
      // Call logout endpoint to revoke refresh token
      await authApi.logout();
    } catch (error) {
      // Logout API call failed - proceeding with local cleanup
    } finally {
      // Clear tokens from localStorage
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
      localStorage.removeItem('user');

      // Clear the persisted Zustand caches so the next user on this browser
      // never sees the previous user's ingested documents or chat history.
      try {
        localStorage.removeItem('soldieriq-documents-cache');
        localStorage.removeItem('soldieriq-chat-session');
      } catch {
        /* ignore */
      }

      // Clear user from state
      set({ user: null });
    }
  },

  fetchUser: async () => {
    try {
      const user = await authApi.getCurrentUser();
      // Store user in localStorage for API access
      localStorage.setItem('user', JSON.stringify(user));
      set({ user });
    } catch (error) {
      // User fetch failed - clearing tokens
      // Clear tokens on failed user fetch
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
      localStorage.removeItem('user');
      set({ user: null });
    }
  },

  clearError: () => set({ error: null }),

  initializeAuth: async () => {
    set({ isInitializing: true });

    const accessToken = localStorage.getItem('access_token');

    if (accessToken) {
      try {
        await get().fetchUser();
      } catch (error) {
        // Auth initialization failed
      }
    }

    set({ isInitializing: false });
  },
}));
