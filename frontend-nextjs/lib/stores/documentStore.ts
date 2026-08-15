import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { Document, KnowledgeBase } from '@/types';
import { documentsApi } from '../api/documents';

// LocalStorage helpers
const STORAGE_KEY_SELECTED_KB = 'soldieriq_selected_kb';
const STORAGE_KEY_SELECTED_DOCS = 'soldieriq_selected_docs';
const STORAGE_KEY_CACHE_VERSION = 'soldieriq_cache_version';
const CURRENT_CACHE_VERSION = '2026-02-16'; // Increment this to force cache clear for all users

// Check and clear cache if version mismatch
const checkAndClearCache = () => {
  if (typeof window === 'undefined') return;
  try {
    const storedVersion = localStorage.getItem(STORAGE_KEY_CACHE_VERSION);
    if (storedVersion !== CURRENT_CACHE_VERSION) {
      localStorage.removeItem(STORAGE_KEY_SELECTED_KB);
      localStorage.removeItem(STORAGE_KEY_SELECTED_DOCS);
      localStorage.setItem(STORAGE_KEY_CACHE_VERSION, CURRENT_CACHE_VERSION);
    }
  } catch (e) {
    // Cache version check failed
  }
};

// Run cache check immediately
checkAndClearCache();

const loadSelectedKB = (): string | null => {
  if (typeof window === 'undefined') return null;
  try {
    const stored = localStorage.getItem(STORAGE_KEY_SELECTED_KB);
    return stored || null;
  } catch (e) {
    // Failed to load selected KB
    return null;
  }
};

const saveSelectedKB = (kb: string | null) => {
  if (typeof window === 'undefined') return;
  try {
    if (kb === null) {
      localStorage.removeItem(STORAGE_KEY_SELECTED_KB);
    } else {
      localStorage.setItem(STORAGE_KEY_SELECTED_KB, kb);
    }
  } catch (e) {
    // Failed to save selected KB
  }
};

const loadSelectedDocs = (): Set<string> => {
  if (typeof window === 'undefined') return new Set();
  try {
    const stored = localStorage.getItem(STORAGE_KEY_SELECTED_DOCS);
    if (stored) {
      const array = JSON.parse(stored);
      return new Set(array);
    }
  } catch (e) {
    // Failed to load selected docs
  }
  return new Set();
};

const saveSelectedDocs = (docs: Set<string>) => {
  if (typeof window === 'undefined') return;
  try {
    const array = Array.from(docs);
    localStorage.setItem(STORAGE_KEY_SELECTED_DOCS, JSON.stringify(array));
  } catch (e) {
    // Failed to save selected docs
  }
};

// Helper function for background polling
const startBackgroundPolling = (
  uploadedDocIds: string[],
  maxFileSizeMB: number,
  get: any,
  set: any
) => {
  // Calculate dynamic polling interval based on file size
  const calculatePollingInterval = (sizeMB: number): number => {
    if (sizeMB < 1) {
      return 1500; // Small files: 1.5 seconds
    } else if (sizeMB < 5) {
      return 2500; // Medium files (1-5MB): 2.5 seconds
    } else if (sizeMB < 20) {
      return 4000; // Large files (5-20MB): 4 seconds
    } else if (sizeMB < 50) {
      return 6000; // Very large files (20-50MB): 6 seconds
    } else if (sizeMB < 100) {
      return 10000; // Huge files (50-100MB): 10 seconds
    } else {
      return 15000; // Massive files (>100MB): 15 seconds
    }
  };

  const pollingIntervalMs = calculatePollingInterval(maxFileSizeMB);

  // Hard cap so a document stuck in "processing" (e.g. a dead worker) can't make
  // this background loop poll the API forever. After this window we give up and
  // let the user refresh manually.
  const startedAt = Date.now();
  const MAX_POLL_MS = 15 * 60 * 1000; // 15 minutes

  // Poll SPECIFIC documents by ID to check for status updates (more efficient)
  const pollInterval = setInterval(async () => {
    if (Date.now() - startedAt > MAX_POLL_MS) {
      clearInterval(pollInterval);
      set({ uploadStatus: null, uploadProgress: null });
      return;
    }
    try {
      // Fetch each document by ID to check status
      const statusChecks = await Promise.all(
        uploadedDocIds.map(docId => documentsApi.getDocument(docId))
      );

      // Update documents in store with latest status
      const currentDocs = get().documents;
      const updatedDocs = currentDocs.map((doc: Document) => {
        const updatedDoc = statusChecks.find(d => d.id === doc.id);
        return updatedDoc || doc;
      });
      set({ documents: updatedDocs });

      // Check if all uploaded documents are completed/failed
      const stillProcessing = statusChecks.filter(d => d.status === 'processing');

      if (stillProcessing.length === 0) {
        // All documents processed
        clearInterval(pollInterval);
        set({ uploadStatus: 'completed' });

        // Refresh the full list once to ensure we have latest data
        await get().fetchDocuments();
        await get().fetchKnowledgeBases();

        // Clear status after 2 seconds
        setTimeout(() => {
          set({ uploadStatus: null, uploadProgress: null });
        }, 2000);
      }
    } catch (error: any) {
      // Polling error
      clearInterval(pollInterval);
      set({ uploadStatus: 'failed', error: error.message, uploadProgress: null });
    }
  }, pollingIntervalMs); // Dynamic interval based on file size
};

interface DocumentState {
  // State
  documents: Document[];
  knowledgeBases: KnowledgeBase[];
  selectedKB: string | null;
  selectedDocs: Set<string>;
  isLoading: boolean;
  error: string | null;
  uploadStatus: string | null;
  uploadProgress: { current: number; total: number } | null;
  deletingKB: string | null;

  // Google Drive import tracking (for the live progress banner)
  driveImportStartedAt: number | null;
  driveImportFolder: string | null;

  // Actions
  startDriveImport: (folderName: string) => void;
  clearDriveImport: () => void;
  fetchDocuments: () => Promise<void>;
  fetchKnowledgeBases: () => Promise<void>;
  uploadDocuments: (files: File[], folderName: string) => Promise<void>;
  uploadYouTubeVideo: (youtubeUrl: string, folderName: string) => Promise<void>;
  deleteDocument: (docId: string) => Promise<void>;
  deleteKnowledgeBase: (folderName: string) => Promise<void>;
  setSelectedKB: (kb: string | null) => void;
  toggleDocSelection: (docId: string) => void;
  selectAllDocs: () => void;
  deselectAllDocs: () => void;
  selectFolderDocs: (folderName: string) => void;
  deselectFolderDocs: (folderName: string) => void;
  selectDocs: (ids: string[]) => void;
  deselectDocs: (ids: string[]) => void;
}

export const useDocumentStore = create<DocumentState>()(
  persist(
    (set, get) => ({
  // State
  documents: [],
  knowledgeBases: [],
  selectedKB: loadSelectedKB(),
  selectedDocs: loadSelectedDocs(),
  isLoading: false,
  error: null,
  uploadStatus: null,
  uploadProgress: null,
  deletingKB: null,
  driveImportStartedAt: null,
  driveImportFolder: null,

  // Actions
  startDriveImport: (folderName: string) =>
    set({ driveImportStartedAt: Date.now(), driveImportFolder: folderName }),

  clearDriveImport: () =>
    set({ driveImportStartedAt: null, driveImportFolder: null }),

  fetchDocuments: async () => {
    try {
      set({ isLoading: true, error: null });
      const docs = await documentsApi.listDocuments();
      set({ documents: Array.isArray(docs) ? docs : [], isLoading: false });

      // Clean up selectedDocs: remove any IDs that don't exist in fetched documents
      // This handles stale IDs from: previous users, deleted documents, wrong organization, etc.
      const currentSelectedDocs = get().selectedDocs;
      const validDocIds = new Set((Array.isArray(docs) ? docs : []).map(d => d.id));
      const cleanedSelectedDocs = new Set<string>();

      currentSelectedDocs.forEach(docId => {
        if (validDocIds.has(docId)) {
          cleanedSelectedDocs.add(docId);
        }
      });

      // Update if we removed any invalid IDs
      if (cleanedSelectedDocs.size !== currentSelectedDocs.size) {
        // Cleaned stale document IDs from selection
        saveSelectedDocs(cleanedSelectedDocs);
        set({ selectedDocs: cleanedSelectedDocs });
      }
    } catch (error: any) {
      set({ error: error.message, isLoading: false, documents: [] });
    }
  },

  fetchKnowledgeBases: async () => {
    try {
      const kbs = await documentsApi.listFolders();
      set({ knowledgeBases: Array.isArray(kbs) ? kbs : [] });
    } catch (error: any) {
      // Failed to fetch knowledge bases
      set({ knowledgeBases: [] });
    }
  },

  uploadDocuments: async (files, folderName) => {
    try {
      set({ error: null, uploadStatus: 'uploading', uploadProgress: { current: 0, total: files.length } });

      // Calculate largest file size for dynamic polling interval
      const fileSizesInMB = files.map(f => f.size / (1024 * 1024));
      const maxFileSizeMB = Math.max(...fileSizesInMB);

      // Optimistic update: Add placeholder documents immediately (no isLoading, keeps existing docs)
      const existingDocs = get().documents;
      const placeholderDocs = files.map((file, index) => ({
        id: `temp_${Date.now()}_${index}`, // Temporary ID
        file_name: file.name,
        folder_name: folderName,
        user_id: '',
        organization_id: '',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        status: 'processing' as const,
        processing_stage: 'uploading',
        processing_stage_description: 'Uploading file to server...',
      }));

      // Add placeholders to existing documents (don't replace)
      set({ documents: [...placeholderDocs, ...existingDocs] });

      // Upload files - backend creates real documents and starts ingestion
      await documentsApi.uploadDocuments(files, folderName);

      // Fetch documents to get the real IDs from backend
      await get().fetchDocuments();
      await get().fetchKnowledgeBases();

      // Get the real document IDs for the files we just uploaded
      const uploadedDocs = get().documents.filter(
        d => d.folder_name === folderName &&
             d.status === 'processing' &&
             !d.id.startsWith('temp_') // Real IDs from backend
      );
      const uploadedDocIds = uploadedDocs.map(d => d.id);

      if (uploadedDocIds.length === 0) {
        // No processing documents found, might already be completed
        set({ uploadStatus: 'completed' });
        setTimeout(() => {
          set({ uploadStatus: null, uploadProgress: null });
        }, 2000);
        return;
      }

      // Set to processing status - at this point upload is done and ingestion started
      set({ uploadStatus: 'processing' });

      // Start background polling (non-blocking) - function returns immediately
      startBackgroundPolling(uploadedDocIds, maxFileSizeMB, get, set);

      // Function returns here - modal can close while polling continues in background

    } catch (error: any) {
      set({ error: error.message, uploadStatus: 'failed', uploadProgress: null });
      throw error;
    }
  },

  uploadYouTubeVideo: async (youtubeUrl: string, folderName: string) => {
    try {
      set({ error: null, uploadStatus: 'uploading', uploadProgress: { current: 0, total: 1 } });

      // Optimistic update: Add placeholder document
      const existingDocs = get().documents;
      const placeholderDoc = {
        id: `temp_yt_${Date.now()}`,
        file_name: 'YouTube Video',
        folder_name: folderName,
        user_id: '',
        organization_id: '',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        status: 'processing' as const,
        processing_stage: 'downloading',
        processing_stage_description: 'Downloading YouTube video...',
      };

      set({ documents: [placeholderDoc, ...existingDocs] });

      // Upload YouTube video
      const result = await documentsApi.uploadYouTubeVideo(youtubeUrl, folderName);

      // Fetch documents to get the real document
      await get().fetchDocuments();
      await get().fetchKnowledgeBases();

      // Set to processing status
      set({ uploadStatus: 'processing' });

      // Start background polling with longer interval for videos (videos take longer to process)
      startBackgroundPolling([result.document_id], 50, get, set); // Assume 50MB for polling interval

    } catch (error: any) {
      set({ error: error.message, uploadStatus: 'failed', uploadProgress: null });
      throw error;
    }
  },

  deleteDocument: async (docId) => {
    try {
      // Remove from selectedDocs if it was selected
      const { selectedDocs } = get();
      if (selectedDocs.has(docId)) {
        const newSelectedDocs = new Set(selectedDocs);
        newSelectedDocs.delete(docId);
        saveSelectedDocs(newSelectedDocs);
        set({ selectedDocs: newSelectedDocs });
      }

      await documentsApi.deleteDocument(docId);
      await get().fetchDocuments();
      await get().fetchKnowledgeBases();
    } catch (error: any) {
      set({ error: error.message });
      throw error;
    }
  },

  deleteKnowledgeBase: async (folderName) => {
    try {
      // Set deleting state
      set({ deletingKB: folderName });

      // Remove all documents from this KB from selectedDocs
      const { documents, selectedDocs } = get();
      const kbDocIds = documents
        .filter(doc => doc.folder_name === folderName)
        .map(doc => doc.id);

      if (kbDocIds.length > 0) {
        const newSelectedDocs = new Set(selectedDocs);
        kbDocIds.forEach(docId => newSelectedDocs.delete(docId));
        saveSelectedDocs(newSelectedDocs);
        set({ selectedDocs: newSelectedDocs });
      }

      await documentsApi.deleteFolder(folderName);
      await get().fetchDocuments();
      await get().fetchKnowledgeBases();
    } catch (error: any) {
      set({ error: error.message });
      throw error;
    } finally {
      // Clear deleting state
      set({ deletingKB: null });
    }
  },

  setSelectedKB: (kb) => {
    saveSelectedKB(kb);
    set({ selectedKB: kb });
  },

  toggleDocSelection: (docId) => {
    const selectedDocs = new Set(get().selectedDocs);
    if (selectedDocs.has(docId)) {
      selectedDocs.delete(docId);
    } else {
      selectedDocs.add(docId);
    }
    saveSelectedDocs(selectedDocs);
    set({ selectedDocs });
  },

  selectAllDocs: () => {
    const { documents } = get();
    const docIds = Array.isArray(documents)
      ? documents
          .filter(d => d.status !== 'failed' && d.status !== 'processing') // Exclude failed and processing files
          .map(d => d.id)
      : [];
    const selectedDocs = new Set(docIds);
    saveSelectedDocs(selectedDocs);
    set({ selectedDocs });
  },

  deselectAllDocs: () => {
    const selectedDocs = new Set<string>();
    saveSelectedDocs(selectedDocs);
    set({ selectedDocs });
  },

  selectFolderDocs: (folderName) => {
    const { documents, selectedDocs } = get();
    const folderDocIds = Array.isArray(documents)
      ? documents
          .filter(d => d.folder_name === folderName)
          .filter(d => d.status !== 'failed' && d.status !== 'processing') // Exclude failed and processing files
          .map(d => d.id)
      : [];
    const newSelected = new Set(selectedDocs);
    folderDocIds.forEach(id => newSelected.add(id));
    saveSelectedDocs(newSelected);
    set({ selectedDocs: newSelected });
  },

  deselectFolderDocs: (folderName) => {
    const { documents, selectedDocs } = get();
    const folderDocIds = Array.isArray(documents)
      ? documents.filter(d => d.folder_name === folderName).map(d => d.id)
      : [];
    const newSelected = new Set(selectedDocs);
    folderDocIds.forEach(id => newSelected.delete(id));
    saveSelectedDocs(newSelected);
    set({ selectedDocs: newSelected });
  },

  selectDocs: (ids) => {
    if (!ids?.length) return;
    const newSelected = new Set(get().selectedDocs);
    ids.forEach(id => newSelected.add(id));
    saveSelectedDocs(newSelected);
    set({ selectedDocs: newSelected });
  },

  deselectDocs: (ids) => {
    if (!ids?.length) return;
    const newSelected = new Set(get().selectedDocs);
    ids.forEach(id => newSelected.delete(id));
    saveSelectedDocs(newSelected);
    set({ selectedDocs: newSelected });
  },
    }),
    {
      // Cache the ingested-documents list + folder list across reloads so the
      // sidebar renders instantly from cache instead of waiting on the backend.
      // fetchDocuments still runs to refresh in the background; this just removes
      // the empty-then-populate flash and the hard dependency on a fresh call.
      name: 'soldieriq-documents-cache',
      storage: createJSONStorage(() => localStorage),
      version: 1,
      // Persist ONLY the cacheable data. Transient state (loading/upload/error,
      // drive-import flags) and selection (handled separately, and a Set isn't
      // JSON-serializable) are deliberately excluded.
      partialize: (state) => ({
        documents: state.documents,
        knowledgeBases: state.knowledgeBases,
      }),
      // After restoring the cached documents, drop any persisted selection IDs
      // that no longer match an existing document. Without this, the separately-
      // stored selection set can carry stale IDs and show a "selected" count
      // higher than the document count until the next fetchDocuments runs.
      onRehydrateStorage: () => (state) => {
        if (!state) return;
        const validIds = new Set((state.documents || []).map((d) => d.id));
        const pruned = new Set(
          Array.from(state.selectedDocs).filter((id) => validIds.has(id))
        );
        if (pruned.size !== state.selectedDocs.size) {
          state.selectedDocs = pruned;
          saveSelectedDocs(pruned);
        }
      },
    }
  )
);
