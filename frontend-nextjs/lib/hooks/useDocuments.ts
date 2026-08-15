import { useQuery } from '@tanstack/react-query';
import { documentsApi } from '@/lib/api/documents';
import { Document, KnowledgeBase } from '@/types';

export const documentKeys = {
  all: ['documents'] as const,
  list: (organizationId?: string) => [...documentKeys.all, 'list', organizationId] as const,
  detail: (docId: string) => [...documentKeys.all, 'detail', docId] as const,
  knowledgeBases: (organizationId?: string) => ['knowledgeBases', organizationId] as const,
};

// A document that has been "processing" without any update for this long is
// treated as stuck (e.g. a dead worker). We stop polling it so the app doesn't
// hammer the backend forever in the background with no user present — the root
// cause of the "constant API calls with no user action" issue.
const STUCK_AFTER_MS = 10 * 60 * 1000; // 10 minutes

/**
 * Fetches the documents list for an organization.
 * Polls every 5s ONLY while a document is ACTIVELY processing (updated within
 * the last STUCK_AFTER_MS). Once processing finishes — or a doc goes stale and
 * is deemed stuck — polling stops entirely.
 */
export function useDocuments(organizationId?: string, folderName?: string) {
  const query = useQuery<Document[]>({
    queryKey: documentKeys.list(organizationId),
    queryFn: () => documentsApi.listDocuments(folderName),
    enabled: !!organizationId,
    refetchInterval: (query) => {
      const docs = query.state.data;
      if (!docs?.length) return false;
      const now = Date.now();
      const hasActiveProcessing = docs.some((d) => {
        if (d.status !== 'processing') return false;
        const ts = new Date(d.updated_at || d.created_at).getTime();
        // Keep polling only if the doc has shown activity recently. A stale
        // "processing" doc is stuck — polling it forever just spams the API.
        return Number.isFinite(ts) && now - ts < STUCK_AFTER_MS;
      });
      return hasActiveProcessing ? 5_000 : false;
    },
  });

  return query;
}

/**
 * Fetches the knowledge bases (folders) for an organization.
 */
export function useKnowledgeBases(organizationId?: string) {
  return useQuery<KnowledgeBase[]>({
    queryKey: documentKeys.knowledgeBases(organizationId),
    queryFn: () => documentsApi.listFolders(),
    enabled: !!organizationId,
  });
}
