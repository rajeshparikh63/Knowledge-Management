import { useQuery } from '@tanstack/react-query';
import { documentsApi } from '@/lib/api/documents';
import { Document, KnowledgeBase } from '@/types';

export const documentKeys = {
  all: ['documents'] as const,
  list: (organizationId?: string) => [...documentKeys.all, 'list', organizationId] as const,
  detail: (docId: string) => [...documentKeys.all, 'detail', docId] as const,
  knowledgeBases: (organizationId?: string) => ['knowledgeBases', organizationId] as const,
};

/**
 * Fetches the documents list for an organization.
 * Auto-refetches every 30s when any document is still processing.
 */
export function useDocuments(organizationId?: string, folderName?: string) {
  const query = useQuery<Document[]>({
    queryKey: documentKeys.list(organizationId),
    queryFn: () => documentsApi.listDocuments(folderName),
    enabled: !!organizationId,
    refetchInterval: (query) => {
      const docs = query.state.data;
      if (docs?.some((d) => d.status === 'processing')) {
        // Poll quickly while anything is processing so the sidebar pipeline
        // indicators and the Drive-import banner update promptly.
        return 5_000;
      }
      return false;
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
