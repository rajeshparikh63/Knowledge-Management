/**
 * SharePoint connector API client (Composio-backed on the server).
 *
 * Talks to the FastAPI router at /api/sharepoint/*.
 * Auth: handled by apiClient (Keycloak Bearer + automatic refresh on 401).
 *
 * Connect uses Composio's hosted OAuth flow: POST /connect returns an auth_url
 * (a connect.composio.dev page that collects the tenant + runs MS consent),
 * which we open in a popup. The popup lands on /oauth-callback, which polls
 * /status and messages this window when the connection goes ACTIVE.
 */
import apiClient from "./client";

export interface SharePointStatus {
  connected: boolean;
  status: string | null;
  connection_id: string | null;
}

export interface SharePointLibrary {
  id: string;
  name: string;
  site_name?: string | null;
  site_display?: string | null;
  path?: string | null;
  web_url?: string | null;
}

export const sharepointApi = {
  /** Start the hosted Composio OAuth flow; returns the URL to open. */
  connect: async (callbackUrl: string): Promise<{ auth_url: string; connection_id: string | null }> => {
    const res = await apiClient.post<{ auth_url: string; connection_id: string | null }>(
      "/sharepoint/connect",
      { callback_url: callbackUrl }
    );
    return res.data;
  },

  status: async (): Promise<SharePointStatus> => {
    const res = await apiClient.get<SharePointStatus>("/sharepoint/status");
    return res.data;
  },

  /** Document libraries across the user's team sites — the pickable units. */
  listLibraries: async (): Promise<SharePointLibrary[]> => {
    const res = await apiClient.get<{ libraries: SharePointLibrary[] }>("/sharepoint/libraries");
    return res.data.libraries;
  },

  /** Ingest every supported file under the selected libraries. */
  ingestLibraries: async (
    libraries: { id: string; name: string }[]
  ): Promise<{ success: boolean; library_count: number }> => {
    const res = await apiClient.post<{ success: boolean; library_count: number }>(
      "/sharepoint/ingest-libraries",
      { libraries }
    );
    return res.data;
  },

  disconnect: async (): Promise<{ success: boolean; removed: number }> => {
    const res = await apiClient.delete<{ success: boolean; removed: number }>("/sharepoint/disconnect");
    return res.data;
  },
};
