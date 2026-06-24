/**
 * Google Drive connector API client.
 *
 * Talks to the FastAPI router at /api/google-drive/*.
 * Auth: handled by apiClient (Keycloak Bearer + automatic refresh on 401).
 */
import apiClient from "./client";

export interface DriveStatus {
  connected: boolean;
  email: string | null;
  display_name: string | null;
  connected_at?: string | null;
  needs_reconnect?: boolean;
}

export interface DriveFolder {
  id: string;
  name: string;
  parents: string[];
  path: string;
  /** Name of the shared drive this folder lives in, or null for My Drive. */
  shared_drive?: string | null;
}

export interface DriveFoldersResponse {
  folders: DriveFolder[];
}

export interface DriveIngestFoldersResponse {
  success: boolean;
  message: string;
  folder_count: number;
}

export interface DriveConnectResponse {
  auth_url: string;
}

export interface DriveListFile {
  id: string;
  name: string;
  mime_type: string;
  size: number;
  modified_time: string | null;
  web_view_link: string | null;
}

export interface DriveListFilesResponse {
  files: DriveListFile[];
  next_page_token: string | null;
}

export interface DriveSyncResponse {
  success: boolean;
  message: string;
}

export interface DrivePickedFile {
  id: string;
  name: string;
  mime_type: string;
  size?: number;
}

export interface DriveIngestResponse {
  success: boolean;
  document_ids: string[];
  folder_name: string;
  queued_count: number;
}

export const googleDriveApi = {
  /** Is the current user connected to Google Drive? */
  status: async (): Promise<DriveStatus> => {
    const res = await apiClient.get<DriveStatus>("/google-drive/status");
    return res.data;
  },

  /** Get the URL the browser should send the user to for the consent screen. */
  connect: async (folderName = "Google Drive"): Promise<DriveConnectResponse> => {
    const params = new URLSearchParams({ folder_name: folderName });
    const res = await apiClient.get<DriveConnectResponse>(
      `/google-drive/connect?${params.toString()}`
    );
    return res.data;
  },

  /**
   * Paginated file list for the custom in-app picker.
   * Pass `next_page_token` from a previous call to get the next page.
   * `search` does a Drive-side substring match on file name.
   */
  listFiles: async (params?: {
    pageToken?: string;
    pageSize?: number;
    search?: string;
  }): Promise<DriveListFilesResponse> => {
    const qs = new URLSearchParams();
    if (params?.pageToken) qs.set("page_token", params.pageToken);
    if (params?.pageSize) qs.set("page_size", String(params.pageSize));
    if (params?.search) qs.set("search", params.search);
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    const res = await apiClient.get<DriveListFilesResponse>(
      `/google-drive/files${suffix}`
    );
    return res.data;
  },

  /** List all folders in the user's Drive (for the folder picker). */
  listFolders: async (): Promise<DriveFoldersResponse> => {
    const res = await apiClient.get<DriveFoldersResponse>(
      "/google-drive/folders"
    );
    return res.data;
  },

  /**
   * Ingest every supported file under the selected folders (recursive).
   * Each folder lands in a KB folder named after it (HR, Services, …).
   */
  ingestFolders: async (
    folders: { id: string; name: string }[]
  ): Promise<DriveIngestFoldersResponse> => {
    const res = await apiClient.post<DriveIngestFoldersResponse>(
      "/google-drive/ingest-folders",
      { folders }
    );
    return res.data;
  },

  /** Manually re-trigger discovery (e.g. user added new files to Drive). */
  sync: async (folderName = "Google Drive"): Promise<DriveSyncResponse> => {
    const params = new URLSearchParams({ folder_name: folderName });
    const res = await apiClient.post<DriveSyncResponse>(
      `/google-drive/sync?${params.toString()}`
    );
    return res.data;
  },

  /** Ingest a specific list of files (used by the Google Picker flow). */
  ingest: async (
    files: DrivePickedFile[],
    folderName = "Google Drive"
  ): Promise<DriveIngestResponse> => {
    const res = await apiClient.post<DriveIngestResponse>(
      "/google-drive/ingest",
      { folder_name: folderName, files }
    );
    return res.data;
  },

  /** Disconnect — wipes stored tokens. Auto-discovered docs stay in the KB. */
  disconnect: async (): Promise<{ success: boolean; removed: boolean }> => {
    const res = await apiClient.delete<{ success: boolean; removed: boolean }>(
      "/google-drive/disconnect"
    );
    return res.data;
  },
};
