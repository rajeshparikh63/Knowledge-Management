"use client";

import React, { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import GoogleDriveSection from "./GoogleDriveSection";
import SharePointSection from "./SharePointSection";

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  folders: string[];
  onUpload: (files: File[], folderName: string) => Promise<void>;
  onYouTubeUpload: (url: string, folderName: string) => Promise<void>;
  uploadStatus: string | null;
}

const UploadModal = React.memo(function UploadModal({
  isOpen,
  onClose,
  folders,
  onUpload,
  onYouTubeUpload,
  uploadStatus,
}: UploadModalProps) {
  const [selectedFolderName, setSelectedFolderName] = useState("");
  const [uploadMode, setUploadMode] = useState<
    "files" | "youtube" | "drive" | "sharepoint"
  >("files");
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [uploadingFiles, setUploadingFiles] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const resetState = () => {
    setSelectedFolderName("");
    setYoutubeUrl("");
    setUploadMode("files");
    setIsUploading(false);
    setUploadingFiles([]);
  };

  const handleClose = () => {
    if (!isUploading) {
      resetState();
      onClose();
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    if (!selectedFolderName || !selectedFolderName.trim()) {
      alert("Please enter a folder name");
      return;
    }

    setIsUploading(true);
    const fileNames = Array.from(files).map((f) => f.name);
    setUploadingFiles(fileNames);

    try {
      const filesArray = Array.from(files);
      const targetFolder = selectedFolderName.trim();
      await onUpload(filesArray, targetFolder);
      resetState();
      onClose();
    } catch (error: any) {
      alert(`Upload failed: ${error.message}`);
    } finally {
      setIsUploading(false);
      setUploadingFiles([]);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleYouTubeUpload = async () => {
    if (!youtubeUrl || !youtubeUrl.trim()) {
      alert("Please enter a YouTube URL");
      return;
    }

    if (!selectedFolderName || !selectedFolderName.trim()) {
      alert("Please enter a folder name");
      return;
    }

    setIsUploading(true);
    setUploadingFiles([youtubeUrl]);

    try {
      const targetFolder = selectedFolderName.trim();
      await onYouTubeUpload(youtubeUrl.trim(), targetFolder);
      resetState();
      onClose();
    } catch (error: any) {
      alert(`YouTube upload failed: ${error.message}`);
    } finally {
      setIsUploading(false);
      setUploadingFiles([]);
    }
  };

  const handleProceedWithUpload = () => {
    if (!selectedFolderName || !selectedFolderName.trim()) {
      alert("Please enter a folder name");
      return;
    }

    if (uploadMode === "youtube") {
      handleYouTubeUpload();
    } else {
      fileInputRef.current?.click();
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            key="modal-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="fixed inset-0 bg-black/70 backdrop-blur-sm z-40"
            onClick={handleClose}
          />
          <div className="fixed inset-0 flex items-center justify-center z-50 p-4">
            <motion.div
              key="modal-content"
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
              className="bg-card border border-brand/30 w-full max-w-md shadow-2xl tactical-panel"
            >
          <div className="p-6">
            <div className="flex items-center gap-2 mb-6">
              <div className="w-1 h-6 bg-brand"></div>
              <h3 className="text-base font-bold text-brand tracking-wider">
                {isUploading
                  ? "UPLOADING FILES"
                  : uploadMode === "drive"
                    ? "GOOGLE DRIVE"
                    : uploadMode === "sharepoint"
                      ? "SHAREPOINT"
                      : uploadMode === "youtube"
                        ? "YOUTUBE VIDEO"
                        : "SELECT FOLDER"}
              </h3>
            </div>

            {isUploading ? (
              <div className="space-y-4">
                <div className="flex items-center gap-3 p-3 bg-muted/50 border border-brand/20">
                  <div className="w-5 h-5 border-2 border-brand/20 border-t-brand rounded-full animate-spin flex-shrink-0"></div>
                  <div className="flex-1">
                    <div className="text-sm text-foreground font-semibold">
                      {uploadMode === "youtube"
                        ? "Downloading YouTube video..."
                        : `Uploading ${uploadingFiles.length} file${uploadingFiles.length !== 1 ? "s" : ""}...`}
                    </div>
                    <div className="text-xs text-muted-foreground mt-0.5">
                      To: {selectedFolderName}
                    </div>
                  </div>
                </div>

                <div className="max-h-48 overflow-y-auto tactical-scrollbar space-y-1">
                  {uploadingFiles.map((fileName, idx) => (
                    <div
                      key={idx}
                      className="text-xs text-muted-foreground py-1 px-2 bg-muted/30 border border-border/30 flex items-center gap-2"
                    >
                      <div className="w-1 h-1 bg-brand rounded-full"></div>
                      <span className="truncate">{fileName}</span>
                    </div>
                  ))}
                </div>

                <div className="text-xs text-muted-foreground text-center">
                  {uploadMode === "youtube"
                    ? "Please wait while the video is being downloaded and processed..."
                    : "Please wait while files are being uploaded..."}
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                {/* Mode Tabs */}
                <div className="flex gap-2 p-1 bg-muted/50 border border-border/50">
                  <button
                    onClick={() => setUploadMode("files")}
                    className={`flex-1 py-2 px-2 text-[11px] font-semibold tracking-wider transition-all ${
                      uploadMode === "files"
                        ? "bg-brand text-foreground"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    FILES
                  </button>
                  <button
                    onClick={() => setUploadMode("youtube")}
                    className={`flex-1 py-2 px-2 text-[11px] font-semibold tracking-wider transition-all ${
                      uploadMode === "youtube"
                        ? "bg-brand text-foreground"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    YOUTUBE
                  </button>
                  <button
                    onClick={() => setUploadMode("drive")}
                    className={`flex-1 py-2 px-2 text-[11px] font-semibold tracking-wider transition-all ${
                      uploadMode === "drive"
                        ? "bg-brand text-foreground"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    DRIVE
                  </button>
                  <button
                    onClick={() => setUploadMode("sharepoint")}
                    className={`flex-1 py-2 px-2 text-[11px] font-semibold tracking-wider transition-all ${
                      uploadMode === "sharepoint"
                        ? "bg-brand text-foreground"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    SHAREPOINT
                  </button>
                </div>

                {/* Drive mode replaces the folder input + proceed button
                    with the Drive connector UI. The Drive ingest path uses
                    its own folder name (settable via env / per-call) so we
                    don't ask the user for one here. */}
                {uploadMode === "drive" && (
                  <GoogleDriveSection onIngestStarted={handleClose} />
                )}

                {/* SharePoint connector — like Drive, uses its own per-library
                    KB folder names, so no folder input needed here. */}
                {uploadMode === "sharepoint" && (
                  <SharePointSection onIngestStarted={handleClose} />
                )}

                {/* Folder Name Input — only for files / youtube modes */}
                {uploadMode !== "drive" && uploadMode !== "sharepoint" && (
                <div>
                  <label className="block text-xs text-muted-foreground tracking-widest mb-2 uppercase">
                    Folder Name (New or Existing)
                  </label>
                  <input
                    type="text"
                    list="existing-folders"
                    value={selectedFolderName}
                    onChange={(e) => setSelectedFolderName(e.target.value)}
                    placeholder="e.g., CSS VSAT"
                    className="tactical-input"
                    autoFocus={uploadMode === "files"}
                  />
                  {folders.length > 0 && (
                    <datalist id="existing-folders">
                      {folders.map((folder) => (
                        <option key={folder} value={folder} />
                      ))}
                    </datalist>
                  )}
                  <div className="text-[10px] text-muted-foreground mt-2">
                    {folders.length > 0 ? (
                      <>Type a new name or select from existing folders</>
                    ) : (
                      <>Enter a name for your first folder</>
                    )}
                  </div>
                </div>
                )}

                {/* YouTube URL Input */}
                {uploadMode === "youtube" && (
                  <div>
                    <label className="block text-xs text-muted-foreground tracking-widest mb-2 uppercase">
                      YouTube URL
                    </label>
                    <input
                      type="url"
                      value={youtubeUrl}
                      onChange={(e) => setYoutubeUrl(e.target.value)}
                      placeholder="https://www.youtube.com/watch?v=..."
                      className="tactical-input"
                      autoFocus={uploadMode === "youtube"}
                    />
                    <div className="text-[10px] text-muted-foreground mt-2">
                      Paste a YouTube video URL to download and process
                    </div>
                  </div>
                )}

                {/* Proceed button — only for files / youtube modes.
                    Drive + SharePoint have their own action buttons inside the section. */}
                {uploadMode !== "drive" && uploadMode !== "sharepoint" && (
                <button
                  onClick={handleProceedWithUpload}
                  disabled={
                    !selectedFolderName.trim() ||
                    (uploadMode === "youtube" && !youtubeUrl.trim())
                  }
                  className="tactical-btn tactical-btn-primary w-full disabled:opacity-50"
                >
                  {uploadMode === "youtube"
                    ? "DOWNLOAD & PROCESS VIDEO"
                    : "PROCEED WITH UPLOAD"}
                </button>
                )}
              </div>
            )}

            {!isUploading && (
              <div className="flex gap-2 pt-4 mt-4 border-t border-border">
                <button onClick={handleClose} className="tactical-btn flex-1">
                  CANCEL
                </button>
              </div>
            )}
          </div>
            </motion.div>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={handleFileSelect}
            className="hidden"
            accept=".txt,.pdf,.doc,.docx,.xlsx,.xls,.csv,.md,.markdown,.zip,.dot,.docm,.dotm,.rtf,.odt,.ppt,.pptx,.pptm,.pot,.potx,.potm,.html,.htm,.xml,.epub,.rst,.org,.jpg,.jpeg,.png,.gif,.webp,.bmp,.tiff,.heic,.mp4,.mov,.avi,.mkv,.webm,.flv,.mp3,.wav,.m4a,.aac,.flac,.ogg"
          />
        </>
      )}
    </AnimatePresence>
  );
});

export default UploadModal;
