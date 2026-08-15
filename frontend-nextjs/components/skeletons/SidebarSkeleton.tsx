import { Skeleton } from "@/components/ui/skeleton";

export default function SidebarSkeleton() {
  return (
    <div className="flex flex-col h-full bg-slate-900/80 border border-amber-400/20 p-4 relative">
      {/* Corner brackets */}
      <div className="absolute top-0 left-0 w-3 h-3 border-t-2 border-l-2 border-amber-400/60 z-10" />
      <div className="absolute top-0 right-0 w-3 h-3 border-t-2 border-r-2 border-amber-400/60 z-10" />
      <div className="absolute bottom-0 left-0 w-3 h-3 border-b-2 border-l-2 border-amber-400/60 z-10" />
      <div className="absolute bottom-0 right-0 w-3 h-3 border-b-2 border-r-2 border-amber-400/60 z-10" />

      {/* Upload button placeholder */}
      <Skeleton className="h-9 w-full bg-slate-800 rounded mb-5" />

      {/* Search bar placeholder */}
      <Skeleton className="h-8 w-full bg-slate-800 rounded mb-4" />

      {/* Folder tree */}
      <div className="space-y-3 flex-1">
        {/* Folder 1 */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Skeleton className="h-4 w-4 bg-slate-800" />
            <Skeleton className="h-4 w-32 bg-slate-800" />
          </div>
          <div className="ml-6 space-y-2">
            <div className="flex items-center gap-2">
              <Skeleton className="h-3 w-3 bg-slate-800" />
              <Skeleton className="h-3 w-28 bg-slate-800" />
            </div>
            <div className="flex items-center gap-2">
              <Skeleton className="h-3 w-3 bg-slate-800" />
              <Skeleton className="h-3 w-24 bg-slate-800" />
            </div>
          </div>
        </div>

        {/* Folder 2 */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Skeleton className="h-4 w-4 bg-slate-800" />
            <Skeleton className="h-4 w-36 bg-slate-800" />
          </div>
          <div className="ml-6 space-y-2">
            <div className="flex items-center gap-2">
              <Skeleton className="h-3 w-3 bg-slate-800" />
              <Skeleton className="h-3 w-20 bg-slate-800" />
            </div>
            <div className="flex items-center gap-2">
              <Skeleton className="h-3 w-3 bg-slate-800" />
              <Skeleton className="h-3 w-30 bg-slate-800" />
            </div>
            <div className="flex items-center gap-2">
              <Skeleton className="h-3 w-3 bg-slate-800" />
              <Skeleton className="h-3 w-24 bg-slate-800" />
            </div>
          </div>
        </div>

        {/* Folder 3 */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Skeleton className="h-4 w-4 bg-slate-800" />
            <Skeleton className="h-4 w-28 bg-slate-800" />
          </div>
          <div className="ml-6 space-y-2">
            <div className="flex items-center gap-2">
              <Skeleton className="h-3 w-3 bg-slate-800" />
              <Skeleton className="h-3 w-26 bg-slate-800" />
            </div>
            <div className="flex items-center gap-2">
              <Skeleton className="h-3 w-3 bg-slate-800" />
              <Skeleton className="h-3 w-22 bg-slate-800" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
