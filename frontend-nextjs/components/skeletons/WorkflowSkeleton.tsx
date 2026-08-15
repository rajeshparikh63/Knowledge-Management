import { Skeleton } from "@/components/ui/skeleton";

export default function WorkflowSkeleton() {
  return (
    <div className="flex flex-col h-full bg-slate-900/80 border border-amber-400/20 p-4 relative">
      {/* Corner brackets */}
      <div className="absolute top-0 left-0 w-3 h-3 border-t-2 border-l-2 border-amber-400/60 z-10" />
      <div className="absolute top-0 right-0 w-3 h-3 border-t-2 border-r-2 border-amber-400/60 z-10" />
      <div className="absolute bottom-0 left-0 w-3 h-3 border-b-2 border-l-2 border-amber-400/60 z-10" />
      <div className="absolute bottom-0 right-0 w-3 h-3 border-b-2 border-r-2 border-amber-400/60 z-10" />

      {/* Panel header */}
      <Skeleton className="h-5 w-32 bg-slate-800 mb-4" />

      {/* Workflow cards grid */}
      <div className="grid grid-cols-1 gap-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <div
            key={i}
            className="p-3 border border-amber-400/10 bg-slate-950/50 rounded space-y-2"
          >
            <div className="flex items-center gap-2">
              <Skeleton className="h-5 w-5 bg-slate-800 rounded" />
              <Skeleton className="h-4 w-28 bg-slate-800" />
            </div>
            <Skeleton className="h-3 w-full bg-slate-800" />
            <Skeleton className="h-3 w-3/4 bg-slate-800" />
          </div>
        ))}
      </div>
    </div>
  );
}
