import { Skeleton } from "@/components/ui/skeleton";

export default function WorkflowSkeleton() {
  return (
    <div className="flex flex-col h-full bg-card/80 border border-brand/20 p-4 relative">
      {/* Corner brackets */}
      <div className="absolute top-0 left-0 w-3 h-3 border-t-2 border-l-2 border-brand/60 z-10" />
      <div className="absolute top-0 right-0 w-3 h-3 border-t-2 border-r-2 border-brand/60 z-10" />
      <div className="absolute bottom-0 left-0 w-3 h-3 border-b-2 border-l-2 border-brand/60 z-10" />
      <div className="absolute bottom-0 right-0 w-3 h-3 border-b-2 border-r-2 border-brand/60 z-10" />

      {/* Panel header */}
      <Skeleton className="h-5 w-32 bg-muted mb-4" />

      {/* Workflow cards grid */}
      <div className="grid grid-cols-1 gap-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <div
            key={i}
            className="p-3 border border-brand/10 bg-background/50 rounded space-y-2"
          >
            <div className="flex items-center gap-2">
              <Skeleton className="h-5 w-5 bg-muted rounded" />
              <Skeleton className="h-4 w-28 bg-muted" />
            </div>
            <Skeleton className="h-3 w-full bg-muted" />
            <Skeleton className="h-3 w-3/4 bg-muted" />
          </div>
        ))}
      </div>
    </div>
  );
}
