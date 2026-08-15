import { Skeleton } from "@/components/ui/skeleton";

export default function ChatSkeleton() {
  return (
    <div className="flex flex-col h-full bg-card/80 border border-brand/20 relative">
      {/* Corner brackets */}
      <div className="absolute top-0 left-0 w-3 h-3 border-t-2 border-l-2 border-brand/60 z-10" />
      <div className="absolute top-0 right-0 w-3 h-3 border-t-2 border-r-2 border-brand/60 z-10" />
      <div className="absolute bottom-0 left-0 w-3 h-3 border-b-2 border-l-2 border-brand/60 z-10" />
      <div className="absolute bottom-0 right-0 w-3 h-3 border-b-2 border-r-2 border-brand/60 z-10" />

      {/* Messages area */}
      <div className="flex-1 p-4 space-y-5 overflow-hidden">
        {/* Message from system (left) */}
        <div className="flex items-start gap-3">
          <Skeleton className="h-8 w-8 rounded bg-muted shrink-0" />
          <div className="space-y-2 max-w-[70%]">
            <Skeleton className="h-4 w-48 bg-muted" />
            <Skeleton className="h-4 w-64 bg-muted" />
            <Skeleton className="h-4 w-36 bg-muted" />
          </div>
        </div>

        {/* Message from user (right) */}
        <div className="flex items-start gap-3 justify-end">
          <div className="space-y-2 max-w-[60%]">
            <Skeleton className="h-4 w-44 bg-muted ml-auto" />
            <Skeleton className="h-4 w-28 bg-muted ml-auto" />
          </div>
          <Skeleton className="h-8 w-8 rounded bg-muted shrink-0" />
        </div>

        {/* Message from system (left) */}
        <div className="flex items-start gap-3">
          <Skeleton className="h-8 w-8 rounded bg-muted shrink-0" />
          <div className="space-y-2 max-w-[70%]">
            <Skeleton className="h-4 w-56 bg-muted" />
            <Skeleton className="h-4 w-40 bg-muted" />
          </div>
        </div>

        {/* Message from user (right) */}
        <div className="flex items-start gap-3 justify-end">
          <div className="space-y-2 max-w-[60%]">
            <Skeleton className="h-4 w-52 bg-muted ml-auto" />
            <Skeleton className="h-4 w-32 bg-muted ml-auto" />
            <Skeleton className="h-4 w-20 bg-muted ml-auto" />
          </div>
          <Skeleton className="h-8 w-8 rounded bg-muted shrink-0" />
        </div>
      </div>

      {/* Input area placeholder */}
      <div className="p-4 border-t border-brand/20">
        <div className="flex items-center gap-3">
          <Skeleton className="h-10 flex-1 bg-muted rounded" />
          <Skeleton className="h-10 w-10 bg-muted rounded" />
        </div>
      </div>
    </div>
  );
}
