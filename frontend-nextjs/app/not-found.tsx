export default function NotFound() {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-6">
      <div className="max-w-md text-center">
        <h1 className="text-6xl font-bold text-brand mb-4">404</h1>
        <h2 className="text-2xl font-semibold text-foreground mb-4">Page Not Found</h2>
        <p className="text-muted-foreground mb-8">
          The page you're looking for doesn't exist or has been moved.
        </p>
        <a
          href="/"
          className="inline-block px-6 py-3 bg-brand text-foreground font-semibold tracking-wider hover:bg-brand transition-colors"
        >
          RETURN HOME
        </a>
      </div>
    </div>
  );
}
