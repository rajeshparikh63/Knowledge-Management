"""
Public, unauthenticated pages required for Google OAuth verification.

Google requires the app's home page, privacy policy, and terms of service to
be live, publicly accessible URLs hosted on the app's authorized domain. These
are served from the backend root (the railway domain registered as the
authorized domain in the OAuth client).

Routes (no /api prefix — they live at the domain root):
    GET /          → home / landing page
    GET /privacy   → privacy policy (with Google Limited Use disclosure)
    GET /terms     → terms of service
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["public"])

APP_NAME = "SoldierIQ"
SUPPORT_EMAIL = "gargkeshav5042004@gmail.com"
LAST_UPDATED = "June 25, 2026"

# Shared page chrome (dark, on-brand)
_STYLE = """
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: #0a0a0a; color: #e4e4e7;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.7;
  }
  .wrap { max-width: 760px; margin: 0 auto; padding: 56px 24px 96px; }
  header { display: flex; align-items: center; gap: 12px; margin-bottom: 40px; }
  .logo {
    width: 40px; height: 40px; border-radius: 9px; background: #f59e0b;
    display: flex; align-items: center; justify-content: center;
    color: #0a0a0a; font-weight: 800; font-size: 15px; letter-spacing: -0.5px;
  }
  .brand { font-size: 18px; font-weight: 700; letter-spacing: -0.3px; }
  h1 { font-size: 28px; letter-spacing: -0.5px; margin: 0 0 8px; }
  h2 { font-size: 18px; margin: 36px 0 10px; color: #fafafa; }
  p, li { color: #a1a1aa; font-size: 15px; }
  a { color: #f59e0b; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .muted { color: #71717a; font-size: 13px; }
  .pill {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    background: #18181b; border: 1px solid #27272a; font-size: 12px;
    color: #a1a1aa; margin-bottom: 24px;
  }
  .card {
    background: #111113; border: 1px solid #27272a; border-radius: 12px;
    padding: 20px 22px; margin: 18px 0;
  }
  nav { margin-top: 48px; padding-top: 24px; border-top: 1px solid #1f1f23; }
  nav a { margin-right: 20px; font-size: 14px; }
  code { background:#18181b; padding:2px 6px; border-radius:5px; font-size:13px; }
"""


def _page(title: str, inner: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} · {APP_NAME}</title>
  <style>{_STYLE}</style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="logo">SIQ</div>
      <div class="brand">{APP_NAME}</div>
    </header>
    {inner}
    <nav>
      <a href="/">Home</a>
      <a href="/privacy">Privacy Policy</a>
      <a href="/terms">Terms of Service</a>
    </nav>
  </div>
</body>
</html>"""


@router.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    inner = f"""
    <span class="pill">Knowledge management for teams</span>
    <h1>{APP_NAME}</h1>
    <p>
      {APP_NAME} is a private knowledge-management platform. It lets you import
      your own documents — including files from Google Drive — and then search,
      summarize, and ask questions across them using AI. Your documents become a
      searchable, conversational knowledge base.
    </p>

    <h2>What it does</h2>
    <div class="card">
      <p style="margin:0 0 10px;"><strong style="color:#fafafa;">Import</strong> —
      Upload files directly, or connect Google Drive and choose which folders to
      bring in.</p>
      <p style="margin:0 0 10px;"><strong style="color:#fafafa;">Understand</strong> —
      {APP_NAME} extracts text, builds a knowledge graph of the entities and
      relationships in your documents, and makes everything searchable.</p>
      <p style="margin:0;"><strong style="color:#fafafa;">Ask</strong> —
      Chat with your documents. Get grounded answers with citations back to the
      source files.</p>
    </div>

    <h2>Google Drive access</h2>
    <p>
      {APP_NAME} requests read-only access to your Google Drive so it can import
      the documents and folders <em>you choose</em>. We never modify or delete
      anything in your Drive. See our
      <a href="/privacy">Privacy Policy</a> for exactly how Google user data is
      handled.
    </p>

    <h2>Contact</h2>
    <p>Questions? Email <a href="mailto:{SUPPORT_EMAIL}">{SUPPORT_EMAIL}</a>.</p>
    """
    return HTMLResponse(_page("Home", inner))


@router.get("/privacy", response_class=HTMLResponse)
async def privacy() -> HTMLResponse:
    inner = f"""
    <h1>Privacy Policy</h1>
    <p class="muted">Last updated: {LAST_UPDATED}</p>

    <p>
      This Privacy Policy explains how {APP_NAME} ("we", "us") collects, uses,
      stores, and protects your information, including data accessed through
      Google APIs. By using {APP_NAME}, you agree to this policy.
    </p>

    <h2>Information we collect</h2>
    <ul>
      <li><strong>Account information</strong> — your name, email, and
        organization, used to authenticate you and scope your data.</li>
      <li><strong>Google account data</strong> — when you connect Google Drive,
        we receive your email and name (to show which account is connected) and
        an OAuth token that grants <strong>read-only</strong> access to your
        Drive.</li>
      <li><strong>Document content</strong> — the files and folders you choose to
        import from Google Drive or upload directly, and data we derive from
        them (extracted text, embeddings, and a knowledge graph).</li>
    </ul>

    <h2>How we use Google user data</h2>
    <ul>
      <li>We use <code>drive.readonly</code> access solely to <strong>list and
        read the files and folders you explicitly select</strong> for import.</li>
      <li>Imported content is processed to provide the core features you
        requested: full-text search, summarization, and AI chat over your own
        documents.</li>
      <li>We do <strong>not</strong> modify, create, or delete anything in your
        Google Drive.</li>
      <li>We do <strong>not</strong> use Google user data for advertising, and we
        do <strong>not</strong> sell it.</li>
    </ul>

    <h2>Limited Use disclosure</h2>
    <div class="card">
      <p style="margin:0;">
        {APP_NAME}'s use and transfer to any other app of information received
        from Google APIs will adhere to the
        <a href="https://developers.google.com/terms/api-services-user-data-policy">Google
        API Services User Data Policy</a>, including the Limited Use
        requirements.
      </p>
    </div>

    <h2>Third-party processing</h2>
    <p>
      To provide search and AI features, document content may be sent to
      third-party AI processors (such as OpenAI and OpenRouter) to generate
      embeddings and extract structure. These processors act on our behalf to
      deliver the features you requested and are contractually restricted from
      using your content to train generalized models. We do not transfer Google
      user data to any third party except as necessary to provide or improve
      these user-facing features, to comply with applicable law, or as part of a
      merger or acquisition, consistent with the Limited Use requirements.
    </p>

    <h2>Storage and security</h2>
    <p>
      Your data is stored in access-controlled databases and object storage.
      OAuth tokens are stored securely and used only to access the data you
      authorized. Access is scoped to your account and organization.
    </p>

    <h2>Data retention and deletion</h2>
    <p>
      You can disconnect Google Drive at any time, which removes our stored
      access tokens. You may request deletion of your imported documents and
      derived data by contacting us at
      <a href="mailto:{SUPPORT_EMAIL}">{SUPPORT_EMAIL}</a>.
    </p>

    <h2>Contact</h2>
    <p>For privacy questions, email
      <a href="mailto:{SUPPORT_EMAIL}">{SUPPORT_EMAIL}</a>.</p>
    """
    return HTMLResponse(_page("Privacy Policy", inner))


@router.get("/terms", response_class=HTMLResponse)
async def terms() -> HTMLResponse:
    inner = f"""
    <h1>Terms of Service</h1>
    <p class="muted">Last updated: {LAST_UPDATED}</p>

    <p>
      These Terms of Service ("Terms") govern your use of {APP_NAME}. By
      accessing or using the service, you agree to these Terms.
    </p>

    <h2>Use of the service</h2>
    <p>
      {APP_NAME} provides document import, search, and AI-assisted question
      answering over content you own or are authorized to use. You are
      responsible for ensuring you have the right to import and process any
      content you bring into the service.
    </p>

    <h2>Google Drive</h2>
    <p>
      When you connect Google Drive, you grant {APP_NAME} read-only access to
      import the files and folders you select. {APP_NAME} will not modify or
      delete your Drive content. You may revoke access at any time from within
      the app or from your
      <a href="https://myaccount.google.com/permissions">Google Account
      permissions</a> page.
    </p>

    <h2>Acceptable use</h2>
    <ul>
      <li>Do not use the service for unlawful purposes.</li>
      <li>Do not attempt to access data belonging to other users or
        organizations.</li>
      <li>Do not upload content you do not have the right to use.</li>
    </ul>

    <h2>Disclaimer</h2>
    <p>
      {APP_NAME} is provided "as is" without warranties of any kind. AI-generated
      answers may contain errors; verify important information against the
      original source documents.
    </p>

    <h2>Changes</h2>
    <p>
      We may update these Terms from time to time. Continued use of the service
      after changes take effect constitutes acceptance of the revised Terms.
    </p>

    <h2>Contact</h2>
    <p>Questions about these Terms? Email
      <a href="mailto:{SUPPORT_EMAIL}">{SUPPORT_EMAIL}</a>.</p>
    """
    return HTMLResponse(_page("Terms of Service", inner))
