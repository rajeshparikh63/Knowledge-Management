from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv(".env", override=True)


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "SoldierIQ Backend"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # Database
    MONGODB_URL: str = ""
    MONGODB_DATABASE: str = "soldieriq"



    # LLM APIs
    OPENAI_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    ELEVENLABS_API_KEY: str = ""
    FUNCTION_API_KEY: str = ""  


    # iDrive E2 Storage
    IDRIVEE2_ENDPOINT_URL: str = ""
    IDRIVEE2_ACCESS_KEY_ID: str = ""
    IDRIVEE2_SECRET_ACCESS_KEY: str = ""
    IDRIVEE2_BUCKET_NAME: str = ""

    # Document Processing
    UNSTRUCTURED_API_KEY: str = ""
    UNSTRUCTURED_API_URL: str = ""
    # For image-based / scanned PDFs (no text layer), parse with Unstructured's
    # native VLM strategy: a vision model reads the rendered pages. Provider must
    # be one Unstructured supports (openai | anthropic | google | bedrock |
    # vertexai | azure_openai); model is that provider's vision model id.
    VLM_MODEL_PROVIDER: str = "openai"
    VLM_MODEL: str = "gpt-4o-mini"  # smaller/cheaper vision model; gpt-4o for higher fidelity

    # Video Processing
    VIDEO_TARGET_FPS: int = 4  # Frame extraction rate (frames per second)
    VIDEO_SSIM_THRESHOLD: float = 0.6  # Scene detection sensitivity (0-1)
    VIDEO_MIN_CLIP_DURATION: float = 10.0  # Minimum scene length in seconds
    VIDEO_MAX_CLIP_DURATION: float = 45.0  # Maximum scene length in seconds

    # File Storage
    PRESIGNED_URL_EXPIRATION: int = 604800  # 7 days in seconds (maximum)

    # Redis Configuration (for Celery)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""

    # Celery Configuration
    CELERY_BROKER_URL: str = ""  # Will be set from REDIS_HOST/PORT
    CELERY_RESULT_BACKEND: str = ""  # Will be set from REDIS_HOST/PORT

    # Keycloak Authentication
    KEYCLOAK_SERVER_URL: str = "http://localhost:8080"
    KEYCLOAK_REALM: str = "SoldierIQ"
    KEYCLOAK_CLIENT_ID: str = "soldieriq-backend"
    KEYCLOAK_CLIENT_SECRET: str = ""

    # Keycloak Admin (for creating users programmatically)
    KEYCLOAK_ADMIN_USERNAME: str = "admin"
    KEYCLOAK_ADMIN_PASSWORD: str = "admin"

    POSTGRES_URL: str = ""

    # FalkorDB
    GRAPH_DATABASE_URL: str = "localhost"
    GRAPH_DATABASE_PORT: int = 6379
    GRAPH_DATABASE_USERNAME: str = ""
    GRAPH_DATABASE_PASSWORD: str = ""
    GRAPH_DATABASE_SSL: bool = False

    # Embeddings + extraction
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIM: int = 1536
    EXTRACTION_MODEL: str = "google/gemini-2.5-flash"
    ONTOLOGY_MODEL: str = "google/gemini-2.5-flash"

    # Ingestion
    # Larger chunks → fewer LLM calls per document (cuts wall time roughly
    # linearly). 5000 keeps enough context for accurate entity extraction
    # without overwhelming the model.
    CHUNK_SIZE: int = 5000
    CHUNK_OVERLAP: int = 200
    # Higher concurrency = more parallel LLM calls in flight. OpenRouter
    # comfortably handles 30–50 concurrent requests per key for Gemini Flash.
    # If you start seeing 429s, dial back to 20.
    LLM_CONCURRENCY: int = 30
    MIN_TRIPLE_CONFIDENCE: float = 0.7
    ENTITY_RESOLUTION_THRESHOLD: float = 0.10  # cosine distance — lower = stricter merge
    ENTITY_CACHE_DIR: str = "./data/entity_cache"  # per-org FAISS indices live here

    # ---- Google Drive connector ----------------------------------------
    # OAuth client created in Google Cloud Console. Paste values into .env.
    # REDIRECT_URI must EXACTLY match what's registered in the OAuth client
    # (e.g. http://localhost:8000/api/google-drive/callback for dev).
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/google-drive/callback"
    # drive.readonly is the minimum scope; openid+email are for identifying
    # which Google account got connected (shown in our UI).
    GOOGLE_SCOPES: str = "openid email profile https://www.googleapis.com/auth/drive.readonly"
    # Where the OAuth flow redirects the user back to in the frontend.
    FRONTEND_URL: str = "http://localhost:3000"

    # ---- Composio (manages SharePoint OAuth + tokens for us) -------------
    # We don't store Microsoft tokens ourselves; Composio owns the OAuth flow
    # and refresh. We just call its SDK with our user_id as the Composio
    # "entity". The auth config (an OAuth app registered in the Composio
    # dashboard) is referenced by id; the share_point one requires a per-user
    # `subdomain` (tenant name) collected at connect time.
    COMPOSIO_API_KEY: str = ""
    COMPOSIO_SHAREPOINT_AUTH_CONFIG_ID: str = "ac_yZpaj2ORI7Fx"

    # Observability

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
