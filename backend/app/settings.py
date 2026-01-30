from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv(".env", override=True)


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "SoldierIQ Backend"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = ""
    DB_NAME: str = "soldieriq"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # Vector Database
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = ""

    # LLM APIs
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

    # AWS
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_S3_BUCKET: str = ""
    AWS_S3_REGION: str = "us-east-1"

    # Auth
    JWT_SECRET: str = ""
    AWS_COGNITO_USER_POOL_ID: str = ""
    AWS_COGNITO_CLIENT_ID: str = ""
    AWS_COGNITO_REGION: str = "us-east-1"

    # Observability
    SENTRY_DSN: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
