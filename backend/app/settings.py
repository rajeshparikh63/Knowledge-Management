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



    # Vector Database
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = ""

    # LLM APIs
    OPENAI_API_KEY: str = ""


    # iDrive E2 Storage
    IDRIVEE2_ENDPOINT_URL: str = ""
    IDRIVEE2_ACCESS_KEY_ID: str = ""
    IDRIVEE2_SECRET_ACCESS_KEY: str = ""
    IDRIVEE2_BUCKET_NAME: str = ""

    # Document Processing
    UNSTRUCTURED_API_KEY: str = ""
    UNSTRUCTURED_API_URL: str = ""

    # File Storage
    PRESIGNED_URL_EXPIRATION: int = 604800  # 7 days in seconds (maximum)

    # Observability


    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
