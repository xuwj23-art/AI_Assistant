"""
配置管理模块

使用 pydantic-settings 从环境变量和 .env 文件加载配置。
这是工程化项目的标准做法,便于在不同环境(开发/测试/生产)之间切换。
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用程序全局配置"""

    # OpenAI Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"

    # Application Configuration
    app_env: str = "development"
    log_level: str = "INFO"

    # ChromaDB Configuration
    chroma_persist_dir: str = "./data/chroma"

    # BERTopic Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    min_topic_size: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# 全局单例配置对象
settings = Settings()

