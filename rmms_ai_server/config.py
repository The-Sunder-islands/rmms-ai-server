from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AI_SERVER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 8420
    api_key: str = ""
    max_concurrent_tasks: int = 4
    max_queue_size: int = 20
    max_upload_mb: int = 500
    task_ttl_seconds: int = 3600
    cleanup_interval_seconds: int = 300
    model_cache_dir: str = ""
    upload_dir: str = ""
    output_dir: str = ""
    mdns_enabled: bool = True
    mdns_name: str = "RMMS AI Server"
    rate_limit_enabled: bool = False
    rate_limit_max_requests: int = 60
    rate_limit_per_second: float = 10.0
    log_level: str = "INFO"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def resolved_upload_dir(self) -> Path:
        if self.upload_dir:
            return Path(self.upload_dir)
        return Path.cwd() / "uploads"

    @property
    def resolved_output_dir(self) -> Path:
        if self.output_dir:
            return Path(self.output_dir)
        return Path.cwd() / "outputs"

    @property
    def resolved_model_cache_dir(self) -> Path:
        if self.model_cache_dir:
            return Path(self.model_cache_dir)
        return Path.cwd() / "models"

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key.strip())


settings = Settings()
