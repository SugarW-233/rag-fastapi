import os
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# 必须放在导入 app.main 之前，防止配置初始化时报缺少密钥。
os.environ.setdefault(
    "OPENAI_API_KEY",
    "test-api-key-not-real",
)

from app.core.config import settings
from app.db.database import Base, get_db
from app.main import app


@pytest.fixture
def db_session(
    tmp_path: Path,
) -> Generator[Session, None, None]:
    """每个测试使用一个全新的临时 SQLite 数据库。"""

    database_path = tmp_path / "test.db"

    test_engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={
            "check_same_thread": False,
        },
    )

    TestingSessionLocal = sessionmaker(
        bind=test_engine,
        autoflush=False,
        expire_on_commit=False,
    )

    Base.metadata.create_all(bind=test_engine)

    with TestingSessionLocal() as session:
        yield session

    test_engine.dispose()


@pytest.fixture
def client(
    db_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[TestClient, None, None]:
    """使用测试数据库和临时存储目录的 API 客户端。"""

    upload_dir = tmp_path / "uploads"
    chroma_dir = tmp_path / "chroma"

    upload_dir.mkdir()
    chroma_dir.mkdir()

    monkeypatch.setattr(
        settings,
        "upload_dir",
        upload_dir,
    )

    monkeypatch.setattr(
        settings,
        "chroma_dir",
        chroma_dir,
    )

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
