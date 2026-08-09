import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import get_db
from app.db.models import DocumentRecord
from app.schemas import DocumentItem, UploadResponse
from app.services.file_storage import (
    EmptyFileError,
    FileTooLargeError,
    save_upload_file,
    validate_pdf_header,
)
from app.services.ingestion import ingest_file
from app.services.vector_store import delete_document_vectors

router = APIRouter(
    prefix="/api/v1/documents",
    tags=["Documents"],
)

ALLOWED_SUFFIXES = {".pdf", ".txt"}


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_document(
    file: UploadFile,
    db: Session = Depends(get_db),
) -> UploadResponse:
    """上传并索引一个 PDF 或 TXT 文件"""

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="缺少文件名",
        )

    # Path(...).name防止../../xxx 之类的路径穿越
    original_name = Path(
        file.filename
    ).name  # Path是字符串转换的路径对象 .name返回最后一级组件
    suffix = Path(original_name).suffix.lower()

    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail="目前只支持 PDF 和 TXT 文件",
        )

    document_id = uuid.uuid4().hex
    stored_name = f"{document_id}{suffix}"
    saved_path = settings.upload_dir / stored_name

    try:
        file_size, file_hash = save_upload_file(
            upload=file,
            target_path=saved_path,
            max_size=settings.max_upload_size_bytes,
        )

        if suffix == ".pdf":
            validate_pdf_header(saved_path)

        existing = db.scalar(
            select(DocumentRecord).where(DocumentRecord.file_hash == file_hash)
        )

        if existing is not None:
            saved_path.unlink(missing_ok=True)

            raise HTTPException(
                status_code=409,
                detail={"message": "该文件已经上传", "document_id": existing.id},
            )

        record = DocumentRecord(
            id=document_id,
            file_name=original_name,
            stored_name=stored_name,
            file_hash=file_hash,
            file_size=file_size,
            status="processing",
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        result = ingest_file(
            file_path=saved_path,
            original_name=original_name,
            document_id=document_id,
        )

        record.page_count = result["document_count"]
        record.chunk_count = result["chunk_count"]
        record.status = "completed"
        record.error_message = None

        db.commit()

        return UploadResponse(
            document_id=document_id,
            file_name=original_name,
            document_count=result["document_count"],
            chunk_count=result["chunk_count"],
            message="文件已成功加入知识库",
        )

    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=413,
            detail=str(exc),
        ) from exc

    except EmptyFileError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        db.rollback()
        saved_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except IntegrityError as exc:
        db.rollback()
        saved_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=409,
            detail="该文件已经上传",
        ) from exc

    except Exception as exc:
        db.rollback()

        record = db.get(DocumentRecord, document_id)

        if record is not None:
            record.status = "failed"
            record.error_message = str(exc)[:1000]
            db.commit()

        raise HTTPException(
            status_code=500,
            detail="文件处理失败",
        ) from exc

    finally:
        file.file.close()


@router.get(
    "",
    response_model=list[DocumentItem],
)
def list_documents(
    db: Session = Depends(get_db),
) -> list[DocumentRecord]:
    statement = select(DocumentRecord).order_by(DocumentRecord.created_at.desc())

    return list(db.scalars(statement).all())


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
) -> Response:
    record = db.get(DocumentRecord, document_id)

    if record is None:
        raise HTTPException(status_code=404, detail="文件不存在")

    try:
        delete_document_vectors(
            document_id=record.id,
            chunk_count=record.chunk_count,
        )

        stored_path = settings.upload_dir / record.stored_name
        stored_path.unlink(missing_ok=True)

        db.delete(record)
        db.commit()

        return Response(status_code=status.HTTP_204_NO_CONTENT)

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail="删除文档失败",
        ) from exc

    # settings.upload_dir.mkdir(
    #     parents=True,
    #     exist_ok=True,
    # )

    # # 服务器上不直接使用用户提供的文件名
    # saved_path = settings.upload_dir / f"{document_id}{suffix}"

    # try:
    #     with saved_path.open("wb") as output_file: #以二进制写入模式打开该路径对应的文件。如果文件不存在，会自动创建；如果已存在，会覆盖。
    #         shutil.copyfileobj(
    #             file.file,
    #             output_file,
    #         )

    #     result = ingest_file(
    #         file_path=saved_path,
    #         original_name=original_name,
    #         document_id=document_id,
    #     )

    #     return UploadResponse(
    #         document_id=document_id,
    #         file_name=original_name,
    #         document_count=result["document_count"],
    #         chunk_count=result["chunk_count"],
    #         message="文件已成功加入知识库",
    #     )

    # except ValueError as exc:
    #     saved_path.unlink(missing_ok=True)

    #     raise HTTPException(
    #         status_code=400,
    #         detail=str(exc),
    #     ) from exc

    # except Exception as exc:
    #     saved_path.unlink(missing_ok=True)

    #     # 产生环境应记录日志，但不要把完整异常返回给前端
    #     raise HTTPException(
    #         status_code=500,
    #         detail="文件处理失败",
    #     ) from exc

    # finally:
    #     file.file.close()
