import hashlib
from pathlib import Path

from fastapi import UploadFile

COPY_CHUNK_SIZE = 1024 * 1024  # 每次读取 1MB


class FileTooLargeError(ValueError):
    pass


class EmptyFileError(ValueError):
    pass


def save_upload_file(
    upload: UploadFile,
    target_path: Path,
    max_size: int,
) -> tuple[int, str]:
    file_size = 0
    file_hash = hashlib.sha256()

    try:
        with target_path.open("wb") as output:
            while chunk := upload.file.read(COPY_CHUNK_SIZE):
                file_size += len(chunk)

                if file_size > max_size:
                    raise FileTooLargeError(
                        f"文件不能超过 {max_size // 1024 // 1024}MB"
                    )

                file_hash.update(chunk)
                output.write(chunk)

        if file_size == 0:
            raise EmptyFileError("不能上传空文件")

        return file_size, file_hash.hexdigest()

    except Exception:
        target_path.unlink(missing_ok=True)
        raise


def validate_pdf_header(file_path: Path) -> None:
    with file_path.open("rb") as file:
        header = file.read(1024)

    if b"%PDF-" not in header:
        raise ValueError("文件扩展名是 PDF，但内容不是有效的 PDF")
