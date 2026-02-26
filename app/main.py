import asyncio
import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional

import ffmpeg
import whisper
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# --- Константы и базовые пути проекта ---
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "app" / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"

for directory in (STATIC_DIR, UPLOAD_DIR, OUTPUT_DIR, DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# --- Pydantic-схемы для API ---
class SubtitleEntry(BaseModel):
    """Одна запись субтитра в человеко-редактируемом формате."""

    id: int = Field(ge=1, description="Порядковый номер субтитра")
    start: float = Field(ge=0, description="Время старта в секундах")
    end: float = Field(gt=0, description="Время окончания в секундах")
    text: str = Field(min_length=1, description="Текст субтитра")


class SubtitlePayload(BaseModel):
    """Контейнер для обновления списка субтитров."""

    subtitles: List[SubtitleEntry]


class GenerateRequest(BaseModel):
    """Параметры запуска AI-транскрибации."""

    file_id: str
    model_size: Literal["tiny", "base", "small", "medium", "large"] = "base"
    language: Optional[str] = "ru"


class ExportRequest(BaseModel):
    """Параметры экспорта видео с вшитыми субтитрами."""

    file_id: str
    font_name: str = Field(default="Arial", min_length=1, max_length=64)
    font_size: int = Field(default=28, ge=12, le=96)
    subtitle_position: Literal["top", "middle", "bottom"] = "bottom"
    margin_v: int = Field(default=30, ge=0, le=300)
    primary_color: str = Field(default="&H00FFFFFF", pattern=r"^&H[0-9A-Fa-f]{8}$")
    outline_color: str = Field(default="&H00000000", pattern=r"^&H[0-9A-Fa-f]{8}$")
    outline: int = Field(default=2, ge=0, le=8)


class TaskState(BaseModel):
    """Унифицированная модель состояния фоновой задачи."""

    status: Literal["queued", "processing", "completed", "failed"]
    progress: int = Field(ge=0, le=100)
    result: Optional[dict] = None
    error: Optional[str] = None


# --- Глобальные in-memory хранилища ---
tasks_store: Dict[str, TaskState] = {}
files_store: Dict[str, Dict[str, str]] = {}


app = FastAPI(title="AI Subtitle Editor", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Разрешаем запросы со всех origin для удобства локальной разработки.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def now_utc() -> datetime:
    """Возвращает текущее время в UTC для единого формата хранения."""

    return datetime.now(timezone.utc)


def parse_srt_time(value: str) -> float:
    """Преобразует формат SRT-времени `HH:MM:SS,mmm` в секунды."""

    hh, mm, rest = value.split(":")
    ss, msec = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(msec) / 1000


def format_srt_time(seconds: float) -> str:
    """Преобразует секунды в строку формата `HH:MM:SS,mmm`."""

    total_ms = int(seconds * 1000)
    hh = total_ms // 3_600_000
    mm = (total_ms % 3_600_000) // 60_000
    ss = (total_ms % 60_000) // 1000
    ms = total_ms % 1000
    return f"{hh:02}:{mm:02}:{ss:02},{ms:03}"


def srt_path(file_id: str) -> Path:
    return OUTPUT_DIR / f"{file_id}.srt"


def escape_subtitles_filter_path(path: Path) -> str:
    """Экранирует путь для безопасной подстановки в ffmpeg subtitles filter."""

    resolved = path.resolve()
    try:
        # Используем путь относительно корня проекта, чтобы избежать проблем с `C:` на Windows.
        # FFmpeg корректно читает относительные пути вида `outputs/file.srt`.
        raw = resolved.relative_to(BASE_DIR.resolve()).as_posix()
    except ValueError:
        raw = resolved.as_posix()

    # Для ffmpeg filtergraph экранируем только потенциально проблемные символы.
    escaped = raw.replace("'", r"\'")
    # Если всё же пришел абсолютный Windows-путь, экранируем двоеточие диска один раз.
    if re.match(r"^[A-Za-z]:/", escaped):
        escaped = escaped.replace(":", r"\:", 1)
    return escaped


def save_subtitles(file_id: str, subtitles: List[SubtitleEntry]) -> Path:
    """Сохраняет список субтитров на диск в формате SRT."""

    output_path = srt_path(file_id)
    lines: List[str] = []
    for idx, entry in enumerate(subtitles, start=1):
        lines.append(str(idx))
        lines.append(f"{format_srt_time(entry.start)} --> {format_srt_time(entry.end)}")
        lines.append(entry.text.strip())
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def parse_srt_file(path: Path) -> List[SubtitleEntry]:
    """Разбирает .srt файл и возвращает массив структурированных сегментов."""

    content = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not content:
        return []

    blocks = re.split(r"\n\s*\n", content)
    subtitles: List[SubtitleEntry] = []

    for idx, block in enumerate(blocks, start=1):
        rows = [row.strip() for row in block.splitlines() if row.strip()]
        if len(rows) < 2:
            continue

        time_line = rows[1] if rows[0].isdigit() else rows[0]
        text_start = 2 if rows[0].isdigit() else 1

        if "-->" not in time_line:
            continue

        start_raw, end_raw = [part.strip() for part in time_line.split("-->")]
        subtitles.append(
            SubtitleEntry(
                id=idx,
                start=parse_srt_time(start_raw),
                end=parse_srt_time(end_raw),
                text=" ".join(rows[text_start:]),
            )
        )

    return subtitles


def safe_video_path(file_id: str) -> Path:
    """Валидирует file_id и возвращает путь к загруженному видео."""

    payload = files_store.get(file_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Файл не найден")
    video_path = Path(payload["video_path"]) if "video_path" in payload else None
    if not video_path or not video_path.exists():
        raise HTTPException(status_code=404, detail="Видео отсутствует")
    return video_path


def safe_subtitle_path(file_id: str) -> Path:
    """Валидирует file_id и возвращает путь к связанному .srt файлу."""

    path = srt_path(file_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Субтитры не найдены")
    return path


def _run_whisper_task(task_id: str, file_id: str, model_size: str, language: Optional[str]) -> None:
    """Фоновая задача: извлекает текст из видео и сохраняет SRT."""

    try:
        tasks_store[task_id] = TaskState(status="processing", progress=10)
        video_path = safe_video_path(file_id)

        tasks_store[task_id] = TaskState(status="processing", progress=30)
        model = whisper.load_model(model_size)

        tasks_store[task_id] = TaskState(status="processing", progress=60)
        result = model.transcribe(str(video_path), language=language)

        segments = []
        for idx, seg in enumerate(result.get("segments", []), start=1):
            segments.append(
                SubtitleEntry(
                    id=idx,
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=seg.get("text", "").strip(),
                )
            )

        save_subtitles(file_id, segments)
        files_store.setdefault(file_id, {})["updated_at"] = now_utc().isoformat()

        tasks_store[task_id] = TaskState(
            status="completed",
            progress=100,
            result={"file_id": file_id, "segments": len(segments)},
        )
    except Exception as exc:  # noqa: BLE001
        tasks_store[task_id] = TaskState(status="failed", progress=100, error=str(exc))


def _build_force_style(payload: ExportRequest) -> str:
    """Собирает ASS force_style для кастомизации шрифта и позиции субтитров."""

    alignment_map = {
        "bottom": 2,
        "middle": 5,
        "top": 8,
    }
    alignment = alignment_map[payload.subtitle_position]
    # Важно: в force_style нельзя передавать запятые и одинарные кавычки без экранирования,
    # иначе ffmpeg/libass может некорректно распарсить стиль и применить дефолт.
    safe_font_name = payload.font_name.replace("'", "").replace(",", " ").strip() or "Arial"

    style = {
        "FontName": safe_font_name,
        "FontSize": payload.font_size,
        "PrimaryColour": payload.primary_color,
        "OutlineColour": payload.outline_color,
        "Outline": payload.outline,
        "Alignment": alignment,
        "MarginV": payload.margin_v,
        "WrapStyle": 2,
    }
    # Для force_style ожидается стандартный список `k=v,k=v`.
    return ",".join(f"{key}={value}" for key, value in style.items())


def _run_export_task(task_id: str, payload: ExportRequest) -> None:
    """Фоновая задача: вшивает субтитры в видео через FFmpeg."""

    try:
        tasks_store[task_id] = TaskState(status="processing", progress=10)
        video_path = safe_video_path(payload.file_id)
        subtitles_path = safe_subtitle_path(payload.file_id)

        output_path = OUTPUT_DIR / f"{payload.file_id}_burned.mp4"
        tasks_store[task_id] = TaskState(status="processing", progress=60)

        subtitles_filter_path = escape_subtitles_filter_path(subtitles_path)
        force_style = _build_force_style(payload)

        src = ffmpeg.input(str(video_path))

        # Для SRT libass использует виртуальное разрешение (PlayRes) и может
        # чрезмерно масштабировать шрифт на вертикальных видео.
        # Передаем original_size реального видео, чтобы размер шрифта оставался корректным.
        filter_kwargs = {"force_style": force_style}
        probe_data = ffmpeg.probe(str(video_path))
        video_stream_info = next(
            (stream for stream in probe_data.get("streams", []) if stream.get("codec_type") == "video"),
            None,
        )
        if video_stream_info:
            width = int(video_stream_info.get("width", 0) or 0)
            height = int(video_stream_info.get("height", 0) or 0)
            if width > 0 and height > 0:
                filter_kwargs["original_size"] = f"{width}x{height}"

        video_stream = src.video.filter("subtitles", subtitles_filter_path, **filter_kwargs)
        audio_stream = src.audio

        (
            ffmpeg.output(
                video_stream,
                audio_stream,
                str(output_path),
                vcodec="libx264",
                acodec="aac",
                movflags="+faststart",
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        tasks_store[task_id] = TaskState(
            status="completed",
            progress=100,
            result={"download_url": f"/outputs/{output_path.name}"},
        )
    except ffmpeg.Error as exc:
        stderr_text = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        message = stderr_text.strip() or str(exc)
        tasks_store[task_id] = TaskState(status="failed", progress=100, error=message)
    except Exception as exc:  # noqa: BLE001
        tasks_store[task_id] = TaskState(status="failed", progress=100, error=str(exc))


async def cleanup_loop() -> None:
    """Периодически удаляет устаревшие записи и файлы старше 1 часа."""

    while True:
        try:
            border = now_utc() - timedelta(hours=1)
            to_delete: List[str] = []

            for file_id, data in files_store.items():
                updated_at = datetime.fromisoformat(data.get("updated_at", now_utc().isoformat()))
                if updated_at < border:
                    to_delete.append(file_id)

            for file_id in to_delete:
                data = files_store.pop(file_id, {})
                for key in ("video_path",):
                    item = data.get(key)
                    if item and Path(item).exists():
                        Path(item).unlink(missing_ok=True)

                srt_file = srt_path(file_id)
                srt_file.unlink(missing_ok=True)
                burned = OUTPUT_DIR / f"{file_id}_burned.mp4"
                burned.unlink(missing_ok=True)

            await asyncio.sleep(300)
        except Exception:
            await asyncio.sleep(60)


@app.on_event("startup")
async def on_startup() -> None:
    """Инициализирует фоновую очистку при старте приложения."""

    app.state.cleanup_task = asyncio.create_task(cleanup_loop())


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """Корректно останавливает фоновую корутину при выключении сервиса."""

    task = getattr(app.state, "cleanup_task", None)
    if task:
        task.cancel()


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Возвращает основной HTML интерфейс приложения."""

    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Создайте app/static/index.html</h1>", status_code=200)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.post("/api/upload/video")
async def upload_video(file: UploadFile = File(...)) -> dict:
    """Сохраняет видео на диск и возвращает file_id."""

    ext = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    file_id = uuid.uuid4().hex
    video_path = UPLOAD_DIR / f"{file_id}{ext}"

    with video_path.open("wb") as f:
        f.write(await file.read())

    files_store[file_id] = {
        "video_path": str(video_path),
        "updated_at": now_utc().isoformat(),
    }
    return {"file_id": file_id, "filename": file.filename}


@app.post("/api/upload/subtitles")
async def upload_subtitles(file_id: str, file: UploadFile = File(...)) -> dict:
    """Принимает пользовательский .srt и связывает его с видео."""

    if file_id not in files_store:
        raise HTTPException(status_code=404, detail="Неизвестный file_id")

    ext = Path(file.filename or "subtitles.srt").suffix.lower()
    if ext != ".srt":
        raise HTTPException(status_code=400, detail="Поддерживается только .srt")

    target = srt_path(file_id)
    target.write_bytes(await file.read())
    files_store[file_id]["updated_at"] = now_utc().isoformat()
    return {"ok": True, "file_id": file_id}


@app.post("/api/generate")
async def generate_subtitles(payload: GenerateRequest, background_tasks: BackgroundTasks) -> dict:
    """Ставит задачу AI-транскрибации в фоновую очередь."""

    _ = safe_video_path(payload.file_id)

    task_id = uuid.uuid4().hex
    tasks_store[task_id] = TaskState(status="queued", progress=0)

    background_tasks.add_task(
        _run_whisper_task,
        task_id,
        payload.file_id,
        payload.model_size,
        payload.language,
    )
    return {"task_id": task_id, "status": "queued"}


@app.get("/api/status/{task_id}")
async def task_status(task_id: str) -> dict:
    """Возвращает статус фоновой задачи."""

    task = tasks_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return task.model_dump()


@app.get("/api/subtitles/{file_id}")
async def get_subtitles(file_id: str) -> dict:
    """Читает субтитры для отображения и редактирования на фронтенде."""

    path = safe_subtitle_path(file_id)
    subtitles = parse_srt_file(path)
    return {"file_id": file_id, "subtitles": [item.model_dump() for item in subtitles]}


@app.put("/api/subtitles/{file_id}")
async def update_subtitles(file_id: str, payload: SubtitlePayload) -> dict:
    """Сохраняет отредактированный список субтитров."""

    if file_id not in files_store:
        raise HTTPException(status_code=404, detail="Неизвестный file_id")

    save_subtitles(file_id, payload.subtitles)
    files_store[file_id]["updated_at"] = now_utc().isoformat()
    return {"ok": True, "entries": len(payload.subtitles)}


@app.post("/api/export")
async def export_video(payload: ExportRequest, background_tasks: BackgroundTasks) -> dict:
    """Запускает задачу вшивания субтитров и возвращает task_id."""

    _ = safe_video_path(payload.file_id)
    _ = safe_subtitle_path(payload.file_id)

    task_id = uuid.uuid4().hex
    tasks_store[task_id] = TaskState(status="queued", progress=0)
    background_tasks.add_task(_run_export_task, task_id, payload)
    return {"task_id": task_id, "status": "queued"}


@app.get("/outputs/{filename}")
async def download_output(filename: str) -> Response:
    """Отдает готовый результат экспорта для скачивания."""

    target = OUTPUT_DIR / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(target, filename=filename)


@app.get("/api/video/{file_id}")
async def stream_video(request: Request, file_id: str) -> Response:
    """Стримит видео с поддержкой HTTP Range для корректной работы <video>."""

    video_path = safe_video_path(file_id)
    file_size = video_path.stat().st_size
    range_header = request.headers.get("range")

    if not range_header:
        return FileResponse(video_path, media_type="video/mp4")

    match = re.match(r"bytes=(\d+)-(\d*)", range_header)
    if not match:
        raise HTTPException(status_code=416, detail="Некорректный Range")

    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else file_size - 1
    end = min(end, file_size - 1)
    length = end - start + 1

    def iterfile() -> bytes:
        with video_path.open("rb") as file_obj:
            file_obj.seek(start)
            remaining = length
            chunk_size = 1024 * 1024
            while remaining > 0:
                chunk = file_obj.read(min(chunk_size, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Content-Type": "video/mp4",
    }
    return StreamingResponse(iterfile(), status_code=206, headers=headers)


@app.get("/api/health")
async def health() -> dict:
    """Технический endpoint для проверки доступности сервиса."""

    return {"status": "ok", "files": len(files_store), "tasks": len(tasks_store)}
