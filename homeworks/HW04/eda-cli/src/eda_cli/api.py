from __future__ import annotations

from time import perf_counter

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .core import compute_quality_flags, missing_table, summarize_dataset

app = FastAPI(
    title="AIE Dataset Quality API",
    version="0.2.0",
    description=(
        "HTTP-сервис-заглушка для оценки готовности датасета к обучению модели. "
        "Использует простые эвристики качества данных вместо настоящей ML-модели."
    ),
    docs_url="/docs",
    redoc_url=None,
)



class QualityRequest(BaseModel):
    """Агрегированные признаки датасета – 'фичи' для заглушки модели."""

    n_rows: int = Field(..., ge=0, description="Число строк в датасете")
    n_cols: int = Field(..., ge=0, description="Число колонок")
    max_missing_share: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Максимальная доля пропусков среди всех колонок (0..1)",
    )
    numeric_cols: int = Field(
        ...,
        ge=0,
        description="Количество числовых колонок",
    )
    categorical_cols: int = Field(
        ...,
        ge=0,
        description="Количество категориальных колонок",
    )


class QualityResponse(BaseModel):
    """Ответ заглушки модели качества датасета."""

    ok_for_model: bool = Field(
        ...,
        description="True, если датасет считается достаточно качественным для обучения модели",
    )
    quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Интегральная оценка качества данных (0..1)",
    )
    message: str = Field(
        ...,
        description="Человекочитаемое пояснение решения",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Время обработки запроса на сервере, миллисекунды",
    )
    flags: dict[str, bool] | None = Field(
        default=None,
        description="Булевы флаги с подробностями (например, too_few_rows, too_many_missing)",
    )
    dataset_shape: dict[str, int] | None = Field(
        default=None,
        description="Размеры датасета: {'n_rows': ..., 'n_cols': ...}, если известны",
    )



@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Простейший health-check сервиса."""
    return {
        "status": "ok",
        "service": "dataset-quality",
        "version": "0.2.0",
    }



@app.post("/quality", response_model=QualityResponse, tags=["quality"])
def quality(req: QualityRequest) -> QualityResponse:
    """
    Эндпоинт-заглушка, который принимает агрегированные признаки датасета
    и возвращает эвристическую оценку качества.
    """

    start = perf_counter()

    score = 1.0

    score -= req.max_missing_share

    if req.n_rows < 1000:
        score -= 0.2

    if req.n_cols > 100:
        score -= 0.1

    if req.numeric_cols == 0 and req.categorical_cols > 0:
        score -= 0.1
    if req.categorical_cols == 0 and req.numeric_cols > 0:
        score -= 0.05

    score = max(0.0, min(1.0, score))

    ok_for_model = score >= 0.7
    if ok_for_model:
        message = "Данных достаточно, модель можно обучать (по текущим эвристикам)."
    else:
        message = "Качество данных недостаточно, требуется доработка (по текущим эвристикам)."

    latency_ms = (perf_counter() - start) * 1000.0

    flags = {
        "too_few_rows": req.n_rows < 1000,
        "too_many_columns": req.n_cols > 100,
        "too_many_missing": req.max_missing_share > 0.5,
        "no_numeric_columns": req.numeric_cols == 0,
        "no_categorical_columns": req.categorical_cols == 0,
    }

    print(
        f"[quality] n_rows={req.n_rows} n_cols={req.n_cols} "
        f"max_missing_share={req.max_missing_share:.3f} "
        f"score={score:.3f} latency_ms={latency_ms:.1f} ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols},
    )



@app.post(
    "/quality-from-csv",
    response_model=QualityResponse,
    tags=["quality"],
    summary="Оценка качества по CSV-файлу с использованием EDA-ядра",
)
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    """
    Эндпоинт, который принимает CSV-файл, запускает EDA-ядро
    (summarize_dataset + missing_table + compute_quality_flags)
    и возвращает оценку качества данных.
    """

    start = perf_counter()

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:

        df = pd.read_csv(file.file)
    except Exception as exc: 
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame).")

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(summary, missing_df)

    score = float(flags_all.get("quality_score", 0.0))
    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7

    if ok_for_model:
        message = "CSV выглядит достаточно качественным для обучения модели (по текущим эвристикам)."
    else:
        message = "CSV требует доработки перед обучением модели (по текущим эвристикам)."

    latency_ms = (perf_counter() - start) * 1000.0

    flags_bool: dict[str, bool] = {
        key: bool(value)
        for key, value in flags_all.items()
        if isinstance(value, bool)
    }

    n_rows = summary.n_rows
    n_cols = summary.n_cols

    print(
        f"[quality-from-csv] filename={file.filename!r} "
        f"n_rows={n_rows} n_cols={n_cols} score={score:.3f} "
        f"latency_ms={latency_ms:.1f} ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags_bool,
        dataset_shape={"n_rows": n_rows, "n_cols": n_cols},
    )



@app.post(
    "/head",
    tags=["preview"],
    summary="Вывод первых n строк CSV-файла (использует ядро HW03)",
)
async def head(
    file: UploadFile = File(...),
    n: int = 5,
) -> dict:
    """
    Эндпоинт, который принимает CSV-файл и возвращает первые n строк датасета.
    Для получения метаданных (размеры, колонки) используется summarize_dataset из core.py.
    """

    start = perf_counter()

    if file.content_type not in (
        "text/csv",
        "application/vnd.ms-excel",
        "application/octet-stream",
    ):
        raise HTTPException(
            status_code=400,
            detail="Ожидается CSV-файл (content-type text/csv).",
        )

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV-файл пустой.")

    head_df = df.head(n)

    summary = summarize_dataset(df)
    
    latency_ms = (perf_counter() - start) * 1000.0

    print(
        f"[head] filename={file.filename!r} "
        f"requested_n={n} returned_n={head_df.shape[0]} "
        f"n_rows_total={summary.n_rows} n_cols={summary.n_cols} "
        f"latency_ms={latency_ms:.1f} ms"
    )

    return {
        "rows": head_df.to_dict(orient="records"),
        "latency_ms": latency_ms,
        "dataset_shape": {"n_rows": summary.n_rows, "n_cols": summary.n_cols},
        "columns": [col.name for col in summary.columns]
    }


@app.post(
    "/sample",
    tags=["preview"],
    summary="Вывод некоторых n строк CSV-файла (использует ядро HW03)",
)
async def sample(
    file: UploadFile = File(...),
    n: int = 5,
) -> dict:
    """
    Эндпоинт, который принимает CSV-файл и возвращает случайные n строк.
    Для получения метаданных используется summarize_dataset из core.py.
    """

    start = perf_counter()

    if file.content_type not in (
        "text/csv",
        "application/vnd.ms-excel",
        "application/octet-stream",
    ):
        raise HTTPException(
            status_code=400,
            detail="Ожидается CSV-файл (content-type text/csv).",
        )

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV-файл пустой.")

    sample_df = df.sample(n=min(n, len(df)))

    summary = summarize_dataset(df)

    latency_ms = (perf_counter() - start) * 1000.0

    print(
        f"[sample] filename={file.filename!r} "
        f"requested_n={n} returned_n={sample_df.shape[0]} "
        f"n_rows_total={summary.n_rows} n_cols={summary.n_cols} "
        f"latency_ms={latency_ms:.1f} ms"
    )

    return {
        "rows": sample_df.to_dict(orient="records"),
        "latency_ms": latency_ms,
        "dataset_shape": {"n_rows": summary.n_rows, "n_cols": summary.n_cols},
        "columns": [col.name for col in summary.columns]
    }


# uv run uvicorn eda_cli.api:app --reload --port 8000