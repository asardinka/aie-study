# Сегментация и классификация опухолей мозга по МРТ

По одному срезу: **классификация** типа опухоли и **сегментация** (плоскости **ax** / **sa** / **co**). Датасет обучения: [BRISC 2025](https://www.kaggle.com/datasets/briscdataset/brisc2025).

## Что нужно на машине

**Python 3.11+**. Остальное подтянется из `pyproject.toml` командами ниже (отдельно ставить PyTorch, FastAPI и т.д. не нужно).

**Рабочий каталог для всех команд** — папка `project/`: в ней лежат `pyproject.toml`, `src/`, `artifacts/` и т.д. Если у вас монорепозиторий или архив курса, сначала перейдите в этот каталог (`cd project` или аналог).

Дальше — один из сценариев.

---

## 1. Обучить свою модель

1. При необходимости замените или расширьте архитектуры в `src/models/` и согласуйте вызовы в коде обучения (`src/training/`).
2. Пути к данным, размеры изображений, эпохи и прочее — в `src/config.py`.
3. Подготовка данных: `data_load.py` в корне `project/` или `notebooks/data_load.ipynb`.

Создание окружения и установка зависимостей **с extras для обучения**:

```bash
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[train]"
```

**Linux / macOS:**

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[train]"
```

Запуск обучения (из корня `project/`, с активированным venv):

```bash
python -m src.training.train_models
```

То же через entrypoint: `aip-train`.

В `train_models.py` блок классификации можно отключить — тогда должны уже существовать файлы `classification_*_model.pt` в каталоге вывода (см. `src/config.py`).

---

## 2. Только запуск сервиса на Linux (сервер)

Команды ниже — из корня `project/`. Нужны готовые веса в `artifacts/` (см. раздел «Веса моделей»). Достаточно базовых зависимостей **без** `[train]`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m uvicorn src.tumor_app.api:app --host 0.0.0.0 --port 8000
```

- Веб-форма: `http://<IP_сервера>:8000/`
- OpenAPI: `http://<IP_сервера>:8000/docs`

Другой каталог с чекпоинтами — переменная окружения `ARTIFACTS_DIR` (абсолютный путь).

**Альтернатива без venv:** из корня `project/` выполните `docker compose up --build` (веса монтируются в `./artifacts`, в контейнере задан `ARTIFACTS_DIR=/app/artifacts`).

На минимальных образах Linux, если при импорте PyTorch ругается на OpenMP, может понадобиться пакет вроде `libgomp1` (как в `Dockerfile`).

---

## 3. Локальная проверка на Windows

Сценарий для ручной проверки UI и API на своей машине; автоматическая проверка — см. раздел «Прочее» (`pytest`).

**PowerShell** (из корня `project/`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m uvicorn src.tumor_app.api:app --host 127.0.0.1 --port 8000
```

Браузер: [http://127.0.0.1:8000/](http://127.0.0.1:8000/) — форма; [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) — Swagger.

Если `Activate.ps1` блокируется политикой выполнения: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, либо **cmd** с `.venv\Scripts\activate.bat`, либо без активации: `.\.venv\Scripts\python.exe -m pip install -e .` и тот же интерпретатор для `uvicorn`.

Для разработки к API можно добавить флаг `--reload` к команде `uvicorn`.

---

## Веса моделей

В каталоге `artifacts/` должны лежать шесть чекпоинтов:

`classification_ax_model.pt`, `classification_sa_model.pt`, `classification_co_model.pt`,  
`segmentation_ax_model.pt`, `segmentation_sa_model.pt`, `segmentation_co_model.pt`.

Путь по умолчанию задаётся относительно расположения `src/config.py`, не от текущей папки в терминале. Другой каталог — `ARTIFACTS_DIR`.

Старые архитектуры для сравнения — `artifacts/v1/` и `src/models/legacy_aip_v1.py`.

---

## Прочее

- **CLI:** `python -m src.tumor_app.cli predict путь\к.jpg --plane ax --out-dir out`
- **Проверка API (pytest):** из корня `project/` с активированным venv: `pip install -e ".[dev]"`, затем `pytest` (нужны шесть `.pt` в `artifacts/`, иначе тест инференса пропускается).
- Зависимости и extras — в `pyproject.toml` (`train` — обучение, `dev` — тесты).
