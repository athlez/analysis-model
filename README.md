# cricket-ai-mvp

MVP for a cricket AI system.

## Structure

```
cricket-ai-mvp/
├── data/       # raw & processed datasets (gitignored contents)
├── pipeline/   # data ingest / transform / orchestration
├── models/     # ML model definitions, training, inference
├── physics/    # ball-flight / trajectory physics models
├── api/        # FastAPI (or similar) service layer
├── utils/      # shared helpers, config, logging
├── output/     # generated artifacts, reports (gitignored contents)
└── main.py     # entry point
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```
