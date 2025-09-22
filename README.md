## Overview
This repo contains the implementation for the assignment: Reply Classification Pipeline.

- **Part A**: ML pipeline in `train_model.py` (trains baseline and DistilBERT, evaluates, saves model).
- **Part B**: FastAPI deployment in `app.py`.
- **Part C**: Answers in `answers.md`.
- Dataset: `reply_classification_dataset.csv` (sample; replace with real if needed).

## Setup Instructions
1. Clone this repo: `git clone <repo-url>`
2. Create a virtual env: `conda create -n RCM python=3.12 -y` and activate it.
3. Install dependencies: `pip install -r requirements.txt`
4. Run training: `python train_model.py` (saves model to `saved_model/`).
5. Run API: `uvicorn main:app --reload`
   - Test endpoint: POST to `http://127.0.0.1:8000/predict` with JSON like `{"text": "Looking forward to the demo!"}`

## Requirements
See `requirements.txt`.

