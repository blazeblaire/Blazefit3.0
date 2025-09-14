        # SACA — Sprint AI Coach Assistant (scaffold)

## This archive contains a minimal scaffold for the SACA project described in your concept note.

## Quick start (local, non-docker)

1. Backend

```bash
cd backend
pip install -r requirements.txt
# prepare a CSV with columns: athlete_id, session_date, time, distance, ...
python app/ml/preprocess.py --input data/raw.csv --out data/processed.parquet
python app/ml/train_svr.py --data data/processed.parquet
python app/ml/train_rnn.py --data data/processed.parquet --epochs 5
uvicorn app.main:app --reload
```

2. Mobile (Expo)

```bash
cd mobile
npm install
npx expo start
```

## Notes

- Mediapipe may not install in all environments; see its docs for platform specifics.
- This is a starting scaffold; add authentication, better error handling, tests and CI before production.
