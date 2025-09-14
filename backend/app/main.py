from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os

# ML imports (existing)
from ml.ensemble import ensemble_predict
from ml.video_features import extract_pose_features

app = FastAPI()

# ✅ Allow mobile app requests (important!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production, restrict to your mobile device IP/domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Existing video upload route
# -------------------------
@app.post('/upload_video/')
async def upload_video(file: UploadFile = File(...)):
    path = f'/tmp/{file.filename}'
    with open(path, 'wb') as f:
        f.write(await file.read())
    feats = extract_pose_features(path)
    seq = feats.fillna(0).values
    flat = feats.iloc[0].to_dict() if not feats.empty else {'stride_frequency':0,'hip_variance':0}
    pred = ensemble_predict(seq, list(flat.values()))
    return JSONResponse({'predicted_time': float(pred)})

# -------------------------
# New: Injury Risk Prediction
# -------------------------
@app.post('/predict_injury')
async def predict_injury(payload: dict):
    # Mock logic for now
    load = payload.get("load", 0)
    fatigue = payload.get("fatigue", 0.0)
    hr = payload.get("heart_rate", 0)

    # Simple toy risk calculation
    risk = "Low"
    if load > 150 or fatigue > 0.7 or hr > 180:
        risk = "High"
    elif load > 100 or fatigue > 0.5:
        risk = "Medium"

    return {"prediction": risk}

# -------------------------
# New: Training Plan
# -------------------------
@app.post('/plan_training')
async def plan_training(payload: dict):
    athlete_id = payload.get("athlete_id", 0)
    # Mock plan for now
    plan = [
        {"day": "Monday", "activity": "Sprint intervals (6x100m)"},
        {"day": "Wednesday", "activity": "Strength training (legs/core)"},
        {"day": "Friday", "activity": "Tempo runs (4x400m)"},
    ]
    return {"athlete_id": athlete_id, "plan": plan}

# -------------------------
# Health Check
# -------------------------
@app.get('/health')
def health():
    return {'status':'ok'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app.main:app', host='0.0.0.0', port=8000, reload=True)
