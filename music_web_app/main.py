import os
import json
import shutil
import librosa
import numpy as np
import psycopg2
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")
os.makedirs("temp_uploads", exist_ok=True)

# Tham số z-score (mean/std) đã được tính trên toàn bộ DB ở bước build_vector_db.
# Phải dùng đúng bộ tham số này để feature query nằm cùng không gian với vector trong DB.
with open("../Results/feature_stats.json") as f:
    FEATURE_STATS = json.load(f)

MFCC_COLS = [f"mfcc_{i}" for i in range(1, 14)]
CHROMA_LABELS = ["C", "Csharp", "D", "Dsharp", "E", "F",
                 "Fsharp", "G", "Gsharp", "A", "Asharp", "B"]
CHROMA_COLS = [f"chroma_{label}" for label in CHROMA_LABELS]


def _z(value, col):
    s = FEATURE_STATS[col]
    return (float(value) - s["mean"]) / s["std"]


def extract_layered_features(y, sr):
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = tempo[0] if isinstance(tempo, np.ndarray) else tempo

    layer1 = [
        _z(zcr, "zero_crossing_rate"),
        _z(tempo_val, "tempo"),
    ]
    layer2 = [
        _z(centroid, "spectral_centroid"),
        _z(bandwidth, "spectral_bandwidth"),
        _z(rolloff, "spectral_rolloff"),
    ]
    layer3 = (
        [_z(v, c) for v, c in zip(mfcc, MFCC_COLS)] +
        [_z(v, c) for v, c in zip(chroma, CHROMA_COLS)]
    )
    return layer1, layer2, layer3


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/audio")
async def get_audio(path: str):
    if os.path.exists(path):
        return FileResponse(path)
    return {"error": "File không tồn tại"}

@app.post("/api/search")
async def search_audio(file: UploadFile = File(...)):
    temp_path = f"temp_uploads/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    y, sr = librosa.load(temp_path, sr=22050)
    window_samples = int(5.0 * sr)
    hop_samples = int(2.5 * sr)

    conn = psycopg2.connect(dbname="music_retrieval", user="postgres", password="admin", host="localhost", port="5433")
    cursor = conn.cursor()

    file_stats = {}
    start_sample = 0
    while start_sample + window_samples <= len(y):
        y_segment = y[start_sample:start_sample + window_samples]
        l1, l2, l3 = extract_layered_features(y_segment, sr)

        # Sau z-score normalization: L1, L2 dùng Euclidean (<->) để giữ thông tin biên độ
        # (vd: bản sôi động vs tĩnh lặng có |zcr|, |centroid| khác nhau thật sự).
        # L3 vẫn dùng cosine (<=>) vì là vector "fingerprint" 25 chiều, hướng vector mang thông tin chính.
        query = """
            WITH L1_Filter AS (
                SELECT file_name, path, layer2_perceptual, layer3_identity,
                       (layer1_physical <-> %s::vector) AS dist1
                FROM music_segments
                ORDER BY dist1 ASC LIMIT 1000
            ),
            L2_Filter AS (
                SELECT file_name, path, layer3_identity, dist1,
                       (layer2_perceptual <-> %s::vector) AS dist2
                FROM L1_Filter
                ORDER BY dist2 ASC LIMIT 100
            )
            SELECT file_name, path, dist1, dist2,
                   (layer3_identity <=> %s::vector) AS dist3
            FROM L2_Filter
            ORDER BY dist3 ASC LIMIT 10;
        """
        cursor.execute(query, (l1, l2, l3))
        results = cursor.fetchall()

        for row in results:
            fname, fpath, d1, d2, d3 = row
            if fname not in file_stats:
                file_stats[fname] = {"votes": 0, "path": fpath, "l1": [], "l2": [], "l3": []}

            file_stats[fname]["votes"] += 1
            # Euclidean distance -> similarity %: 100 / (1 + d). d=0 → 100%, d=1 → 50%, d=2 → 33%.
            file_stats[fname]["l1"].append(100.0 / (1.0 + float(d1)))
            file_stats[fname]["l2"].append(100.0 / (1.0 + float(d2)))
            # Cosine distance -> similarity %: (1 - d) * 100, clamp tại 0.
            file_stats[fname]["l3"].append(max(0.0, (1.0 - float(d3)) * 100.0))

        start_sample += hop_samples
    cursor.close()
    conn.close()

    # Tính similarity_score cho từng file rồi xếp hạng THEO similarity (không xếp theo votes)
    # — tránh việc file dài trong DB có nhiều segment lọt top-10 → chiếm thứ hạng nhờ số phiếu.
    scored = []
    for fname, data in file_stats.items():
        l1_sim = float(np.mean(data["l1"]))
        l2_sim = float(np.mean(data["l2"]))
        l3_sim = float(np.mean(data["l3"]))
        overall_sim = (l1_sim + l2_sim + l3_sim) / 3
        scored.append({
            "file_name": fname,
            "path": data["path"],
            "votes": data["votes"],
            "l1_sim": round(l1_sim, 2),
            "l2_sim": round(l2_sim, 2),
            "l3_sim": round(l3_sim, 2),
            "similarity_score": round(overall_sim, 2),
        })

    scored.sort(key=lambda x: x["similarity_score"], reverse=True)
    final_results = []
    for rank, item in enumerate(scored[:5], 1):
        item["rank"] = rank
        final_results.append(item)

    return {"input_audio_path": temp_path, "results": final_results}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
