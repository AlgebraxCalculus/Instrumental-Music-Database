import os
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

def extract_layered_features(y, sr):
    # Trích xuất
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = tempo[0] if isinstance(tempo, np.ndarray) else tempo

    # Gom theo Layer
    layer1 = [float(zcr), float(tempo_val)]
    layer2 = [float(centroid), float(bandwidth), float(rolloff)]
    layer3 = np.hstack([mfcc, chroma]).tolist()
    
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
        
        # Sử dụng toán tử <=> (Cosine Distance) cho từng layer
        query = """
            WITH L1_Filter AS (
                SELECT file_name, path, layer2_perceptual, layer3_identity,
                       (layer1_physical <=> %s::vector) AS dist1
                FROM music_segments
                ORDER BY dist1 ASC LIMIT 1000
            ),
            L2_Filter AS (
                SELECT file_name, path, layer3_identity, dist1,
                       (layer2_perceptual <=> %s::vector) AS dist2
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
            # Chuyển đổi Cosine Distance sang Similarity %: (1 - distance) * 100
            file_stats[fname]["l1"].append(max(0, (1 - d1) * 100))
            file_stats[fname]["l2"].append(max(0, (1 - d2) * 100))
            file_stats[fname]["l3"].append(max(0, (1 - d3) * 100))
            
        start_sample += hop_samples
    cursor.close()
    conn.close()
    
    # Sắp xếp và tính toán kết quả cuối cùng
    sorted_files = sorted(file_stats.items(), key=lambda x: x[1]["votes"], reverse=True)[:5]
    final_results = []
    for rank, (fname, data) in enumerate(sorted_files, 1):
        l1_sim = np.mean(data["l1"])
        l2_sim = np.mean(data["l2"])
        l3_sim = np.mean(data["l3"])
        # Tính Similarity Score tổng hợp
        overall_sim = (l1_sim + l2_sim + l3_sim) / 3
        
        final_results.append({
            "rank": rank,
            "file_name": fname,
            "path": data["path"],
            "similarity_score": round(overall_sim, 2),
            "l1_sim": round(l1_sim, 2),
            "l2_sim": round(l2_sim, 2),
            "l3_sim": round(l3_sim, 2)
        })
        
    return {"input_audio_path": temp_path, "results": final_results}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)