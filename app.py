# from fastapi import FastAPI, Request, Form
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
# import joblib
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# app = FastAPI()
# templates = Jinja2Templates(directory="templates")


# try:
#     df_scaled = joblib.load('df_scaled.pkl')
#     metadata = joblib.load('metadata.pkl')
# except:
#     print("Error: Model files not found. Please run the notebook first.")

# @app.get("/")
# def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/recommend")
# def get_recommendations(request: Request, song_name: str = Form(...)):
#     # البحث عن الأغنية
#     song_search = metadata[metadata['track_name'].str.lower() == song_name.lower()]
    
#     if song_search.empty:
#         return templates.TemplateResponse("index.html", {
#             "request": request, 
#             "error": f"أوبس! ملقتش أغنية بالاسم ده: {song_name}"
#         })

#     idx = song_search.index[0]
#     song_vector = df_scaled[idx].reshape(1, -1)
    
#     # حساب التشابه لحظياً
#     scores = cosine_similarity(song_vector, df_scaled)[0]
#     similar_indices = scores.argsort()[-7:-1][::-1] # جلب أفضل 6 نتائج
    
#     results = metadata.iloc[similar_indices].to_dict('records')
    
#     return templates.TemplateResponse("index.html", {
#         "request": request, 
#         "recs": results, 
#         "target_song": song_search.iloc[0]['track_name']
#     })

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# تحميل الموديلات
df_scaled = joblib.load('df_scaled.pkl')
metadata = joblib.load('metadata.pkl')

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# محرك الاقتراحات أثناء الكتابة (Auto-complete)
@app.get("/search_suggestions")
def search_suggestions(q: str):
    if not q: return []
    # البحث في الأسماء أو الأنواع
    mask = (metadata['track_name'].str.contains(q, case=False)) | \
           (metadata['track_genre'].str.contains(q, case=False))
    suggestions = metadata[mask].head(8)
    
    results = []
    for _, row in suggestions.iterrows():
        results.append({
            "label": f"{row['track_name']} ({row['track_genre']})",
            "value": row['track_name']
        })
    return results

@app.post("/recommend")
def recommend(request: Request, song_name: str = Form(...)):
    search_result = metadata[metadata['track_name'].str.lower() == song_name.lower()]
    
    if search_result.empty:
        return templates.TemplateResponse("index.html", {"request": request, "error": "ملقتش الأغنية دي يا بطل!"})

    idx = search_result.index[0]
    song_vector = df_scaled[idx].reshape(1, -1)
    
    # حساب التشابه لحظياً (Memory Safe)
    scores = cosine_similarity(song_vector, df_scaled)[0]
    similar_indices = scores.argsort()[-7:-1][::-1]
    
    recs = metadata.iloc[similar_indices].to_dict('records')
    return templates.TemplateResponse("index.html", {
        "request": request, "recs": recs, "target": search_result.iloc[0]['track_name']
    })