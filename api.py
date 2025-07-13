import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi.responses import JSONResponse, StreamingResponse
from urllib.parse import urlparse, parse_qs
from main import main

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
assert YOUTUBE_API_KEY, "Missing YOUTUBE_API_KEY in .env"

VIDEO_DIR = "./"  # ✅ CHANGE this to your video directory path

app = FastAPI()

YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/search"

class SearchRequest(BaseModel):
    q: str = Field(..., description="Search query string")
    max_results: int = Field(10, ge=1, le=50, description="Number of results (1-50)")

class YoutubeURL(BaseModel):
    link: str = Field(..., description="Youtube URL string")

@app.post("/youtube_search")
def youtube_search(body: SearchRequest):
    params = {
        "key": YOUTUBE_API_KEY,
        "part": "snippet",
        "q": body.q,
        "type": "video",
        "maxResults": body.max_results
    }
    r = requests.get(YOUTUBE_API_URL, params=params)
    if r.status_code != 200:
        return JSONResponse(content={"error": r.text}, status_code=r.status_code)
    return r.json()

@app.post("/youtube_url")
def youtube_url(body: YoutubeURL):
    # Extract video ID from URL
    video_id = main(
    youtube_url=body.link,
    keep_final_output=True,
    fade_duration=1.0,
    padding=12.0,
    fps=60,
    yt_format='bestvideo[height<=720]+bestaudio/best[height<=720]'
    )

    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    # Construct expected video file path
    video_path = os.path.join(VIDEO_DIR, video_id)

    if not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    # Stream and delete logic
    def stream_and_delete():
        try:
            with open(video_path, "rb") as f:
                yield from f
        finally:
            try:
                os.remove(video_path)
                print(f"Deleted file: {video_path}")
            except Exception as e:
                print(f"Error deleting file: {e}")

    return StreamingResponse(stream_and_delete(), media_type="video/mp4")

