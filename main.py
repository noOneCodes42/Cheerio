import os
import requests
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import asyncio
import json
import logging
import uuid
import cv2
import numpy as np
from typing import Optional, Dict, Any
import threading
import time
from queue import Queue
import base64
import secrets
from pathlib import Path

# Import your modified main function
from streaming_main import StreamingHighlightProcessor

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
assert YOUTUBE_API_KEY, "Missing YOUTUBE_API_KEY in .env"

app = FastAPI()

YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/search"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create tmp directory for storing files
TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)

# File cleanup configuration
FILE_EXPIRY_MINUTES = 15
CLEANUP_INTERVAL_MINUTES = 5

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.processing_sessions: Dict[str, bool] = {}
        self.processors: Dict[str, StreamingHighlightProcessor] = {}
        self.video_streamers: Dict[str, WebSocketVideoStreamer] = {}
        self.download_tokens: Dict[str, Dict[str, Any]] = {}  # token -> file info
        self.session_files: Dict[str, str] = {}  # session_id -> file_path
        self._main_loop = None  # Store the main event loop for thread-safe operations
    
    def set_main_loop(self, loop):
        """Set the main event loop for thread-safe operations"""
        self._main_loop = loop
    
    def generate_session_id(self) -> str:
        """Generate a unique session ID"""
        return str(uuid.uuid4())
    
    def generate_download_token(self, session_id: str, file_path: str) -> str:
        """Generate a secure download token for a file"""
        token = secrets.token_urlsafe(32)
        self.download_tokens[token] = {
            "session_id": session_id,
            "file_path": file_path,
            "created_at": time.time()
        }
        self.session_files[session_id] = file_path
        
        # Schedule automatic cleanup after FILE_EXPIRY_MINUTES
        asyncio.create_task(self._schedule_file_cleanup(token, FILE_EXPIRY_MINUTES * 60))
        
        return token
    
    def get_file_by_token(self, token: str) -> Optional[str]:
        """Get file path by download token"""
        token_info = self.download_tokens.get(token)
        if not token_info:
            return None
            
        file_path = token_info["file_path"]
        if not os.path.exists(file_path):
            # Clean up invalid token
            del self.download_tokens[token]
            return None
            
        return file_path
    
    def cleanup_download_token(self, token: str):
        """Clean up download token and associated file"""
        token_info = self.download_tokens.get(token)
        if token_info:
            file_path = token_info["file_path"]
            session_id = token_info["session_id"]
            
            # Delete the file
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"🗑️ Deleted file: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {e}")
            
            # Clean up tracking
            del self.download_tokens[token]
            if session_id in self.session_files:
                del self.session_files[session_id]
    
    async def _schedule_file_cleanup(self, token: str, delay_seconds: int):
        """Schedule automatic file cleanup after delay"""
        try:
            await asyncio.sleep(delay_seconds)
            
            # Check if token still exists (might have been downloaded and cleaned up)
            if token in self.download_tokens:
                token_info = self.download_tokens[token]
                file_path = token_info["file_path"]
                
                logger.info(f"⏰ Auto-cleaning up expired file: {file_path}")
                self.cleanup_download_token(token)
                
        except Exception as e:
            logger.error(f"Error in scheduled cleanup for token {token}: {e}")
    
    async def cleanup_expired_files(self):
        """Cleanup all expired files (called periodically)"""
        try:
            current_time = time.time()
            expired_tokens = []
            
            for token, token_info in self.download_tokens.items():
                file_age = current_time - token_info["created_at"]
                if file_age > (FILE_EXPIRY_MINUTES * 60):
                    expired_tokens.append(token)
            
            for token in expired_tokens:
                token_info = self.download_tokens[token]
                file_path = token_info["file_path"]
                logger.info(f"🧹 Cleaning up expired file: {file_path} (age: {file_age/60:.1f} minutes)")
                self.cleanup_download_token(token)
                
            if expired_tokens:
                logger.info(f"🧹 Cleaned up {len(expired_tokens)} expired files")
                
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")
    
    async def connect_websocket(self, websocket: WebSocket) -> str:
        """Connect WebSocket and return generated session ID"""
        session_id = self.generate_session_id()
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.processing_sessions[session_id] = False
        
        # Send session ID to client
        await websocket.send_text(json.dumps({
            "type": "session_created",
            "session_id": session_id
        }))
        
        logger.info(f"WebSocket connected with new session: {session_id}")
        return session_id
    
    def disconnect(self, session_id: str):
        """Disconnect session and cleanup - but preserve download files"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.processing_sessions:
            del self.processing_sessions[session_id]
        if session_id in self.processors:
            # Set stop flag on processor
            processor = self.processors[session_id]
            processor.should_stop = True
            del self.processors[session_id]
        if session_id in self.video_streamers:
            # Stop video streaming
            self.video_streamers[session_id].stop_streaming()
            del self.video_streamers[session_id]
        
        # DON'T clean up session files - let them expire naturally or be downloaded
        # The automatic cleanup system will handle them after 15 minutes
        if session_id in self.session_files:
            file_path = self.session_files[session_id]
            logger.info(f"📁 Preserving download file for session {session_id}: {file_path}")
            # Remove from session tracking but don't delete the file
            del self.session_files[session_id]
        
        logger.info(f"Session {session_id} disconnected (download files preserved)")
    
    async def send_progress(self, session_id: str, progress_data: dict):
        """Send progress update via WebSocket"""
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(json.dumps(progress_data))
            except Exception as e:
                logger.error(f"Error sending progress to {session_id}: {e}")
                self.disconnect(session_id)
    
    async def send_video_frame(self, session_id: str, frame_data: str):
        """Send video frame directly via WebSocket"""
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(json.dumps({
                    "type": "video_frame",
                    "data": frame_data
                }))
            except Exception as e:
                logger.error(f"Error sending video frame to {session_id}: {e}")
                self.disconnect(session_id)

manager = ConnectionManager()

# Background task for periodic cleanup
async def periodic_cleanup():
    """Background task that runs periodic cleanup every CLEANUP_INTERVAL_MINUTES"""
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL_MINUTES * 60)  # Convert to seconds
            await manager.cleanup_expired_files()
        except Exception as e:
            logger.error(f"Error in periodic cleanup task: {e}")

# Start the background cleanup task
@app.on_event("startup")
async def startup_event():
    """Start background tasks when the app starts"""
    # Store the main event loop for thread-safe operations
    loop = asyncio.get_running_loop()
    manager.set_main_loop(loop)
    
    asyncio.create_task(periodic_cleanup())
    logger.info(f"🚀 Started periodic cleanup task (every {CLEANUP_INTERVAL_MINUTES} minutes)")
    logger.info(f"📁 Files will expire after {FILE_EXPIRY_MINUTES} minutes if not downloaded")

class SearchRequest(BaseModel):
    q: str = Field(..., description="Search query string")
    max_results: int = Field(10, ge=1, le=50, description="Number of results (1-50)")

class YoutubeStreamRequest(BaseModel):
    link: str = Field(..., description="Youtube URL string")
    session_id: str = Field(..., description="Session ID from WebSocket connection")
    fade_duration: float = Field(1.0, description="Fade duration in seconds")
    padding: float = Field(5.0, description="Padding in seconds")
    fps: int = Field(25, description="Frames per second")
    yt_format: str = Field('bestvideo[vcodec!*=av01][height<=720]+bestaudio/best[height<=720]', description="YouTube format (AV1-safe)")

# Progress tracking class
class ProgressTracker:
    def __init__(self, session_id: str, connection_manager: ConnectionManager):
        self.session_id = session_id
        self.manager = connection_manager
        self.current_step = 0
        self.total_steps = 6
        
    async def send_update(self, message: str, progress: float = None, data: Dict[Any, Any] = None):
        update = {
            "type": "progress",
            "message": message,
            "step": self.current_step,
            "total_steps": self.total_steps,
            "progress": progress or (self.current_step / self.total_steps),
            "data": data or {}
        }
        await self.manager.send_progress(self.session_id, update)
    
    async def next_step(self, message: str, data: Dict[Any, Any] = None):
        self.current_step += 1
        await self.send_update(message, data=data)
    
    async def send_download_link(self, file_path: str, filename: str):
        """Send download link for the final video file"""
        try:
            # Generate secure download token
            token = self.manager.generate_download_token(self.session_id, file_path)
            download_url = f"/download/{token}"
            
            # Send download link to client
            await self.send_update(
                f"✅ Video processing complete! Download ready.",
                progress=1.0,
                data={
                    "completed": True,
                    "download_url": download_url,
                    "filename": filename,
                    "message": "Click the download link to get your video file"
                }
            )
            
            logger.info(f"📎 Download link sent for session {self.session_id}: {download_url}")
            
        except Exception as e:
            logger.error(f"Error sending download link: {e}")
            await self.send_update(
                "Error preparing download link",
                data={"error": True, "message": str(e)}
            )
    
    # Add sync methods for use from synchronous code
    def sync_update(self, message: str, progress: float = None, data: Dict[Any, Any] = None):
        """Send progress update synchronously"""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.send_update(message, progress, data))
            loop.close()
        except Exception as e:
            logger.error(f"Progress update error: {e}")
    
    def sync_next_step(self, message: str, data: Dict[Any, Any] = None):
        """Send next step update synchronously"""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.next_step(message, data))
            loop.close()
        except Exception as e:
            logger.error(f"Progress step error: {e}")

# WebSocket-based video streaming class
class WebSocketVideoStreamer:
    def __init__(self, session_id: str, connection_manager: ConnectionManager):
        self.session_id = session_id
        self.manager = connection_manager
        self.current_video_path = None
        self.streaming = False
        self.stream_thread = None
        
    def set_video_path(self, video_path: str):
        """Set the current video file to stream"""
        self.current_video_path = video_path
        logger.info(f"Video path set for session {self.session_id}: {video_path}")
        
    def start_streaming(self):
        """Start streaming video frames via WebSocket"""
        if not self.streaming and self.current_video_path:
            self.streaming = True
            self.stream_thread = threading.Thread(target=self._stream_video)
            self.stream_thread.daemon = True
            self.stream_thread.start()
            logger.info(f"Started WebSocket video streaming for session {self.session_id}")
    
    def stop_streaming(self):
        """Stop streaming video frames"""
        self.streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2)
        logger.info(f"Stopped WebSocket video streaming for session {self.session_id}")
    
    def _stream_video(self):
        """Stream video frames via WebSocket"""
        if not self.current_video_path or not os.path.exists(self.current_video_path):
            logger.error(f"Video file not found: {self.current_video_path}")
            return
            
        cap = cv2.VideoCapture(self.current_video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {self.current_video_path}")
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_time = 1.0 / fps
        
        try:
            while self.streaming:
                ret, frame = cap.read()
                if not ret:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Resize frame for streaming
                height, width = frame.shape[:2]
                if width > 640:
                    scale = 640 / width
                    new_width = 640
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Encode frame as JPEG and base64
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_data = base64.b64encode(buffer).decode('utf-8')
                
                # Send frame via WebSocket
                asyncio.create_task(
                    self.manager.send_video_frame(self.session_id, frame_data)
                )
                
                # Control frame rate
                time.sleep(frame_time)
                
        except Exception as e:
            logger.error(f"Error in WebSocket video streaming: {e}")
        finally:
            cap.release()

@app.post("/youtube_search")
def youtube_search(body: SearchRequest):
    params = {
        "key": YOUTUBE_API_KEY,
        "part": "snippet",
        "q": body.q,
        "type": "video",
        "videoCategoryId": "17",
        "videoDuration": "long",
        "maxResults": body.max_results
    }
    r = requests.get(YOUTUBE_API_URL, params=params)
    if r.status_code != 200:
        return JSONResponse(content={"error": r.text}, status_code=r.status_code)
    return r.json()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint that auto-generates session ID"""
    session_id = await manager.connect_websocket(websocket)
    
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                        
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_text(json.dumps({"type": "ping"}))
                continue
            except WebSocketDisconnect:
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        manager.disconnect(session_id)

@app.post("/youtube_stream")
async def youtube_stream(body: YoutubeStreamRequest):
    session_id = body.session_id
    
    # Check if session exists
    if session_id not in manager.active_connections:
        raise HTTPException(status_code=400, detail="Session not found. Connect via WebSocket first.")
    
    # Check if already processing
    if manager.processing_sessions.get(session_id, False):
        raise HTTPException(status_code=409, detail="Session is already processing a video")
    
    # Mark session as processing
    manager.processing_sessions[session_id] = True
    
    try:
        # Start processing in background
        asyncio.create_task(process_video_async(
            youtube_url=body.link,
            session_id=session_id,
            fade_duration=body.fade_duration,
            padding=body.padding,
            fps=body.fps,
            yt_format=body.yt_format
        ))
        
        return JSONResponse(content={
            "message": "Video processing started", 
            "session_id": session_id
        })
        
    except Exception as e:
        manager.processing_sessions[session_id] = False
        raise HTTPException(status_code=500, detail=f"Error starting video processing: {str(e)}")

async def process_video_async(
    youtube_url: str,
    session_id: str,
    fade_duration: float,
    padding: float,
    fps: int,
    yt_format: str
):
    """Async wrapper for video processing with file-based delivery"""
    try:
        # Create progress tracker
        progress_tracker = ProgressTracker(session_id, manager)
        
        # Create WebSocket video streamer
        video_streamer = WebSocketVideoStreamer(session_id, manager)
        manager.video_streamers[session_id] = video_streamer
        
        # Create processor instance
        processor = StreamingHighlightProcessor()
        manager.processors[session_id] = processor
        
        await progress_tracker.send_update("Starting video processing...", 0.0)
        
        # Process video and get the file path
        result_file_path = await asyncio.get_event_loop().run_in_executor(
            None,
            processor.main_with_streaming,
            youtube_url,
            progress_tracker,
            video_streamer,
            fade_duration,
            padding,
            fps,
            yt_format
        )
        
        # Send download link instead of the file content
        if result_file_path and os.path.exists(result_file_path):
            await progress_tracker.send_download_link(result_file_path, "highlight_video.mp4")
        else:
            await progress_tracker.send_update(
                "Error: No output file generated", 
                data={"error": True}
            )
        
    except Exception as e:
        await progress_tracker.send_update(f"Error: {str(e)}", data={"error": True})
        logger.error(f"Error in process_video_async for session {session_id}: {e}")
    finally:
        # Mark session as not processing
        manager.processing_sessions[session_id] = False

@app.get("/download/{token}")
async def download_file(token: str, background_tasks: BackgroundTasks):
    """Stream file by token and clean up after download"""
    file_path = manager.get_file_by_token(token)
    
    if not file_path:
        raise HTTPException(status_code=404, detail="Download link expired or invalid")
    
    if not os.path.exists(file_path):
        manager.cleanup_download_token(token)
        raise HTTPException(status_code=404, detail="File not found")
    
    # Get file info for logging and headers
    token_info = manager.download_tokens.get(token, {})
    file_age = time.time() - token_info.get("created_at", 0)
    file_size = os.path.getsize(file_path)
    
    logger.info(f"📥 Streaming file: {file_path} (age: {file_age/60:.1f} minutes, size: {file_size} bytes)")
    
    # Schedule cleanup after download
    background_tasks.add_task(manager.cleanup_download_token, token)
    
    def file_streamer():
        """Generator function to stream the file in chunks"""
        try:
            with open(file_path, "rb") as file:
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = file.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        except Exception as e:
            logger.error(f"Error streaming file {file_path}: {e}")
            raise
    
    # Return streaming response with appropriate headers
    return StreamingResponse(
        file_streamer(),
        media_type="video/mp4",
        headers={
            "Content-Disposition": "attachment; filename=highlight_video.mp4",
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.get("/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    """Get status of a session"""
    is_connected = session_id in manager.active_connections
    is_processing = manager.processing_sessions.get(session_id, False)
    has_file = session_id in manager.session_files
    
    return {
        "session_id": session_id,
        "websocket_connected": is_connected,
        "processing": is_processing,
        "file_ready": has_file
    }

@app.delete("/sessions/{session_id}")
async def terminate_session(session_id: str):
    """Terminate a session, stop processing, and cleanup all resources"""
    if session_id not in manager.active_connections:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Send termination message to client before cleanup
    try:
        await manager.send_progress(session_id, {
            "type": "session_terminated",
            "message": "Session terminated by user",
            "data": {"terminated": True}
        })
    except:
        pass  # Client might already be disconnected
    
    # Stop processing
    if session_id in manager.processors:
        processor = manager.processors[session_id]
        processor.should_stop = True
    
    # Stop video streaming
    if session_id in manager.video_streamers:
        manager.video_streamers[session_id].stop_streaming()
    
    # Mark session as not processing
    if session_id in manager.processing_sessions:
        manager.processing_sessions[session_id] = False
    
    # Force cleanup of any temp files for this session
    await cleanup_session_files(session_id)
    
    # Disconnect the session
    manager.disconnect(session_id)
    
    return {
        "message": f"Session {session_id} terminated successfully",
        "processing_stopped": True,
        "files_cleaned": True
    }

async def cleanup_session_files(session_id: str):
    """Clean up all temporary files for a specific session"""
    import glob
    import time
    
    try:
        # Find recent temp files (last 10 minutes)
        current_time = time.time()
        recent_threshold = current_time - 600  # 10 minutes
        
        cleanup_files = []
        
        # Find recent temp files in tmp directory
        try:
            tmp_files = glob.glob("tmp/*.mp4") + glob.glob("tmp/*.mp3") + glob.glob("tmp/*.wav")
            for file_path in tmp_files:
                try:
                    file_time = os.path.getmtime(file_path)
                    if file_time > recent_threshold:
                        cleanup_files.append(file_path)
                except:
                    continue
        except:
            pass
        
        # Clean up files
        cleaned_files = []
        for file_path in set(cleanup_files):
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    cleaned_files.append(file_path)
                    logger.info(f"🗑️ Force deleted: {file_path}")
            except Exception as e:
                logger.warning(f"⚠️ Could not delete {file_path}: {e}")
                
        logger.info(f"🧹 Cleaned up {len(cleaned_files)} files for session {session_id}")
        
    except Exception as e:
        logger.error(f"Error during cleanup for session {session_id}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)