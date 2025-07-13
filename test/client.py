#!/usr/bin/env python3
"""
Python client for testing YouTube Highlight Streaming API
Supports WebSocket streaming and file downloads (no WebRTC complexity)
"""

import asyncio
import websockets
import json
import aiohttp
import logging
import cv2
import numpy as np
from typing import Optional, Callable
import argparse
import sys
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YouTubeHighlightClient:
    """Python client for YouTube Highlight Streaming API - Simplified without WebRTC"""
    
    def __init__(self, server_url: str = "wss://thetechtitans.vip"):
        self.server_url = server_url
        self.http_url = self._convert_to_http_url(server_url)
        self.websocket = None
        self.session_id = None
        self.is_connected = False
        self.is_processing = False
        
        # Callbacks
        self.progress_callback: Optional[Callable] = None
        self.video_callback: Optional[Callable] = None
        self.completion_callback: Optional[Callable] = None
        self.error_callback: Optional[Callable] = None
        self.download_callback: Optional[Callable] = None
        
    def _convert_to_http_url(self, server_url: str) -> str:
        """Convert WebSocket URL to HTTP URL properly"""
        if server_url.startswith("wss://"):
            http_url = server_url.replace("wss://", "https://")
        elif server_url.startswith("ws://"):
            http_url = server_url.replace("ws://", "http://")
        elif server_url.startswith("https://") or server_url.startswith("http://"):
            http_url = server_url
        else:
            # Assume HTTPS for domain names without protocol
            http_url = f"https://{server_url}"
        
        # Remove /ws suffix if present
        if http_url.endswith("/ws"):
            http_url = http_url[:-3]
            
        return http_url
    
    def _get_websocket_url(self, server_url: str) -> str:
        """Get proper WebSocket URL"""
        if server_url.startswith("wss://") or server_url.startswith("ws://"):
            ws_url = server_url
        elif server_url.startswith("https://"):
            ws_url = server_url.replace("https://", "wss://")
        elif server_url.startswith("http://"):
            ws_url = server_url.replace("http://", "ws://")
        else:
            # Assume WSS for domain names without protocol
            ws_url = f"wss://{server_url}"
        
        # Ensure /ws suffix
        if not ws_url.endswith("/ws"):
            ws_url = f"{ws_url}/ws"
            
        return ws_url
        
    async def connect(self) -> str:
        """Connect to WebSocket and get session ID"""
        try:
            ws_url = self._get_websocket_url(self.server_url)
            logger.info(f"Connecting to {ws_url}")
            
            self.websocket = await websockets.connect(ws_url)
            self.is_connected = True
            logger.info("✅ WebSocket connected, waiting for session ID...")
            
            # Wait for session_created message
            while True:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                if data.get("type") == "session_created":
                    self.session_id = data["session_id"]
                    logger.info(f"✅ Session created: {self.session_id}")
                    return self.session_id
                    
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            raise
    
    async def start_processing(self, youtube_url: str, **options) -> bool:
        """Start processing YouTube video"""
        if not self.session_id:
            raise RuntimeError("Must connect first")
            
        if self.is_processing:
            logger.warning("⚠️  Already processing a video")
            return False
        
        # Default options with updated values
        request_data = {
            "link": youtube_url,
            "session_id": self.session_id,
            "fade_duration": options.get("fade_duration", 1.0),
            "padding": options.get("padding", 12.0),  # Updated to 12.0
            "fps": options.get("fps", 60),             # Updated to 60
            "yt_format": options.get("yt_format", "bestvideo[vcodec!*=av01][height<=720]+bestaudio/best[height<=720]")
        }
        
        # Construct proper HTTP URL
        endpoint_url = f"{self.http_url}/youtube_stream"
        logger.info(f"📤 Sending request to: {endpoint_url}")
        logger.info(f"🔍 Request data: {json.dumps(request_data, indent=2)}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint_url, json=request_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"🚀 Processing started: {result['message']}")
                        self.is_processing = True
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Failed to start processing: HTTP {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Error starting processing: {e}")
            return False
    
    async def listen_for_messages(self):
        """Listen for WebSocket messages (progress updates, video frames, download links)"""
        if not self.websocket:
            return
            
        try:
            while self.is_connected:
                message = await self.websocket.recv()
                data = json.loads(message)
                await self._handle_message(data)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 WebSocket connection closed")
            self.is_connected = False
        except Exception as e:
            logger.error(f"❌ Error listening for messages: {e}")
    
    async def _handle_message(self, data):
        """Handle incoming WebSocket messages"""
        msg_type = data.get("type")
        
        if msg_type == "progress":
            await self._handle_progress(data)
        elif msg_type == "session_terminated":
            self.is_processing = False
            logger.info("⏹️ Processing terminated by server")
        elif msg_type == "ping":
            await self.websocket.send(json.dumps({"type": "pong"}))
        elif msg_type == "pong":
            pass  # Keep-alive
        else:
            logger.debug(f"Unknown message type: {msg_type}")
    
    async def _handle_video_frame(self, frame_data):
        """Handle video frame received via WebSocket"""
        if frame_data and self.video_callback:
            try:
                # Decode base64 image
                import base64
                frame_bytes = base64.b64decode(frame_data)
                
                # Convert to numpy array
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Convert BGR to RGB for consistency
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.video_callback(frame_rgb)
                
            except Exception as e:
                logger.error(f"Error processing video frame: {e}")
    
    async def _handle_progress(self, data):
        """Handle progress updates"""
        message = data.get("message", "")
        progress = data.get("progress", 0)
        step = data.get("step", 0)
        total_steps = data.get("total_steps", 1)
        progress_data = data.get("data", {})
        
        logger.info(f"📊 Progress: {message} ({progress*100:.1f}% - Step {step}/{total_steps})")
        
        # Call progress callback
        if self.progress_callback:
            self.progress_callback(data)
        
        # Check for completion with download link
        if progress_data.get("completed") and progress_data.get("download_url"):
            self.is_processing = False
            download_url = progress_data["download_url"]
            filename = progress_data.get("filename", "highlight_video.mp4")
            
            logger.info(f"✅ Processing completed! Download available at: {download_url}")
            
            # Call download callback
            if self.download_callback:
                self.download_callback({
                    "download_url": download_url,
                    "filename": filename,
                    "full_url": f"{self.http_url}{download_url}"
                })
            
            # Call completion callback
            if self.completion_callback:
                self.completion_callback({
                    "completed": True,
                    "download_url": download_url,
                    "filename": filename,
                    "message": progress_data.get("message", "Download ready")
                })
                
        elif progress_data.get("error"):
            self.is_processing = False
            logger.error(f"❌ Processing error: {message}")
            if self.error_callback:
                self.error_callback(message)
                
        elif progress_data.get("stopped") or progress_data.get("terminated"):
            self.is_processing = False
            logger.info("⏹️ Processing stopped/terminated")
    
    async def download_file(self, download_url: str, save_path: str = "highlight_video.mp4") -> bool:
        """Download the processed video file"""
        try:
            full_url = f"{self.http_url}{download_url}"
            logger.info(f"📥 Downloading file from: {full_url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(full_url) as response:
                    if response.status == 200:
                        with open(save_path, "wb") as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                        
                        file_size = len(open(save_path, "rb").read())
                        logger.info(f"✅ File downloaded successfully: {save_path} ({file_size} bytes)")
                        return True
                    else:
                        logger.error(f"❌ Download failed: HTTP {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Error downloading file: {e}")
            return False
    
    async def cancel_processing(self):
        """Cancel current processing"""
        if not self.session_id:
            logger.warning("No active session to cancel")
            return False
        
        try:
            cancel_url = f"{self.http_url}/sessions/{self.session_id}"
            logger.info(f"🛑 Sending cancel request to: {cancel_url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.delete(cancel_url) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"✅ Processing cancelled: {result['message']}")
                        self.is_processing = False
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Failed to cancel: HTTP {response.status} - {error_text}")
                        return False
        except Exception as e:
            logger.error(f"❌ Error cancelling processing: {e}")
            return False
    
    async def get_status(self) -> dict:
        """Get session status"""
        if not self.session_id:
            return {"error": "Not connected"}
            
        try:
            status_url = f"{self.http_url}/sessions/{self.session_id}/status"
            async with aiohttp.ClientSession() as session:
                async with session.get(status_url) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"error": f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {"error": str(e)}
    
    async def disconnect(self):
        """Disconnect and cleanup"""
        logger.info("🔌 Disconnecting...")
        
        self.is_connected = False
            
        if self.websocket:
            await self.websocket.close()
        
        logger.info("✅ Disconnected")

# Test functions
async def simple_test_with_video(server_url: str, youtube_url: str, display_video: bool = False, auto_download: bool = True):
    """Simple test with progress tracking and automatic file download (no live video)"""
    client = YouTubeHighlightClient(server_url)
    
    # Track download info
    download_info = {}
    
    # Progress callback
    def on_progress(data):
        progress = data.get("progress", 0)
        message = data.get("message", "")
        intervals = data.get("data", {}).get("intervals", [])
        if intervals:
            print(f"📊 Progress: {progress*100:.1f}% - {message} (Found {len(intervals)} highlights)")
        else:
            print(f"📊 Progress: {progress*100:.1f}% - {message}")
    
    # Note: Video streaming is disabled for stability
    def on_video_frame(frame):
        pass  # No-op since live streaming is disabled
    
    # Download callback
    def on_download_ready(data):
        nonlocal download_info
        download_info = data
        print(f"📎 Download ready!")
        print(f"📁 Filename: {data['filename']}")
        print(f"🔗 Download URL: {data['download_url']}")
        print(f"🌐 Full URL: {data['full_url']}")
    
    # Completion callback
    def on_complete(data):
        print(f"✅ Processing completed!")
        if data.get("download_url"):
            print(f"📎 File ready for download: {data['filename']}")
    
    def on_error(message):
        print(f"❌ Error: {message}")
    
    client.progress_callback = on_progress
    client.video_callback = on_video_frame  # Still set but won't be called
    client.download_callback = on_download_ready
    client.completion_callback = on_complete
    client.error_callback = on_error
    
    try:
        # Connect
        session_id = await client.connect()
        print(f"🔗 Connected with session ID: {session_id}")
        print(f"🔗 HTTP URL: {client.http_url}")
        
        # Start processing
        success = await client.start_processing(youtube_url)
        if not success:
            print("❌ Failed to start processing")
            return
        
        print("⏳ Processing video with YAMNet cheer detection...")
        print("💡 Live video streaming disabled for stability - download will be available when complete")
        
        # Listen for messages (progress updates and download links)
        await client.listen_for_messages()
        
        # If auto-download is enabled and we have download info, download the file
        if auto_download and download_info:
            print(f"\n📥 Auto-downloading file...")
            download_success = await client.download_file(
                download_info["download_url"], 
                download_info["filename"]
            )
            if download_success:
                print(f"🎉 Video downloaded successfully as: {download_info['filename']}")
            else:
                print(f"❌ Download failed, but you can manually download from: {download_info['full_url']}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user - cancelling processing...")
        await client.cancel_processing()
    finally:
        await client.disconnect()

async def download_only_test(server_url: str, youtube_url: str):
    """Test that processes video and downloads file without live streaming"""
    client = YouTubeHighlightClient(server_url)
    
    download_info = {}
    
    def on_progress(data):
        progress = data.get("progress", 0)
        message = data.get("message", "")
        print(f"📊 {progress*100:.1f}% - {message}")
    
    def on_download_ready(data):
        nonlocal download_info
        download_info = data
        print(f"\n🎉 Processing complete! Download ready.")
        print(f"📁 Filename: {data['filename']}")
        print(f"🔗 Download URL: {data['download_url']}")
    
    client.progress_callback = on_progress
    client.download_callback = on_download_ready
    
    try:
        # Connect and process
        session_id = await client.connect()
        print(f"🔗 Connected with session ID: {session_id}")
        
        success = await client.start_processing(youtube_url)
        if not success:
            print("❌ Failed to start processing")
            return
        
        print("⏳ Processing video (no live stream)...")
        
        # Listen for messages until completion
        await client.listen_for_messages()
        
        # Download the file
        if download_info:
            print(f"\n📥 Downloading processed video...")
            download_success = await client.download_file(
                download_info["download_url"], 
                "my_highlight_video.mp4"
            )
            if download_success:
                print(f"✅ Video saved as: my_highlight_video.mp4")
            else:
                print(f"❌ Download failed")
        else:
            print("❌ No download information received")
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        await client.cancel_processing()
    finally:
        await client.disconnect()

async def status_test(server_url: str):
    """Test session status endpoint"""
    client = YouTubeHighlightClient(server_url)
    
    try:
        session_id = await client.connect()
        print(f"Session ID: {session_id}")
        print(f"HTTP URL: {client.http_url}")
        
        status = await client.get_status()
        print(f"Status: {json.dumps(status, indent=2)}")
        
    finally:
        await client.disconnect()

# CLI interface
def main():
    parser = argparse.ArgumentParser(description="Test YouTube Highlight Streaming API")
    parser.add_argument("--server", default="thetechtitans.vip", help="Server URL (domain name or full URL)")
    parser.add_argument("--test", choices=["simple", "video", "download", "status"], default="video", help="Test type")
    parser.add_argument("--url", help="YouTube URL to process")
    parser.add_argument("--display", action="store_true", help="Display video in OpenCV window")
    parser.add_argument("--no-auto-download", action="store_true", help="Don't automatically download the file")
    
    args = parser.parse_args()
    
    if args.test in ["simple", "video", "download"] and not args.url:
        print("❌ YouTube URL required for processing tests")
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.test == "simple":
            asyncio.run(simple_test_with_video(args.server, args.url, False, not args.no_auto_download))
        elif args.test == "video":
            asyncio.run(simple_test_with_video(args.server, args.url, args.display, not args.no_auto_download))
        elif args.test == "download":
            asyncio.run(download_only_test(args.server, args.url))
        elif args.test == "status":
            asyncio.run(status_test(args.server))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()

# Example usage:
"""
# Simple test with auto-download
python client.py --test simple --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Video streaming test with live display and auto-download
python client.py --test video --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --display

# Download-only test (no live streaming)
python client.py --test download --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Video streaming without auto-download (get link only)
python client.py --test video --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --no-auto-download

# Status test
python client.py --test status

# Using different server formats:
python client.py --server thetechtitans.vip --test video --url "https://youtube.com/watch?v=..."
python client.py --server https://thetechtitans.vip --test video --url "https://youtube.com/watch?v=..."
"""