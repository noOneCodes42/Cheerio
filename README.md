# Sports Highlight Processor ⚽🏀🏈

**Submission for United Hacks v5**

An intelligent video processing service specifically designed for **sports content** that automatically detects and creates highlight reels from YouTube sports videos using advanced audio analysis and machine learning. Built with FastAPI, WebSocket streaming, and powered by Google's YAMNet for audio classification to identify crowd reactions, cheering, and exciting moments in sports footage.

## 🌟 Features

- **🎯 Smart Sports Cheer Detection**: Uses Google's YAMNet neural network to identify crowd cheering, applause, and excitement moments specifically optimized for sports content
- **⚡ Real-time Processing**: WebSocket-based streaming with live progress updates for sports video analysis
- **🏆 Sports-Focused Pipeline**: Automatically filters for sports content (Category ID 17) for optimal crowd reaction detection
- **🎬 Professional Output**: Automatic crossfade transitions and optional watermarking perfect for sports highlight reels
- **🔄 Audio Synchronization**: Advanced FFT-based cross-correlation for perfect audio-video sync in sports broadcasts
- **📱 RESTful API**: Easy integration for sports apps, fantasy platforms, or team management tools
- **🛡️ Secure Downloads**: Token-based file delivery with automatic cleanup
- **🎵 Fallback Detection**: Librosa-based energy analysis when YAMNet is unavailable, tuned for sports audio patterns

## 🏅 Sports-Focused Features

### 🎯 **Optimized for Sports Content**
- **Category Filtering**: Automatically targets YouTube Sports category (ID: 17) for best results
- **Crowd Reaction Detection**: Specialized algorithms for detecting cheers, applause, and fan excitement
- **Game Audio Recognition**: Identifies whistles, buzzers, and sport-specific sounds
- **Commentary Analysis**: Detects broadcaster excitement and key call moments

### 🏆 **Smart Sports Timing**
- **30-Second Spacing**: Ensures highlights don't overlap, perfect for distinct plays
- **Configurable Padding**: Add context before/after exciting moments (default: 12 seconds)
- **Play-by-Play Awareness**: Understands natural breaks in sports action

## 🏗️ Architecture

```
├── api.py              # Legacy API endpoint (simple version)
├── main.py             # Main FastAPI app with WebSocket streaming
├── streaming_main.py   # Core sports video processing logic
└── test/
    └── client.py       # Python test client with sports examples
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install fastapi uvicorn websockets aiohttp
pip install librosa opencv-python numpy scipy matplotlib
pip install yt-dlp tensorflow tensorflow-hub
pip install python-dotenv pydantic

# Install FFmpeg (required for video processing)
# Ubuntu/Debian:
sudo apt update && sudo apt install ffmpeg

# macOS:
brew install ffmpeg

# Windows: Download from https://ffmpeg.org/
```

### Environment Setup

Create a `.env` file in the project root:

```bash
YOUTUBE_API_KEY=your_youtube_api_key_here
```

### Running the Server

```bash
# Start the streaming server
python main.py

# Or use uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000
```

The server will be available at:
- **WebSocket**: `ws://localhost:8000/ws`
- **HTTP API**: `http://localhost:8000`
- **API Docs**: `http://localhost:8000/docs`

## 📡 API Endpoints

### WebSocket Connection
- **`/ws`** - Main WebSocket endpoint for real-time communication

### HTTP Endpoints
- **`POST /youtube_search`** - Search YouTube videos
- **`POST /youtube_stream`** - Start video processing with streaming updates
- **`GET /download/{token}`** - Download processed video (auto-cleanup)
- **`GET /sessions/{session_id}/status`** - Check session status
- **`DELETE /sessions/{session_id}`** - Cancel processing and cleanup

## 🧪 Testing with Python Client

```bash
# Simple processing with auto-download
python test/client.py --test simple --url "https://www.youtube.com/watch?v=VIDEO_ID"

# Download-only test (no live streaming)
python test/client.py --test download --url "https://www.youtube.com/watch?v=VIDEO_ID"

# Video processing without auto-download (get link only)
python test/client.py --test video --url "https://www.youtube.com/watch?v=VIDEO_ID" --no-auto-download

# Check session status
python test/client.py --test status

# Use custom server
python test/client.py --server your-domain.com --test simple --url "https://youtube.com/watch?v=..."
```

## 🎬 Processing Pipeline

1. **📥 Download**: Extract video and audio from YouTube sports content using yt-dlp
2. **🏆 Sports Filtering**: Automatically focus on sports videos (Category ID 17) for optimal results
3. **🎵 Audio Analysis**: 
   - **Primary**: YAMNet neural network for semantic audio classification (crowd cheering, applause, excitement)
   - **Fallback**: Librosa energy-based analysis with spectral features optimized for sports audio
4. **🎯 Peak Detection**: Identify top excitement moments with smart spacing (30+ seconds apart) - perfect for game highlights
5. **🔄 Synchronization**: FFT-based cross-correlation for perfect audio-video alignment in sports broadcasts
6. **✂️ Clip Extraction**: Create highlight clips with configurable padding and watermarks ideal for sports content
7. **🎬 Final Assembly**: Crossfade transitions between clips for professional sports highlight reels
8. **📦 Delivery**: Secure token-based download with automatic cleanup

## ⚙️ Configuration Options

```json
{
  "fade_duration": 1.0,
  "padding": 12.0,
  "fps": 60,
  "yt_format": "bestvideo[vcodec!*=av01][height<=720]+bestaudio/best[height<=720]"
}
```

## 🎯 YAMNet Sports Audio Classification

The system uses Google's YAMNet model specifically tuned for sports content to detect:
- **Crowd Cheering & Applause** - Fan reactions to goals, touchdowns, great plays
- **Excitement Peaks** - Moments of high energy and crowd engagement  
- **Stadium Atmosphere** - Crowd noise, chants, and collective reactions
- **Commentary Excitement** - Broadcaster enthusiasm during key moments
- **Sports-Specific Sounds** - Whistles, buzzers, and game-related audio cues

## 🛡️ Security Features

- **🔐 Token-based Downloads**: Secure file access with expiring tokens
- **⏰ Automatic Cleanup**: Files auto-delete after 15 minutes
- **🚫 Session Isolation**: Each client gets isolated processing environment
- **⛔ Process Termination**: Graceful cancellation with resource cleanup

## 📊 WebSocket Message Types

### Progress Updates
```json
{
  "type": "progress",
  "message": "Processing step description",
  "progress": 0.75,
  "step": 4,
  "total_steps": 6,
  "data": {
    "intervals": [...],
    "method": "yamnet"
  }
}
```

### Download Ready
```json
{
  "type": "progress",
  "data": {
    "completed": true,
    "download_url": "/download/secure_token",
    "filename": "highlight_video.mp4"
  }
}
```

## 🚧 Development & Deployment

### Local Development
```bash
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn main:app --reload --port 8000

# Test with client
python test/client.py --server localhost:8000 --test simple --url "YOUR_URL"
```

### Production Deployment
```bash
# Using uvicorn with multiple workers
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Or with gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 🎯 Use Cases

- **🏆 Game Highlights**: Automatically extract the most exciting moments from full game recordings
- **⚽ Goal Compilations**: Create instant highlight reels from soccer, hockey, or basketball games
- **🏈 Big Play Moments**: Capture touchdowns, interceptions, and game-changing plays
- **🎾 Match Summaries**: Generate exciting point compilations from tennis, volleyball, or badminton
- **🏀 Clutch Moments**: Extract buzzer-beaters, slam dunks, and crowd-pleasing plays
- **📱 Social Media Sports Content**: Generate shareable sports clips for team accounts and fan pages
- **📺 Sports Broadcasting**: Automated highlight packages for sports news and recap shows
- **🎮 Sports Gaming Content**: Create reaction compilations from FIFA, NBA 2K, or other sports games

## 🤝 Contributing

This project was built for United Hacks v5. Feel free to:

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request

## 📋 Requirements

- **Python 3.8+**
- **FFmpeg** (for video processing)
- **YouTube API Key** (for metadata, optional)
- **8GB+ RAM** (recommended for video processing)

## 🐛 Troubleshooting

### Common Issues

**YAMNet Not Loading**: Falls back to librosa energy detection automatically

**FFmpeg Errors**: Ensure FFmpeg is installed and accessible in PATH

**Memory Issues**: Reduce video quality in `yt_format` parameter

**Download Failures**: Check if files expired (15-minute limit)

## 📜 License

Built for United Hacks v5 - See hackathon guidelines for usage terms.

## 🎉 Acknowledgments

- **Google YAMNet** - Audio classification model
- **librosa** - Audio analysis library  
- **yt-dlp** - YouTube downloading
- **FastAPI** - Modern Python web framework
- **United Hacks v5** - Inspiring this project

---

**Made with ❤️ for United Hacks v5**

*Transform any sports video into an engaging highlight reel with the power of AI! Perfect for teams, fans, coaches, and sports content creators! 🏆⚽🏀*