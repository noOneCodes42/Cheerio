    // COMPLETE VideoPreviewView.swift with full UI integration




    

import SwiftUI
import WebKit
import AVKit
import AVFoundation
import Combine
import AudioToolbox
#if canImport(CoreHaptics)
import CoreHaptics
#endif

import Foundation

class YouTubeHighlightWebSocket: ObservableObject {
    private var webSocketTask: URLSessionWebSocketTask?
    @Published var sessionID: String? = nil
    @Published var downloadURL: URL? = nil
    @Published var progressMessage: String? = nil
    @Published var progress: Double? = nil
    @Published var isSending: Bool = false
    @Published var sendError: String? = nil

    private let url = URL(string: "wss://thetechtitans.vip/ws")!

    func connect() {
        print("🔌 Connecting to WebSocket: \(url)")
        let session = URLSession(configuration: .default)
        webSocketTask = session.webSocketTask(with: url)
        webSocketTask?.resume()
        receiveMessage()
    }

    func receiveMessage() {
        print("👂 Waiting for WebSocket message...")
        webSocketTask?.receive { [weak self] result in
            switch result {
            case .failure(let error):
                print("WebSocket receive error: \(error)")
                DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                    self?.connect()
                }
            case .success(let message):
                switch message {
                case .string(let text):
                    print("📩 Received string message: \(text)")
                    self?.handleMessage(text)
                default:
                    break
                }
                self?.receiveMessage()
            }
        }
    }

    func handleMessage(_ text: String) {
        print("🔍 Handling message: \(text)")
        guard let data = text.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String else {
            return
        }

        if type == "session_created", let sessionID = json["session_id"] as? String {
            DispatchQueue.main.async { self.sessionID = sessionID }
        }

        if type == "progress" {
            DispatchQueue.main.async {
                self.progressMessage = json["message"] as? String
                self.progress = json["progress"] as? Double
            }

            if let data = json["data"] as? [String: Any],
               let completed = data["completed"] as? Bool,
               completed,
               let downloadPath = data["download_url"] as? String {

                let fullURLString = downloadPath.hasPrefix("http") ? downloadPath : "https://thetechtitans.vip\(downloadPath)"

                if let url = URL(string: fullURLString) {
                    DispatchQueue.main.async {
                        self.downloadURL = url
                        self.isSending = false
                    }
                }
            } else if let errorMessage = json["data"] as? [String: Any],
                      let error = errorMessage["error"] as? String {
                DispatchQueue.main.async {
                    self.sendError = error
                    self.isSending = false
                }
            }
        }

        if type == "session_terminated" {
            DispatchQueue.main.async {
                self.sessionID = nil
                self.downloadURL = nil
                self.progress = nil
                self.progressMessage = "Processing cancelled"
                self.isSending = false
            }
        }
    }

    func cancelProcessing() {
        guard let sessionID = sessionID else { return }
        let cancelURL = URL(string: "https://thetechtitans.vip/sessions/\(sessionID)")!
        var request = URLRequest(url: cancelURL)
        request.httpMethod = "DELETE"

        URLSession.shared.dataTask(with: request) { data, response, error in
            if let httpResponse = response as? HTTPURLResponse, (200..<300).contains(httpResponse.statusCode) {
                DispatchQueue.main.async {
                    self.sessionID = nil
                    self.downloadURL = nil
                    self.progress = nil
                    self.progressMessage = "Processing cancelled"
                    self.isSending = false
                }
            }
        }.resume()
    }

    func disconnect() {
        webSocketTask?.cancel(with: .goingAway, reason: nil)
    }
}

class VideoPreviewViewModel: ObservableObject {
    @Published var tempVideoState: TempVideoState = .none
    @Published var showStream = false
    @Published var videoPlayer: AVPlayer? = nil

    enum TempVideoState {
        case none, downloading, downloaded(URL)
    }

    func downloadVideo(from url: URL) {
        tempVideoState = .downloading
        let task = URLSession.shared.downloadTask(with: url) { tempURL, _, error in
            if let tempURL = tempURL {
                let fileManager = FileManager.default
                let documents = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
                let destination = documents.appendingPathComponent(url.lastPathComponent).appendingPathExtension("mp4")
                do {
                    if fileManager.fileExists(atPath: destination.path) {
                        try fileManager.removeItem(at: destination)
                    }
                    try fileManager.copyItem(at: tempURL, to: destination)
                    print("✅ Video saved to: \(destination.path)")

                    let playableAsset = AVAsset(url: destination)
                    playableAsset.loadValuesAsynchronously(forKeys: ["playable"]) {
                        var error: NSError? = nil
                        let status = playableAsset.statusOfValue(forKey: "playable", error: &error)
                        if status == .loaded {
                            let audioTracks = playableAsset.tracks(withMediaType: .audio)
                            print("🎧 Audio track count: \(audioTracks.count)")
                            DispatchQueue.main.async {
                                do {
                                    try AVAudioSession.sharedInstance().setCategory(.playback, mode: .moviePlayback, options: [])
                                    try AVAudioSession.sharedInstance().setActive(true)
                                } catch {
                                    print("⚠️ Failed to set AVAudioSession: \(error.localizedDescription)")
                                }

                                self.videoPlayer = AVPlayer(url: destination)
                                self.videoPlayer?.volume = 1.0
                                self.videoPlayer?.play()
                                self.tempVideoState = .downloaded(destination)
                                self.showStream = true
                            }
                        } else {
                            print("❌ File not playable: \(String(describing: error?.localizedDescription))")
                            DispatchQueue.main.async {
                                self.tempVideoState = .none
                            }
                        }
                    }
                } catch {
                    print("❌ Error saving file: \(error)")
                    DispatchQueue.main.async {
                        self.tempVideoState = .none
                    }
                }
            } else if let error = error {
                print("❌ Download failed: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    self.tempVideoState = .none
                }
            }
        }
        task.resume()
    }
}







// Use your previously defined YouTubeHighlightWebSocket here

struct VideoPreviewView: View {
    enum TempVideoState {
        case none, downloading, downloaded(URL)
    }
    let video: YoutubeVideo
    @Environment(\.dismiss) private var dismiss
    @StateObject private var wsClient = YouTubeHighlightWebSocket()
    @StateObject private var viewModel = VideoPreviewViewModel()
    @State private var isSending = false
    @State private var didSend = false
    @State private var showProgressDetail = false
    @State private var streamURL: URL? = nil
    
    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text(video.title)
                    .font(.title2)
                    .bold()
                    .padding()
                Spacer()
                Button("Close") {
                    dismiss()
                    wsClient.cancelProcessing()
                    if case .downloaded(let localURL) = viewModel.tempVideoState {
                        try? FileManager.default.removeItem(at: localURL)
                        print("🗑️ Deleted temporary video at: \(localURL.path)")
                        viewModel.tempVideoState = .none
                    }
                }
                .padding(.horizontal)
            }
            .padding(.vertical)
            
            Divider()
            WebView(url: youtubeEmbedURL)
                .edgesIgnoringSafeArea(.all)
            
            Divider()
            Text(video.description)
                .font(.subheadline)
                .padding()
            
            Button(action: sendVideoLink) {
                if isSending {
                    ProgressView()
                } else {
                    Text(didSend ? "Sent!" : "Send to Server")
                        .bold()
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(didSend ? Color.green : Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                }
            }
            .disabled(isSending || didSend || wsClient.sessionID == nil)
            .padding(.horizontal)
            
            if let error = wsClient.sendError {
                Text(error)
                    .foregroundColor(.red)
                    .font(.caption)
            }
        }
        .sheet(isPresented: $showProgressDetail) {
            ProgressDetailView(
                progress: wsClient.progress ?? 0,
                message: wsClient.progressMessage ?? "",
                isPresented: $showProgressDetail,
                onCancel: wsClient.cancelProcessing
            )
        }
        .sheet(isPresented: $viewModel.showStream) {
            if case .downloaded(let localURL) = viewModel.tempVideoState {
                VStack(spacing: 0) {
                    VideoPlayer(player: AVPlayer(url: localURL))
                    Button("Keep Video") {
                        viewModel.showStream = false
                    }
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                    .padding()
                }
            }
        }
        .onAppear {
            wsClient.connect()
        }
        .onDisappear {
            wsClient.disconnect()
        }
        .onChange(of: wsClient.downloadURL) { url in
            if let url = url {
                print("🚀 Triggering video stream view for URL: \(url)")
                viewModel.downloadVideo(from: url)
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    viewModel.showStream = true
                }
            }
        }
        .onChange(of: wsClient.progress) { progress in
            showProgressDetail = (progress ?? 0) < 1.0
        }
    }
    
    var youtubeEmbedURL: URL? {
        let videoURLString = "https://www.youtube.com/watch?v=\(video.id)"
        if let url = URL(string: videoURLString), let videoID = extractYoutubeID(from: url) {
            return URL(string: "https://www.youtube.com/embed/\(videoID)?autoplay=1")
        }
        return URL(string: videoURLString)
    }
    
    func extractYoutubeID(from url: URL) -> String? {
        let host = url.host ?? ""
        if host.contains("youtu.be") {
            return url.lastPathComponent
        } else if host.contains("youtube.com") {
            let comps = URLComponents(url: url, resolvingAgainstBaseURL: false)
            return comps?.queryItems?.first(where: { $0.name == "v" })?.value
        }
        return nil
    }
    
    func sendVideoLink() {
        guard !isSending, let sessionID = wsClient.sessionID else {
            wsClient.sendError = "WebSocket not connected"
            return
        }
        
        isSending = true
        wsClient.sendError = nil
        didSend = false
        
        guard let apiURL = URL(string: "https://thetechtitans.vip/youtube_stream") else {
            wsClient.sendError = "Invalid API URL"
            isSending = false
            return
        }
        
        var request = URLRequest(url: apiURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let payload: [String: Any] = [
            "link": "https://www.youtube.com/watch?v=\(video.id)",
            "session_id": sessionID,
            "fade_duration": 1.0,
            "padding": 12.0,
            "fps": 60,
            "yt_format": "bestvideo[vcodec!*=av01][height<=720]+bestaudio/best[height<=720]",
            "enable_streaming": true
        ]
        
        request.httpBody = try? JSONSerialization.data(withJSONObject: payload)
        
        URLSession.shared.dataTask(with: request) { _, response, error in
            DispatchQueue.main.async {
                isSending = false
                if let error = error {
                    wsClient.sendError = error.localizedDescription
                    return
                }
                guard let httpResponse = response as? HTTPURLResponse, (200..<300).contains(httpResponse.statusCode) else {
                    wsClient.sendError = "Request failed"
                    return
                }
                didSend = true
            }
        }.resume()
    }
    


}

import SafariServices

struct SafariView: UIViewControllerRepresentable {
    let url: URL
    func makeUIViewController(context: Context) -> SFSafariViewController {
        return SFSafariViewController(url: url)
    }
    func updateUIViewController(_ vc: SFSafariViewController, context: Context) {}
}

struct WebView: UIViewRepresentable {
    let url: URL?
    func makeUIView(context: Context) -> WKWebView {
        WKWebView()
    }
    func updateUIView(_ webView: WKWebView, context: Context) {
        if let url = url {
            webView.load(URLRequest(url: url))
        }
    }
}

struct ProgressDetailView: View {
    let progress: Double
    let message: String
    @Binding var isPresented: Bool
    let onCancel: () -> Void
    
    @State private var displayedProgress: Int = 0
    @State private var timer: Timer? = nil
    
    var body: some View {
        VStack(spacing: 30) {
            Spacer()
            Text(message)
                .font(.title2)
                .multilineTextAlignment(.center)
            ProgressView(value: Double(displayedProgress), total: 100)
                .progressViewStyle(LinearProgressViewStyle())
                .scaleEffect(x: 1, y: 3, anchor: .center)
                .padding(.horizontal, 40)
            Text("\(displayedProgress)%")
                .font(.largeTitle)
                .monospacedDigit()
                .foregroundColor(.gray)
            Spacer()
            Button("Cancel Processing") {
                isPresented = false
                onCancel()
            }
            .foregroundColor(.red)
            .padding()
        }
        .onAppear {
            updateProgress(to: Int(progress * 100))
        }
        .onChange(of: progress) { newValue in
            updateProgress(to: Int(newValue * 100))
        }
    }
    
    func updateProgress(to newValue: Int) {
        timer?.invalidate()
        let difference = abs(newValue - displayedProgress)
        if difference > 2 {
            var current = displayedProgress
            timer = Timer.scheduledTimer(withTimeInterval: 0.03, repeats: true) { t in
                if current == newValue {
                    t.invalidate()
                    timer = nil
                } else {
                    current += (current < newValue ? 1 : -1)
                    displayedProgress = current
                }
            }
        } else {
            displayedProgress = newValue
        }
    }
}

