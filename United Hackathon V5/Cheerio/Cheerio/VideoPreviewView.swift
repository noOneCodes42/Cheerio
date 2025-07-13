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

// MARK: - DownloadedVideo model
struct DownloadedVideo: Identifiable, Equatable {
    let id: UUID
    let fileURL: URL
    let date: Date
    let isShort: Bool

    init(id: UUID = UUID(), fileURL: URL, date: Date = Date(), isShort: Bool = false) {
        self.id = id
        self.fileURL = fileURL
        self.date = date
        self.isShort = isShort
    }

    static func == (lhs: DownloadedVideo, rhs: DownloadedVideo) -> Bool {
        lhs.id == rhs.id && lhs.isShort == rhs.isShort
    }
}

// MARK: - DownloadManager observable class
class DownloadManager: ObservableObject {
    @Published var downloads: [DownloadedVideo] = []

    func add(_ video: DownloadedVideo) {
        downloads.append(video)
    }

    func remove(_ video: DownloadedVideo) {
        if let index = downloads.firstIndex(of: video) {
            let fileURL = downloads[index].fileURL
            do {
                try FileManager.default.removeItem(at: fileURL)
                print("🗑️ Deleted file at \(fileURL.path)")
            } catch {
                print("⚠️ Failed to delete file at \(fileURL.path): \(error.localizedDescription)")
            }
            downloads.remove(at: index)
        }
    }
}

// MARK: - MonitoredVideoPlayer View
struct MonitoredVideoPlayer: UIViewControllerRepresentable {
    let videoURL: URL
    let shouldPlay: Bool
    let id: UUID

    func makeUIViewController(context: Context) -> AVPlayerViewController {
        let controller = AVPlayerViewController()
        let player = AVPlayer(url: videoURL)
        controller.player = player
        controller.videoGravity = .resizeAspectFill
        context.coordinator.player = player
        if shouldPlay {
            player.play()
            NotificationCenter.default.post(name: .monitoredVideoPlayerDidStartPlaying, object: id)
        }
        // Listen for PauseAllVideos notification
        NotificationCenter.default.addObserver(context.coordinator,
                                               selector: #selector(Coordinator.pauseIfNeeded(_:)),
                                               name: .pauseAllVideos,
                                               object: nil)
        return controller
    }

    func updateUIViewController(_ uiViewController: AVPlayerViewController, context: Context) {
        if shouldPlay {
            uiViewController.player?.seek(to: .zero)
            uiViewController.player?.play()
            NotificationCenter.default.post(name: .monitoredVideoPlayerDidStartPlaying, object: id)
        } else {
            uiViewController.player?.pause()
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(id: id)
    }

    class Coordinator: NSObject {
        var player: AVPlayer?
        let id: UUID

        init(id: UUID) {
            self.id = id
            super.init()
        }

        @objc func pauseIfNeeded(_ notification: Notification) {
            // Notification object is UUID of player that started playing
            guard let playingID = notification.object as? UUID else { return }
            if playingID != id {
                player?.pause()
            }
        }

        deinit {
            NotificationCenter.default.removeObserver(self)
        }
    }
}

extension Notification.Name {
    static let pauseAllVideos = Notification.Name("PauseAllVideos")
    static let monitoredVideoPlayerDidStartPlaying = Notification.Name("MonitoredVideoPlayerDidStartPlaying")
}

// MARK: - VideoPlayerSheet
struct VideoPlayerSheet: View {
    let player: AVPlayer?
    let localURL: URL
    @Binding var isPresented: Bool
    let onKeep: (URL) -> Void
    let onClose: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            HStack {
                Spacer()
                Button(action: {
                    onClose()
                }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title)
                        .foregroundColor(Color.gray.opacity(0.7))
                }
                .padding()
            }

            VideoPlayer(player: player)
                .aspectRatio(9/16, contentMode: .fill)
                .frame(maxWidth: .infinity, maxHeight: 500)
                .clipped()

            Button("Keep Video") {
                onKeep(localURL)
                // No longer deleting video here; deletion only on explicit delete action
            }
            .padding()
            .frame(maxWidth: .infinity)
            .background(Color.blue)
            .foregroundColor(Color.white)
            .cornerRadius(10)
            .padding()
        }
    }
}

// MARK: - ProgressDetailView
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
                .font(.title)
                .fontWeight(.bold)
                .foregroundColor(.gray)
                .multilineTextAlignment(.center)
                .padding()
#if targetEnvironment(macCatalyst)
            ProgressView(value: Double(displayedProgress), total: 100)
                .scaleEffect(x: 1, y: 3, anchor: .center)
                .padding(.horizontal, 40)
#else
            ProgressView(value: Double(displayedProgress), total: 100)
                .progressViewStyle(LinearProgressViewStyle())
                .scaleEffect(x: 1, y: 3, anchor: .center)
                .padding(.horizontal, 40)
#endif
            Text("\(displayedProgress)%")
                .font(.largeTitle)
                .monospacedDigit()
                .foregroundColor(.gray)
            Spacer()
            Button("Cancel Processing") {
                isPresented = false
                onCancel()
            }
            .foregroundColor(Color.red)
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

// MARK: - SafariView
struct WebView: UIViewRepresentable {
    let url: URL?

    func makeUIView(context: Context) -> WKWebView {
        return WKWebView()
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {
        if let url = url {
            let request = URLRequest(url: url)
            if uiView.url != url {
                uiView.load(request)
            }
        }
    }
}

// MARK: - YouTubeHighlightWebSocket
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

// MARK: - VideoPreviewViewModel
class VideoPreviewViewModel: ObservableObject {
    @Published var tempVideoState: TempVideoState = .none
    @Published var showStream = false

    enum TempVideoState: Equatable {
        case none, downloading, downloaded(URL)

        static func == (lhs: TempVideoState, rhs: TempVideoState) -> Bool {
            switch (lhs, rhs) {
            case (.none, .none): return true
            case (.downloading, .downloading): return true
            case let (.downloaded(url1), .downloaded(url2)):
                return url1 == url2
            default:
                return false
            }
        }
    }

    func downloadVideo(from url: URL) {
        // The 'url' here is the direct media URL provided by the server.
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
                    DispatchQueue.main.async {
                        self.tempVideoState = .downloaded(destination)
                        self.showStream = true
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

// MARK: - VideoPreviewView
struct VideoPreviewView: View {
    enum TempVideoState {
        case none, downloading, downloaded(URL)
    }
    
    let video: YoutubeVideo
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject var downloadManager: DownloadManager
    @StateObject private var wsClient = YouTubeHighlightWebSocket()
    @StateObject private var viewModel = VideoPreviewViewModel()
    @State private var isSending = false
    @State private var didSend = false
    @State private var showProgressDetail = false
    @State private var player: AVPlayer? = nil

    // State to track if the video was saved explicitly by the user pressing "Keep Video"
    @State private var savedVideoURL: URL? = nil
    
    // States for result popup
    @State private var showResultPopup = false
    @State private var resultPopupMessage = ""
    @State private var resultPopupSuccess = true

    private var resultPopupOverlay: some View {
        Group {
            if showResultPopup {
                ZStack {
                    Color.black.opacity(0.5)
                        .edgesIgnoringSafeArea(.all)
                    VStack(spacing: 16) {
                        Text(resultPopupMessage)
                            .bold()
                            .foregroundColor(resultPopupSuccess ? Color.green : Color.red)
                            .padding()
                            .background(Color.white)
                            .cornerRadius(12)
                            .shadow(radius: 10)
                    }
                    .padding()
                }
                .transition(.opacity)
                .animation(.easeInOut, value: showResultPopup)
            }
        }
    }

    private var videoPlayerSheetContent: some View {
        Group {
            if case .downloaded(let localURL) = viewModel.tempVideoState {
                VideoPlayerSheet(
                    player: player,
                    localURL: localURL,
                    isPresented: $viewModel.showStream,
                    onKeep: { url in
                        let fileManager = FileManager.default
                        guard fileManager.fileExists(atPath: url.path) else {
                            print("❌ File at URL does not exist: \(url.path)")
                            resultPopupMessage = "Error: File does not exist."
                            resultPopupSuccess = false
                            DispatchQueue.main.async {
                                showResultPopup = true
                                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                                    showResultPopup = false
                                }
                            }
                            return
                        }
                        // Treat all downloaded videos as Shorts (isShort = true)
                        let newVideo = DownloadedVideo(fileURL: url, isShort: true)
                        if !downloadManager.downloads.contains(where: { $0.fileURL == url }) {
                            downloadManager.add(newVideo)
                            print(url)
                            print("✅ Added video to downloads")
                            resultPopupMessage = "Video saved to downloads!"
                            resultPopupSuccess = true

                            #if canImport(AudioToolbox)
                            AudioServicesPlaySystemSound(1322)
                            #endif

                            // Set savedVideoURL to track that the video was kept by the user
                            savedVideoURL = url
                        } else {
                            resultPopupMessage = "Video already in downloads."
                            resultPopupSuccess = false
                        }
                        DispatchQueue.main.async {
                            // Close the video player sheet after keeping
                            viewModel.showStream = false
                            showResultPopup = true
                            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                                showResultPopup = false
                            }
                        }
                    },
                    // onClose closure only closes sheet; no deletion of file here
                    onClose: {
                        // Close the sheet
                        viewModel.showStream = false
                        // We do NOT delete the temp video file here anymore.
                        // Deletion only happens explicitly via delete button.
                        // Commented out player cleanup and savedVideoURL clearing here as it is now handled in onDismiss
//                        savedVideoURL = nil
//                        player?.pause()
//                        player = nil
                    }
                )
            }
        }
    }
    
    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text(video.title)
                    .font(.title2)
                    .bold()
                    .padding()
                Spacer()
                Button("Close") {
                    NotificationCenter.default.post(name: .pauseAllVideos, object: UUID())
                    if let player = player {
                        player.pause()
                        player.seek(to: .zero)
                        player.volume = 0.0
                        player.replaceCurrentItem(with: nil)
                        self.player = nil
                    }
                    #if os(macOS)
                    try? AVAudioSession.sharedInstance().setActive(false, options: [])
                    #endif
                    dismiss()
                    wsClient.cancelProcessing()
                    savedVideoURL = nil
                }
                .padding(.horizontal)
            }
            .padding(.vertical)
            
            Divider()
            WebView(url: youtubeEmbedURL)
                .aspectRatio(16/9, contentMode: .fit)
                .frame(maxWidth: 420)
                .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                .shadow(color: .black.opacity(0.18), radius: 12, y: 4)
                .padding(.vertical, 16)
                .padding(.horizontal, 12)
            
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
                        .foregroundColor(Color.white)
                        .cornerRadius(8)
                }
            }
            .disabled(isSending || didSend || wsClient.sessionID == nil)
            .padding(.horizontal)
            
            if let error = wsClient.sendError {
                Text(error)
                    .foregroundColor(Color.red)
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
        .sheet(isPresented: $viewModel.showStream, onDismiss: {
            NotificationCenter.default.post(name: .pauseAllVideos, object: UUID())
            if let player = player {
                player.pause()
                player.seek(to: .zero)
                player.volume = 0.0
                player.replaceCurrentItem(with: nil)
                self.player = nil
            }
            #if os(macOS)
            try? AVAudioSession.sharedInstance().setActive(false, options: [])
            #endif
            savedVideoURL = nil
        }) {
            videoPlayerSheetContent
        }
        .onAppear {
            wsClient.connect()
        }
        .onDisappear {
            wsClient.disconnect()
        }
        .onChange(of: wsClient.downloadURL) { url in
            // The 'url' here is the direct media URL provided by the server,
            // NOT the original YouTube video URL.
            if let url = url {
                print("🚀 Triggering video stream view for URL: \(url)")
                #if canImport(AudioToolbox)
                AudioServicesPlaySystemSound(1322)
                #endif
                viewModel.downloadVideo(from: url)
            }
        }
        .onChange(of: viewModel.tempVideoState) { state in
            switch state {
            case .downloaded(let localURL):
                let playableAsset = AVAsset(url: localURL)
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
                            if player == nil {
                                player = AVPlayer(url: localURL)
                                player?.volume = 1.0
                                // Delay play slightly to allow AVFoundation to sync audio and video tracks better
                                DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                                    player?.play()
                                }
                            }
                            viewModel.showStream = true
                        }
                    } else {
                        print("❌ File not playable: \(String(describing: error?.localizedDescription))")
                        DispatchQueue.main.async {
                            viewModel.tempVideoState = .none
                        }
                    }
                }
            default:
                break
            }
        }
        .onChange(of: wsClient.progress) { progress in
            showProgressDetail = (progress ?? 0) < 1.0
        }
        .overlay(resultPopupOverlay)
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

// MARK: - CheerioScreeniosView
// MARK: - CheerioScreeniosView
import SwiftUI
import AVKit

struct CheerioScreeniosView: View {
    @EnvironmentObject var downloadManager: DownloadManager
    @State private var shortsSelection = 0
    @State private var othersSelection = 0
    @State private var currentShortIndex = 0

    private func shortsWithIDs(from shorts: [DownloadedVideo]) -> [(video: DownloadedVideo, id: UUID)] {
        shorts.map { ($0, $0.id) }
    }

    var body: some View {
        ZStack {
            if downloadManager.downloads.isEmpty {
                emptyState
            } else {
                content
            }
        }
        .navigationBarTitle("Cheerio Screenios", displayMode: .inline)
        .onDisappear {
            NotificationCenter.default.post(name: .pauseAllVideos, object: UUID())
            currentShortIndex = -1
        }
    }

    private var emptyState: some View {
        VStack(spacing: 24) {
            Spacer()
            Image(systemName: "tray.fill")
                .font(.system(size: 52))
                .foregroundColor(.gray)
            Text("No Cheerio Screenios yet!")
                .font(.title2).bold()
            Text("Your saved videos will appear here.")
                .foregroundColor(.secondary)
            Spacer()
        }
        .padding(.bottom, 100)
    }

    private var content: some View {
        let shorts = downloadManager.downloads.filter { $0.isShort }
        let others = downloadManager.downloads.filter { !$0.isShort }
        let paired = shortsWithIDs(from: shorts)

        return ScrollView {
            VStack(spacing: 32) {
                if !shorts.isEmpty {
                    VStack(alignment: .leading) {
                        Text("Shorts")
                            .font(.title2).bold()
                            .padding(.leading)

                        TabView(selection: $shortsSelection) {
                            ForEach(paired.indices, id: \.self) { index in
                                let tuple = paired[index]

                                VStack {
                                    ZStack(alignment: .bottom) {
                                        Color.black.ignoresSafeArea()

                                        MonitoredVideoPlayer(
                                            videoURL: tuple.video.fileURL,
                                            shouldPlay: index == currentShortIndex,
                                            id: tuple.id
                                        )
                                        .aspectRatio(9/16, contentMode: .fit)
                                        .frame(maxWidth: .infinity, maxHeight: 600)

                                        shortOverlay(for: tuple.video)
                                    }
                                    .frame(height: 600)
                                }
                                .padding(.vertical, 24)
                                .tag(index)
                            }
                        }
                        .tabViewStyle(.page(indexDisplayMode: .automatic))
                        .frame(height: 700)
                        .onChange(of: shortsSelection) { newValue in
                            withAnimation { currentShortIndex = newValue }
                            if paired.indices.contains(newValue) {
                                NotificationCenter.default.post(
                                    name: .pauseAllVideos,
                                    object: paired[newValue].id
                                )
                            }
                        }
                        .onAppear {
                            currentShortIndex = shortsSelection
                            if paired.indices.contains(currentShortIndex) {
                                NotificationCenter.default.post(
                                    name: .pauseAllVideos,
                                    object: paired[currentShortIndex].id
                                )
                            }
                        }
                    }
                }

                if !others.isEmpty {
                    downloadsSection(others: others)
                }
            }
        }
        .background(Color.black.ignoresSafeArea())
    }

    private func shortOverlay(for video: DownloadedVideo) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            LinearGradient(
                gradient: Gradient(colors: [.black.opacity(0.8), .clear]),
                startPoint: .bottom, endPoint: .top
            )
            .frame(height: 160)
            .overlay(
                VStack(alignment: .leading, spacing: 6) {
                    Text("Saved \(video.date.formatted(date: .abbreviated, time: .shortened))")
                        .font(.footnote)
                        .foregroundColor(.white.opacity(0.85))

                    Button {
                        withAnimation { delete(video) }
                    } label: {
                        Label("Delete", systemImage: "trash").bold()
                    }
                    .foregroundColor(.red)
                    .padding(.top, 8)
                }
                .padding(.horizontal)
                .padding(.bottom, 12),
                alignment: .bottomLeading
            )
        }
    }

    private func downloadsSection(others: [DownloadedVideo]) -> some View {
        VStack(alignment: .leading) {
            Text("Downloads")
                .font(.title2).bold()
                .padding(.leading)

            List {
                ForEach(others) { video in
                    HStack {
                        Image(systemName: "video.fill").foregroundColor(.blue)
                        Text("Saved \(video.date.formatted(date: .abbreviated, time: .shortened))")
                            .font(.footnote).foregroundColor(.gray)
                        Spacer()
                        Button {
                            play(video)
                        } label: {
                            Image(systemName: "play.fill")
                        }
                    }
                }
                .onDelete { indexSet in
                    indexSet.map { others[$0] }.forEach(delete)
                }
            }
            .listStyle(.plain)
            .frame(maxHeight: 350)
        }
    }

    private func play(_ video: DownloadedVideo) {
        let playerVC = AVPlayerViewController()
        playerVC.player = AVPlayer(url: video.fileURL)
        playerVC.videoGravity = .resizeAspectFill

        if let rootVC = UIApplication.shared.connectedScenes
            .compactMap({ $0 as? UIWindowScene })
            .flatMap({ $0.windows })
            .first(where: { $0.isKeyWindow })?.rootViewController {
            rootVC.present(playerVC, animated: true) {
                playerVC.player?.play()
            }
        }
    }

    private func delete(_ video: DownloadedVideo) {
        NotificationCenter.default.post(name: .pauseAllVideos, object: video.id)
        downloadManager.remove(video)
        if video.isShort {
            shortsSelection = 0
            currentShortIndex = 0
            NotificationCenter.default.post(name: .pauseAllVideos, object: UUID())
        } else {
            othersSelection = 0
        }
    }
}

#if DEBUG
struct CheerioScreeniosView_Previews: PreviewProvider {
    static var previews: some View {
        let manager = DownloadManager()
        CheerioScreeniosView()
            .environmentObject(manager)
    }
}
#endif

