// VideoPreviewView.swift
// This view presents a preview of the selected YouTube video and sends the video link to the specified API when opened.

import SwiftUI
import WebKit
import AVKit
import AVFoundation
import Combine

struct VideoPreviewView: View {
    let video: YoutubeVideo
    @Environment(\.dismiss) private var dismiss
    @State private var hasSentRequest = false
    @State private var isSending = false
    @State private var didSend = false
    @State private var sendError: String? = nil
    @State private var streamURL: URL? = nil
    @State private var showStream = false

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text(video.title)
                    .font(.title2)
                    .bold()
                Spacer()
                Button("Close") { dismiss() }
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
            
            Button(action: {
                sendVideoLink()
            }) {
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
            .disabled(isSending || didSend)
            .padding(.horizontal)
            
            if let sendError {
                Text(sendError)
                    .foregroundColor(.red)
                    .font(.caption)
            }
        }
        .sheet(isPresented: $showStream) {
            if let streamURL = streamURL {
                VideoStreamPlayerView(url: streamURL)
            }
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
        // Supports both youtu.be and youtube.com URLs
        let host = url.host ?? ""
        if host.contains("youtu.be") {
            return url.lastPathComponent
        } else if host.contains("youtube.com") {
            let comps = URLComponents(url: url, resolvingAgainstBaseURL: false)
            let queryItems = comps?.queryItems ?? []
            return queryItems.first(where: { $0.name == "v" })?.value
        }
        return nil
    }

    func sendVideoLink() {
        guard !isSending else { return }
        isSending = true
        sendError = nil
        didSend = false
        guard let apiURL = URL(string: "https://youtubeapi.thetechtitans.vip/youtube_url") else {
            sendError = "Invalid API URL"
            isSending = false
            return
        }
        var request = URLRequest(url: apiURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let videoURL = "https://www.youtube.com/watch?v=\(video.id)"
        let payload = ["link": videoURL]
        request.httpBody = try? JSONSerialization.data(withJSONObject: payload)
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                isSending = false
                if let error = error {
                    sendError = error.localizedDescription
                    return
                }
                guard let httpResponse = response as? HTTPURLResponse, (200..<300).contains(httpResponse.statusCode) else {
                    sendError = "Request failed"
                    return
                }
                guard let data = data else {
                    sendError = "No data received"
                    return
                }
                // Write video data to a temp file
                let tempDir = FileManager.default.temporaryDirectory
                let tempFileURL = tempDir.appendingPathComponent(UUID().uuidString).appendingPathExtension("mp4")
                do {
                    try data.write(to: tempFileURL)
                    self.streamURL = tempFileURL
                    self.didSend = true
                    self.showStream = true
                } catch {
                    sendError = "Failed to save video file"
                }
            }
        }.resume()
    }
}

// Minimal SwiftUI WebView wrapper
struct WebView: UIViewRepresentable {
    let url: URL?
    func makeUIView(context: Context) -> WKWebView {
        return WKWebView()
    }
    func updateUIView(_ webView: WKWebView, context: Context) {
        if let url = url {
            let request = URLRequest(url: url)
            webView.load(request)
        }
    }
}

struct VideoStreamPlayerView: View {
    let url: URL
    @StateObject private var playerHolder = PlayerHolder()
    
    var body: some View {
        VideoPlayer(player: playerHolder.player)
            .onAppear {
                playerHolder.setURL(url)
            }
            .edgesIgnoringSafeArea(.all)
    }
    
    @MainActor
    class PlayerHolder: ObservableObject {
        @Published var player: AVPlayer = AVPlayer()
        func setURL(_ url: URL) {
            do {
                try AVAudioSession.sharedInstance().setCategory(.playback, mode: .moviePlayback)
                try AVAudioSession.sharedInstance().setActive(true)
            } catch {
                print("Audio session error: \(error)")
            }
            if (player.currentItem == nil || (player.currentItem?.asset as? AVURLAsset)?.url != url) {
                self.player.replaceCurrentItem(with: AVPlayerItem(url: url))
                self.player.isMuted = false
                self.player.play()
            }
        }
    }
}

