import SwiftUI

struct YoutubeVideoDetailView: View {
    let video: YoutubeVideo
    @State private var isSending = false
    @State private var didSend = false
    @State private var sendError: String? = nil

    var body: some View {
        VStack {
            Text(video.title)
                .font(.title2)
                .bold()
                .multilineTextAlignment(.center)
                .padding(.top)
            AsyncImage(url: URL(string: video.thumbnailURL)) { image in
                image.resizable()
            } placeholder: {
                ProgressView()
            }
            .aspectRatio(16/9, contentMode: .fit)
            .cornerRadius(8)
            .padding(.vertical)
            Text(video.description)
                .font(.body)
                .padding(.bottom)
            Spacer()
            Button(action: sendVideoLink) {
                if isSending {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                } else {
                    Text("Send to API")
                        .bold()
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                }
            }
            .disabled(isSending || didSend)
            .padding(.horizontal)
            if didSend {
                Text("Sent!")
                    .foregroundColor(.green)
                    .padding(.top, 4)
            }
            if let sendError {
                Text(sendError)
                    .foregroundColor(.red)
                    .padding(.top, 4)
            }
        }
        .padding()
    }

    func sendVideoLink() {
        guard let url = URL(string: "https://youtubeapi.thetechtitans.vip/youtube_url") else {
            sendError = "Invalid URL"
            return
        }
        isSending = true
        sendError = nil
        didSend = false
        let videoURL = "https://www.youtube.com/watch?v=\(video.id)"
        let body = ["link": videoURL]
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
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
                didSend = true
            }
        }.resume()
    }
}
