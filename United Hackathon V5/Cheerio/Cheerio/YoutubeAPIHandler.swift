//
//  YoutubeAPIHandler.swift
//  Cheerio
//
//  Created by Neel Arora on 7/11/25.
//

import Foundation

struct YoutubeVideo: Codable, Identifiable {
    let id: String
    let title: String
    let description: String
    let thumbnailURL: String
}

struct YoutubeSearchResponse: Codable {
    let items: [YoutubeAPIItem]
}

struct YoutubeAPIItem: Codable {
    let id: YoutubeID
    let snippet: YoutubeSnippet
}

struct YoutubeID: Codable {
    let videoId: String?
}

struct YoutubeSnippet: Codable {
    let title: String
    let description: String
    let thumbnails: YoutubeThumbnails
}

struct YoutubeThumbnails: Codable {
    let high: YoutubeThumbnail
}

struct YoutubeThumbnail: Codable {
    let url: String
}

struct YoutubeAPIHandler {
    static func searchYoutube(query: String, maxResults: Int) async throws -> [YoutubeVideo] {
        guard let url = URL(string: "https://youtubeapi.thetechtitans.vip/youtube_search") else {
            throw URLError(.badURL)
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = [
            "q": query,
            "max_results": maxResults
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse, (200..<300).contains(httpResponse.statusCode) else {
            throw URLError(.badServerResponse)
        }
        let decoded = try JSONDecoder().decode(YoutubeSearchResponse.self, from: data)
        return decoded.items.compactMap { item in
            guard let videoId = item.id.videoId else { return nil }
            return YoutubeVideo(
                id: videoId,
                title: item.snippet.title,
                description: item.snippet.description,
                thumbnailURL: item.snippet.thumbnails.high.url
            )
        }
    }
}
