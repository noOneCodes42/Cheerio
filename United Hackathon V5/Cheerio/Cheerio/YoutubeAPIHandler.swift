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
    let duration: Double? // duration in seconds
}

struct YoutubeSearchResponse: Codable {
    let items: [YoutubeAPIItem]
}

struct YoutubeAPIItem: Codable {
    let id: YoutubeID
    let snippet: YoutubeSnippet
    let contentDetails: YoutubeContentDetails?
}

struct YoutubeContentDetails: Codable {
    let duration: String
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
        guard let url = URL(string: "https://thetechtitans.vip/youtube_search") else {
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
        
        func parseISO8601Duration(_ isoDuration: String) -> Double? {
            // Example: "PT1H2M10S" -> seconds
            var duration = 0.0
            var value = ""
            var lastChar: Character?
            for char in isoDuration {
                if char.isNumber {
                    value.append(char)
                } else {
                    if let last = lastChar, !value.isEmpty {
                        let doubleVal = Double(value) ?? 0
                        switch last {
                        case "H":
                            duration += doubleVal * 3600
                        case "M":
                            duration += doubleVal * 60
                        case "S":
                            duration += doubleVal
                        default:
                            break
                        }
                        value = ""
                    }
                    lastChar = char
                }
            }
            // Catch last value if string ends with number (unlikely for ISO8601 duration)
            if let last = lastChar, !value.isEmpty {
                let doubleVal = Double(value) ?? 0
                switch last {
                case "H":
                    duration += doubleVal * 3600
                case "M":
                    duration += doubleVal * 60
                case "S":
                    duration += doubleVal
                default:
                    break
                }
            }
            return duration > 0 ? duration : nil
        }
        
        return decoded.items.compactMap { item in
            guard let videoId = item.id.videoId else { return nil }
            let duration: Double?
            if let isoDuration = item.contentDetails?.duration {
                duration = parseISO8601Duration(isoDuration)
            } else {
                duration = nil
            }
            return YoutubeVideo(
                id: videoId,
                title: item.snippet.title,
                description: item.snippet.description,
                thumbnailURL: item.snippet.thumbnails.high.url,
                duration: duration
            )
        }
    }
}

