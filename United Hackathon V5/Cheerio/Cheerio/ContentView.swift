
//  ContentView.swift
//  Cheerio
//
//  Created by Neel Arora on 7/11/25. SUBMIT THIS ONE NOT THE OTHER ONE
//

import SwiftUI
import Combine
import Foundation
import WebKit


struct ContentView: View {
    @Environment(\.colorScheme) private var colorScheme
    @State private var selection: SidebarItem? = .home
    @State private var searchText = ""
    @State private var isSidebarCollapsed = true
    @StateObject private var orientationManager = OrientationManager()
    
    @State private var searchCancellable: AnyCancellable?
    @State private var videos: [YoutubeVideo] = []
    @State private var isLoading = false
    @State private var selectedVideo: YoutubeVideo? = nil
    
    var body: some View {
        Group {
            if orientationManager.landscapeSide == .landscapeRight {
                HStack(spacing: 0) {
                    VStack(alignment: .leading, spacing: 0) {
                        Button(action: {
                            withAnimation(.easeInOut) {
                                isSidebarCollapsed.toggle()
                            }
                        }) {
                            Image(systemName: isSidebarCollapsed ? "sidebar.left" : "sidebar.right")
                                .resizable()
                                .frame(width: 24, height: 24)
                                .foregroundStyle(colorScheme == .light ? .black : .white)
                                .rotationEffect(.degrees(isSidebarCollapsed ? 0 : 0))
                                .padding(.vertical, 24)
                                .padding(.horizontal, isSidebarCollapsed ? 18 : 20)
                                .padding(.top, 50)
                        }
                        .buttonStyle(.plain)
                        SidebarButton(selection: $selection, item: .home, text: "Home", systemImage: "house", collapsed: isSidebarCollapsed)
                        SidebarButton(selection: $selection, item: .download, text: "Download", systemImage: "arrow.down.circle", collapsed: isSidebarCollapsed)
                        Spacer()
                    }
                    .frame(width: isSidebarCollapsed ? 60 : 180)
                    .background(colorScheme == .light ? Color(.systemGray6) : Color.gray.opacity(0.23))
                    .ignoresSafeArea()
                    Divider()
                    ZStack {
                        (colorScheme == .light ? Color.white : Color(.black)).ignoresSafeArea()
                        switch selection ?? .home {
                        case .home:
                            VStack {
                                SearchBar(searchText: $searchText, onSearch: {
                                    Task {
                                        isLoading = true
                                        do {
                                            videos = try await YoutubeAPIHandler.searchYoutube(query: searchText, maxResults: 20)
                                            isLoading = false
                                        } catch {
                                            isLoading = false
                                            print("Search failed: \(error)")
                                        }
                                    }
                                })
                                .frame(maxWidth: 400)
                                .padding(.horizontal, 20)
                                .padding(.top, 0)
                                .background(Color.clear)
                                
                                if isLoading {
                                    ProgressView()
                                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                        .scaleEffect(1.5)
                                        .padding(.vertical, 20)
                                }
                                
                                List(videos) { video in
                                    Button(action: { selectedVideo = video }) {
                                        HStack(alignment: .top, spacing: 10) {
                                            AsyncImage(url: URL(string: video.thumbnailURL)) { image in
                                                image
                                                    .resizable()
                                            } placeholder: {
                                                ProgressView()
                                            }
                                            .frame(width: 100, height: 56)
                                            .cornerRadius(6)
                                            
                                            VStack(alignment: .leading) {
                                                Text(video.title)
                                                    .bold()
                                                Text(video.description)
                                                    .font(.caption)
                                                    .lineLimit(2)
                                            }
                                        }
                                        .padding(.vertical, 4)
                                    }
                                }
                                .listStyle(PlainListStyle())
                                .frame(maxWidth: 400)
                            }
                            .foregroundStyle(.primary)
                            
                        case .download:
                            VStack {
                                Spacer()
                                Image(systemName: "arrow.down.circle.fill")
                                    .resizable()
                                    .foregroundStyle(.gray)
                                    .frame(width: 100, height: 100)
                                    .padding(.trailing, 30)
                                Text("Downloads")
                                    .foregroundStyle(.primary)
                                    .font(.title)
                                    .padding(.trailing, 25)
                                Spacer()
                            }
                        }
                    }
                }
            } else if orientationManager.landscapeSide == .landscapeLeft {
                HStack(spacing: 0) {
                    ZStack {
                        (colorScheme == .light ? Color.white : Color(.black)).ignoresSafeArea()
                        switch selection ?? .home {
                        case .home:
                            VStack {
                                SearchBar(searchText: $searchText, onSearch: {
                                    Task {
                                        isLoading = true
                                        do {
                                            videos = try await YoutubeAPIHandler.searchYoutube(query: searchText, maxResults: 20)
                                            isLoading = false
                                        } catch {
                                            isLoading = false
                                            print("Search failed: \(error)")
                                        }
                                    }
                                })
                                .frame(maxWidth: 400)
                                .padding(.horizontal, 20)
                                .padding(.top, 0)
                                .background(Color.clear)
                                
                                if isLoading {
                                    ProgressView()
                                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                        .scaleEffect(1.5)
                                        .padding(.vertical, 20)
                                }
                                
                                List(videos) { video in
                                    Button(action: { selectedVideo = video }) {
                                        HStack(alignment: .top, spacing: 10) {
                                            AsyncImage(url: URL(string: video.thumbnailURL)) { image in
                                                image
                                                    .resizable()
                                            } placeholder: {
                                                ProgressView()
                                            }
                                            .frame(width: 100, height: 56)
                                            .cornerRadius(6)
                                            
                                            VStack(alignment: .leading) {
                                                Text(video.title)
                                                    .bold()
                                                Text(video.description)
                                                    .font(.caption)
                                                    .lineLimit(2)
                                            }
                                        }
                                        .padding(.vertical, 4)
                                    }
                                }
                                .listStyle(PlainListStyle())
                                .frame(maxWidth: 400)
                            }
                            .foregroundStyle(.primary)
                            
                        case .download:
                            VStack {
                                Spacer()
                                Image(systemName: "arrow.down.circle.fill")
                                    .resizable()
                                    .foregroundStyle(.gray)
                                    .frame(width: 100, height: 100)
                                    .padding(.trailing, 30)
                                Text("Downloads")
                                    .foregroundStyle(.primary)
                                    .font(.title)
                                    .padding(.trailing, 25)
                                Spacer()
                            }
                        }
                    }
                    Divider()
                    VStack(alignment: .leading, spacing: 0) {
                        Button(action: {
                            withAnimation(.easeInOut) {
                                isSidebarCollapsed.toggle()
                            }
                        }) {
                            Image(systemName: isSidebarCollapsed ? "sidebar.left" : "sidebar.right")
                                .resizable()
                                .frame(width: 24, height: 24)
                                .foregroundStyle(colorScheme == .light ? .black : .white)
                                .rotationEffect(.degrees(isSidebarCollapsed ? 0 : 0))
                                .padding(.vertical, 24)
                                .padding(.horizontal, isSidebarCollapsed ? 18 : 20)
                                .padding(.top, 50)
                        }
                        .buttonStyle(.plain)
                        SidebarButton(selection: $selection, item: .home, text: "Home", systemImage: "house", collapsed: isSidebarCollapsed)
                        SidebarButton(selection: $selection, item: .download, text: "Download", systemImage: "arrow.down.circle", collapsed: isSidebarCollapsed)
                        Spacer()
                    }
                    .frame(width: isSidebarCollapsed ? 60 : 180)
                    .background(colorScheme == .light ? Color(.systemGray6) : Color.gray.opacity(0.23))
                    .ignoresSafeArea()
                }
            } else {
                HStack(spacing: 0) {
                    VStack(alignment: .leading, spacing: 0) {
                        Button(action: {
                            withAnimation(.easeInOut) {
                                isSidebarCollapsed.toggle()
                            }
                        }) {
                            Image(systemName: isSidebarCollapsed ? "sidebar.left" : "sidebar.right")
                                .resizable()
                                .frame(width: 24, height: 24)
                                .foregroundStyle(colorScheme == .light ? .black : .white)
                                .rotationEffect(.degrees(isSidebarCollapsed ? 0 : 0))
                                .padding(.vertical, 24)
                                .padding(.horizontal, isSidebarCollapsed ? 18 : 20)
                                .padding(.top, 50)
                        }
                        .buttonStyle(.plain)
                        SidebarButton(selection: $selection, item: .home, text: "Home", systemImage: "house", collapsed: isSidebarCollapsed)
                        SidebarButton(selection: $selection, item: .download, text: "Download", systemImage: "arrow.down.circle", collapsed: isSidebarCollapsed)
                        Spacer()
                    }
                    .frame(width: isSidebarCollapsed ? 60 : 180)
                    .background(colorScheme == .light ? Color(.systemGray6) : Color.gray.opacity(0.23))
                    .ignoresSafeArea()
                    Divider()
                    ZStack {
                        (colorScheme == .light ? Color.white : Color(.black)).ignoresSafeArea()
                        switch selection ?? .home {
                        case .home:
                            VStack {
                                SearchBar(searchText: $searchText, onSearch: {
                                    Task {
                                        isLoading = true
                                        do {
                                            videos = try await YoutubeAPIHandler.searchYoutube(query: searchText, maxResults: 20)
                                            isLoading = false
                                        } catch {
                                            isLoading = false
                                            print("Search failed: \(error)")
                                        }
                                    }
                                })
                                .frame(maxWidth: 400)
                                .padding(.horizontal, 20)
                                .padding(.top, 0)
                                .background(Color.clear)
                                
                                if isLoading {
                                    ProgressView()
                                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                        .scaleEffect(1.5)
                                        .padding(.vertical, 20)
                                }
                                
                                List(videos) { video in
                                    Button(action: { selectedVideo = video }) {
                                        HStack(alignment: .top, spacing: 10) {
                                            AsyncImage(url: URL(string: video.thumbnailURL)) { image in
                                                image
                                                    .resizable()
                                            } placeholder: {
                                                ProgressView()
                                            }
                                            .frame(width: 100, height: 56)
                                            .cornerRadius(6)
                                            
                                            VStack(alignment: .leading) {
                                                Text(video.title)
                                                    .bold()
                                                Text(video.description)
                                                    .font(.caption)
                                                    .lineLimit(2)
                                            }
                                        }
                                        .padding(.vertical, 4)
                                    }
                                }
                                .listStyle(PlainListStyle())
                                .frame(maxWidth: 400)
                            }
                            .foregroundStyle(.primary)
                            
                        case .download:
                            VStack {
                                Spacer()
                                Image(systemName: "arrow.down.circle.fill")
                                    .resizable()
                                    .foregroundStyle(.gray)
                                    .frame(width: 100, height: 100)
                                    .padding(.trailing, 30)
                                Text("Downloads")
                                    .foregroundStyle(.primary)
                                    .font(.title)
                                    .padding(.trailing, 25)
                                Spacer()
                            }
                        }
                    }
                }
            }
        }
        .accentColor(.gray)
        .sheet(item: $selectedVideo) { video in
            VideoPreviewView(video: video)
        }
        .onAppear {
            orientationManager.updateOrientation()
        }
    }
}

#Preview {
    ContentView()
        .preferredColorScheme(.dark)
}
