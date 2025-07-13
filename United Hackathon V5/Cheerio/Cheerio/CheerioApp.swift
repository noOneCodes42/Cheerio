//
//  CheerioApp.swift
//  Cheerio
//
//  Created by Neel Arora on 7/11/25.
//

import SwiftUI

@main
struct CheerioApp: App {
    @StateObject private var downloadManager = DownloadManager()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(downloadManager)
        }
    }
}
