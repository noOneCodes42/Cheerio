//
//  PopUpView.swift
//  Cheerio
//
//  Created by Neel Arora on 7/12/25.
//

import SwiftUI
struct ToastView: View {
    let message: String
    let success: Bool
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: success ? "checkmark.circle.fill" : "xmark.circle.fill")
                .foregroundColor(success ? .green : .red)
                .font(.title2)
            Text(message)
                .foregroundColor(.white)
                .bold()
        }
        .padding()
        .background(Color.black.opacity(0.85))
        .cornerRadius(12)
        .padding(.horizontal, 40)
        .shadow(radius: 12)
    }
}
