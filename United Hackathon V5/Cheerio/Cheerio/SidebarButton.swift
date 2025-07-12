// SidebarButton.swift
// SwiftUI View for sidebar navigation button

import SwiftUI

struct SidebarButton: View {
    @Binding var selection: SidebarItem?
    let item: SidebarItem
    let text: String
    let systemImage: String
    let collapsed: Bool

    var body: some View {
        Button(action: {
            selection = item
        }) {
            HStack {
                Image(systemName: systemImage)
                    .foregroundStyle(selection == item ? .white : .gray)
                    .frame(width: 24)
                if !collapsed {
                    Text(text)
                        .foregroundStyle(selection == item ? .white : .gray)
                }
                Spacer()
            }
            .padding(.vertical, 10)
            .padding(.horizontal, collapsed ? 14 : 20)
            .background(selection == item ? Color(.darkGray).opacity(0.7) : Color.clear)
            .cornerRadius(8)
        }
        .buttonStyle(.plain)
    }
}
