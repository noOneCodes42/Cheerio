import SwiftUI

struct SearchBar: View {
    @Binding var searchText: String
    var onSearch: (() -> Void)? = nil
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        HStack {
            TextField("Search", text: $searchText)
                .padding(7)
                .foregroundColor(colorScheme == .light ? .black : .white)
                .padding(.horizontal, 25)
                .background(Color(.systemGray6))
                .cornerRadius(8)
                .overlay(
                    HStack {
                        Image(systemName: "magnifyingglass")
                            .foregroundColor(.gray)
                            .frame(minWidth: 0, maxWidth: .infinity, alignment: .leading)
                            .padding(.leading, 8)

                        Button(action: {
                            onSearch?()
                            #if canImport(UIKit)
                            UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                            #endif
                        }) {
                            Image(systemName: "magnifyingglass")
                                .foregroundColor(.gray)
                                .padding(.trailing, 8)
                        }
                    }
                )
                .padding(.horizontal, 10)
                .onSubmit {
                    onSearch?()
                    #if canImport(UIKit)
                    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                    #endif
                }
        }
    }

    init(searchText: Binding<String>, onSearch: (() -> Void)? = nil) {
        self._searchText = searchText
        self.onSearch = onSearch
    }
}

