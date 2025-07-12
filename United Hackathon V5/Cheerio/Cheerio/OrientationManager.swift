// OrientationManager.swift
// Model for device orientation

import SwiftUI
import Combine

final class OrientationManager: ObservableObject {
    @Published var landscapeSide: UIDeviceOrientation = .unknown
    private var cancellable: AnyCancellable?

    init() {
        updateOrientation()
        cancellable = NotificationCenter.default.publisher(for: UIDevice.orientationDidChangeNotification)
            .sink { [weak self] _ in
                self?.updateOrientation()
            }
    }

    func updateOrientation() {
        let orientation = UIDevice.current.orientation
        DispatchQueue.main.async {
            if orientation == .landscapeLeft || orientation == .landscapeRight {
                self.landscapeSide = orientation
            } else {
                self.landscapeSide = .unknown
            }
        }
    }
}
