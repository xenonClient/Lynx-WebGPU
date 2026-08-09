import Foundation
import os

/// Shared logger for the library.
///
/// Nothing logs from inside the frame loop — at 60fps even a single os_log line adds up until it
/// can no longer be ignored. Use these only on low-frequency paths: creation, teardown, errors.
public enum WGPULog {
    public static let subsystem = "org.lynxwebgpu"

    public static func logger(category: String) -> Logger {
        Logger(subsystem: subsystem, category: category)
    }

    public static let device = logger(category: "Device")
    public static let shader = logger(category: "Shader")
    public static let bridge = logger(category: "Bridge")
    public static let canvas = logger(category: "Canvas")
    public static let registry = logger(category: "Registry")
    /// Rare engine-level events — the self-pump giving up, and the like. Never on the frame path.
    public static let runtime = logger(category: "Runtime")
}
