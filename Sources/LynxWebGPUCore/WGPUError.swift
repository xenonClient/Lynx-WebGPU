import Foundation

/// Why a WebGPU call failed.
///
/// WebGPU is a **safe API** — bad arguments return an error instead of killing the process.
/// This library keeps the same contract: every failure raised while interpreting commands is
/// collected as a `WGPUError` and returned to JS as an array; nothing traps natively.
public struct WGPUError: Error, Equatable, CustomStringConvertible {
    /// Classification matching the WebGPU `GPUError` family.
    public enum Kind: String, Sendable {
        /// Bad argument or state (WebGPU `GPUValidationError`). Usually a caller-side bug.
        case validation
        /// Resource creation failed (WebGPU `GPUOutOfMemoryError`).
        case outOfMemory = "out-of-memory"
        /// A WebGPU feature this implementation does not support yet. The request is valid per spec.
        case unsupported
        /// Internal backend error (Metal, shader compilation).
        case backend
    }

    public let kind: Kind
    public let message: String
    /// Position within the command stream (`commands[3].vertex.buffers[0].format`). A debugging clue.
    public let path: String?
    /// Line number in the shader source (1-based). Present on shader errors only.
    ///
    /// It also appears in the message text, but an editor can only jump to that line if the value
    /// is available **as a number** (`getCompilationInfo()`'s `lineNum` passes it straight through).
    public let line: Int?

    public init(kind: Kind, message: String, path: String? = nil, line: Int? = nil) {
        self.kind = kind
        self.message = message
        self.path = path
        self.line = line
    }

    public var description: String {
        guard let path, !path.isEmpty else { return "[\(kind.rawValue)] \(message)" }
        return "[\(kind.rawValue)] \(path): \(message)"
    }

    /// Serialized form returned to JS.
    public var payload: [String: Any] {
        var result: [String: Any] = ["kind": kind.rawValue, "message": message]
        if let path, !path.isEmpty { result["path"] = path }
        if let line { result["line"] = line }
        return result
    }

    public static func validation(_ message: String, path: String? = nil) -> WGPUError {
        WGPUError(kind: .validation, message: message, path: path)
    }

    public static func unsupported(_ message: String, path: String? = nil) -> WGPUError {
        WGPUError(kind: .unsupported, message: message, path: path)
    }

    public static func backend(_ message: String, path: String? = nil) -> WGPUError {
        WGPUError(kind: .backend, message: message, path: path)
    }

    public static func outOfMemory(_ message: String, path: String? = nil) -> WGPUError {
        WGPUError(kind: .outOfMemory, message: message, path: path)
    }
}

/// The filter accepted by `device.pushErrorScope(filter)` (`GPUErrorFilter`).
///
/// The spec defines only three, while this implementation classifies errors four ways. Where the
/// two extras land is decided by **what a JS author would see in a browser**:
///
/// - `unsupported` means "valid per spec, but this implementation doesn't do it yet". Running the
///   same code in a browser, that call is valid, so it either raises nothing or raises a validation
///   error. Hence the `validation` scope catches it — if it didn't, there would be no way to react.
/// - `backend` (Metal or shader compilation failure) is a valid call that failed for system-side
///   reasons, which is exactly the spec's `internal`.
public enum WGPUErrorFilter: String, CaseIterable, Sendable {
    case validation
    case outOfMemory = "out-of-memory"
    case `internal`

    public func captures(_ kind: WGPUError.Kind) -> Bool {
        switch self {
        case .validation: return kind == .validation || kind == .unsupported
        case .outOfMemory: return kind == .outOfMemory
        case .internal: return kind == .backend
        }
    }
}
