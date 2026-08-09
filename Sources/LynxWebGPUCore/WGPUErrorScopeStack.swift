import Foundation

/// The three possible outcomes of `popErrorScope`. JS uses this to decide whether to resolve or
/// reject the promise.
public enum WGPUPoppedErrorScope: Equatable {
    /// The scope existed and caught nothing → resolve with `null`.
    case clean
    /// The scope caught an error → resolve with that error.
    case captured(WGPUError)
    /// No matching `push` → **reject** with `OperationError`, as the spec requires.
    /// This failure is not a GPUError, so no error object is created — a GPUError the spec never
    /// defined must not leak into the app's global handler or telemetry.
    case unmatched

    /// Serialized form returned to JS (one element of the batch result's `errorScopes` array).
    public var payload: Any {
        switch self {
        case .clean: return NSNull()
        case .captured(let error): return error.payload
        case .unmatched: return ["rejected": true]
        }
    }
}

/// Stack of open error scopes (`GPUDevice.pushErrorScope`/`popErrorScope`).
///
/// This is **backend-independent wire policy**, which is why it lives here in Core — every runtime
/// has to catch by the same rules and answer in the same shape to pass the conformance check
/// (`error-scope-capture`). All of the rules come from the spec:
///
/// - **Scopes outlive a batch** — in WebGPU an error scope is device state, and any number of
///   `submit`s may fall between `push` and `pop`. A runtime must keep this stack outside batch
///   lifetime.
/// - The **innermost matching** scope catches the error (`WGPUErrorFilter.captures`).
/// - A scope returns **the first error it caught**, and only that one.
/// - A caught error is not included in the batch result's `errors` — otherwise the app's global
///   handler (`device.onError`) would report an error it already agreed to handle.
public struct WGPUErrorScopeStack {
    private var scopes: [(filter: WGPUErrorFilter?, captured: WGPUError?)] = []

    public init() {}

    /// Number of open scopes (innermost last).
    public var depth: Int { scopes.count }

    /// Opens a scope.
    ///
    /// A nil `filter` is a **placeholder that catches nothing** — it exists so the stack depth stays
    /// correct even when filter parsing failed. Skip the push and a later pop takes the enclosing
    /// scope instead, so the app believes it is reading the inner region's result when it is in fact
    /// reading the outer one's.
    public mutating func push(_ filter: WGPUErrorFilter?) {
        scopes.append((filter, nil))
    }

    /// Closes the innermost scope and returns its result. An empty stack yields `.unmatched`.
    public mutating func pop() -> WGPUPoppedErrorScope {
        guard let scope = scopes.popLast() else { return .unmatched }
        return scope.captured.map(WGPUPoppedErrorScope.captured) ?? .clean
    }

    /// Files an error with the innermost matching scope.
    ///
    /// Returns true if it was caught — the caller then keeps that error out of the batch `errors`.
    /// False means the caller reports it in the batch result.
    public mutating func capture(_ error: WGPUError) -> Bool {
        for index in scopes.indices.reversed()
        where scopes[index].filter?.captures(error.kind) == true {
            if scopes[index].captured == nil { scopes[index].captured = error }
            return true
        }
        return false
    }

    /// Discards open scopes along with the device (`GPUDevice.destroy`) — leaving them behind means
    /// the next page's errors get caught by a dead scope and reported nowhere.
    public mutating func discardAll() {
        scopes.removeAll()
    }
}
