import Foundation

/// Canvas report carried in the batch result's `canvases` — the pixel size of every surface that
/// handed out a drawable this batch.
public struct WGPUCanvasReport: Equatable {
    public let width: Int
    public let height: Int

    public init(width: Int, height: Int) {
        self.width = width
        self.height = height
    }
}

/// Assembles one `execute` batch's response — `{ok, commandCount, objects, errors, canvases,
/// errorScopes}`.
///
/// The key spelling and the omission rules are **wire contract** (`docs/COMMAND-STREAM.md` §2).
/// Rebuilding this shape by hand in each backend goes wrong on a single misspelling, so every
/// runtime assembles it through this type — the conformance check (`error-accumulation`) judges
/// exactly this shape.
public struct WGPUBatchResult {
    public var commandCount: Int
    /// Live native object count — lets JS watch for a missing destroy (registry growth).
    public var liveObjectCount: Int
    /// Errors no scope caught (`WGPUErrorScopeStack.capture` returned false).
    public var errors: [WGPUError]
    public var canvases: [String: WGPUCanvasReport]
    /// Results of scopes popped this batch, in pop order — matched 1:1 with the JS promise order.
    public var poppedScopes: [WGPUPoppedErrorScope]

    public init(
        commandCount: Int,
        liveObjectCount: Int,
        errors: [WGPUError] = [],
        canvases: [String: WGPUCanvasReport] = [:],
        poppedScopes: [WGPUPoppedErrorScope] = []
    ) {
        self.commandCount = commandCount
        self.liveObjectCount = liveObjectCount
        self.errors = errors
        self.canvases = canvases
        self.poppedScopes = poppedScopes
    }

    /// The shape that goes back to JS.
    ///
    /// `ok`/`commandCount`/`objects` are always present; `errors`/`canvases`/`errorScopes` ride
    /// along **only when non-empty** — the omission is part of the contract too.
    public var payload: [String: Any] {
        var result: [String: Any] = [
            "ok": errors.isEmpty,
            "commandCount": commandCount,
            "objects": liveObjectCount,
        ]
        if !errors.isEmpty {
            result["errors"] = errors.map(\.payload)
        }
        if !canvases.isEmpty {
            result["canvases"] = canvases.mapValues { report -> [String: Any] in
                ["width": report.width, "height": report.height]
            }
        }
        if !poppedScopes.isEmpty {
            result["errorScopes"] = poppedScopes.map(\.payload)
        }
        return result
    }

    /// A top-level failure that never made it into the batch — no `commands` array in the payload,
    /// and the like. No command was seen, so batch-result keys such as `commandCount` are omitted.
    public static func failure(_ errors: [WGPUError]) -> [String: Any] {
        ["ok": false, "errors": errors.map(\.payload)]
    }
}
