import Foundation
import LynxWebGPUCore

/// The **conformance suite** for the command stream contract.
///
/// ## Why a library (rather than a test target)
///
/// What this suite judges is not "is this repository's implementation correct?" but **"does this
/// runtime keep the contract?"** So it is not tied to XCTest — an SPM test target cannot be consumed
/// by another package. As a library, a runtime built outside this repository (an implementation on
/// top of [Dawn](https://github.com/google/dawn), say) can run **the very same suite** and prove itself:
///
/// ```swift
/// let outcomes = WebGPUConformance.run(on: try DawnWebGPURuntime())
/// for outcome in outcomes where outcome.status == .failed {
///     print(outcome.name, outcome.detail)
/// }
/// ```
///
/// Every check uses **only the command stream, `readCanvasPixels`, `readBuffer` and `adapterInfo`**.
/// A check that inspects backend internals cannot come in here — it would not run on another runtime.
/// Metal internals (argument table assignment, the staging pool, drawable accounting) are covered
/// separately by the corresponding files in `Tests/LynxWebGPUTests`.
///
/// The contract itself lives in `docs/COMMAND-STREAM.md`.
public enum WebGPUConformance {

    /// One check.
    public struct Check {
    /// The place in the command stream contract this check looks at.
        public let name: String
    /// Skipped on an adapter that does not advertise this feature (a name from `adapter.features`).
        public let requiresFeature: String?
        public let body: (ConformanceHarness) throws -> Void

        public init(_ name: String, requiresFeature: String? = nil,
                    _ body: @escaping (ConformanceHarness) throws -> Void) {
            self.name = name
            self.requiresFeature = requiresFeature
            self.body = body
        }
    }

    public enum Status: String {
        case passed, failed, skipped
    }

    public struct Outcome {
        public let name: String
        public let status: Status
        /// Why it failed or was skipped. An empty string when it passed.
        public let detail: String
    }

    /// Runs one runtime through the entire suite.
    ///
    /// **The runtime is reset for every check** (`reset()` plus reattaching the canvas) — so objects or
    /// open error scopes left by one check cannot change the next one's result.
    ///
    /// - Parameter names: when given, only checks with those names run (for debugging).
    public static func run(on runtime: WebGPURuntime, only names: Set<String>? = nil) -> [Outcome] {
        checks
            .filter { names?.contains($0.name) ?? true }
            .map { check in
                do {
                    let harness = try ConformanceHarness(runtime: runtime)
                    if let feature = check.requiresFeature, !harness.advertises(feature: feature) {
                        return Outcome(
                            name: check.name, status: .skipped,
                            detail: "the adapter does not advertise '\(feature)'"
                        )
                    }
                    try check.body(harness)
                    return Outcome(name: check.name, status: .passed, detail: "")
                } catch let failure as ConformanceFailure {
                    return Outcome(name: check.name, status: .failed, detail: failure.message)
                } catch {
                    return Outcome(name: check.name, status: .failed, detail: "\(error)")
                }
            }
    }

    /// A one-line summary — for logs and CI output.
    public static func summary(_ outcomes: [Outcome]) -> String {
        let passed = outcomes.filter { $0.status == .passed }.count
        let skipped = outcomes.filter { $0.status == .skipped }.count
        let failed = outcomes.filter { $0.status == .failed }.count
        return "conformance \(passed)/\(outcomes.count) passed"
            + (skipped > 0 ? " · \(skipped) skipped" : "")
            + (failed > 0 ? " · \(failed) failed" : "")
    }
}
