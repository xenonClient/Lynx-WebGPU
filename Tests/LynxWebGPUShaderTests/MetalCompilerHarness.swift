import Foundation
import XCTest

/// A verification harness that runs generated MSL **through the real Metal compiler**.
///
/// String comparison (a golden test) alone cannot catch "plausible MSL that does not compile".
/// Every transpiler test must pass this harness too (docs/TESTING.md §3).
enum MetalCompilerHarness {
    struct Result {
        let succeeded: Bool
        let diagnostics: String
    }

    /// Tries compiling with `xcrun -sdk macosx metal -c`. Nil when the toolchain is absent.
    static func compile(_ source: String) -> Result? {
        guard let metal = locateMetalCompiler() else { return nil }

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("lynx-webgpu-msl-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let sourceURL = directory.appendingPathComponent("shader.metal")
        let outputURL = directory.appendingPathComponent("shader.air")
        do {
            try source.write(to: sourceURL, atomically: true, encoding: .utf8)
        } catch {
            return Result(succeeded: false, diagnostics: "failed to write the source: \(error)")
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        process.arguments = metal + ["-c", sourceURL.path, "-o", outputURL.path]
        let pipe = Pipe()
        process.standardError = pipe
        process.standardOutput = pipe

        do {
            try process.run()
        } catch {
            return nil
        }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return Result(
            succeeded: process.terminationStatus == 0,
            diagnostics: String(data: data, encoding: .utf8) ?? ""
        )
    }

    private static func locateMetalCompiler() -> [String]? {
        #if os(macOS)
        return ["-sdk", "macosx", "metal"]
        #else
        return nil
        #endif
    }

    /// When the MSL does not compile, the failure message carries both the source and the diagnostics.
    static func assertCompiles(_ source: String, file: StaticString = #filePath, line: UInt = #line) {
        guard let result = compile(source) else {
            // Skipped silently where the Metal toolchain is absent (a CI container, say).
            return
        }
        XCTAssertTrue(
            result.succeeded,
            "the generated MSL did not pass the Metal compiler:\n\(result.diagnostics)\n--- MSL ---\n\(numbered(source))",
            file: file,
            line: line
        )
    }

    private static func numbered(_ source: String) -> String {
        source.split(separator: "\n", omittingEmptySubsequences: false)
            .enumerated()
            .map { String(format: "%3d| %@", $0.offset + 1, String($0.element)) }
            .joined(separator: "\n")
    }
}
