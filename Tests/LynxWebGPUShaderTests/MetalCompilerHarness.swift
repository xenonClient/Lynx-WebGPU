import Foundation
import XCTest

/// 생성된 MSL을 **실제 Metal 컴파일러로 통과시키는** 검증 하네스.
///
/// 문자열 비교(골든 테스트)만으로는 "그럴듯하지만 컴파일되지 않는 MSL"을 잡지 못한다.
/// 트랜스파일러 테스트는 항상 이 하네스를 함께 통과해야 한다 (docs/TESTING.md §3).
enum MetalCompilerHarness {
    struct Result {
        let succeeded: Bool
        let diagnostics: String
    }

    /// `xcrun -sdk macosx metal -c` 로 컴파일해 본다. 툴체인이 없으면 nil.
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
            return Result(succeeded: false, diagnostics: "소스 기록 실패: \(error)")
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

    /// MSL이 컴파일되지 않으면 소스와 진단을 함께 실패 메시지로 낸다.
    static func assertCompiles(_ source: String, file: StaticString = #filePath, line: UInt = #line) {
        guard let result = compile(source) else {
            // Metal 툴체인이 없는 환경(CI 컨테이너 등)에서는 조용히 건너뛴다.
            return
        }
        XCTAssertTrue(
            result.succeeded,
            "생성된 MSL이 Metal 컴파일러를 통과하지 못했다:\n\(result.diagnostics)\n--- MSL ---\n\(numbered(source))",
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
