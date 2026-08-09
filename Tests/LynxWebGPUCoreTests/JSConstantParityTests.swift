import XCTest
@testable import LynxWebGPUCore

/// Checks that the JS client's constants (`JS/webgpu.js`) and the Swift `OptionSet` values agree.
///
/// The two meet **only as numbers** across the command stream — fixing one side still compiles and
/// still passes tests, while the wrong usage flags apply at runtime. This guards that silent drift.
final class JSConstantParityTests: XCTestCase {
    /// The repository root's `JS/webgpu.js`. Skipped where the source distribution is absent.
    private func loadShim() throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // LynxWebGPUCoreTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // repository root
        let url = root.appendingPathComponent("JS/webgpu.js")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("could not find JS/webgpu.js (\(url.path))")
        }
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// Extracts the value from a `NAME: 0x0020,` form.
    private func constant(_ name: String, in source: String) -> Int? {
        let pattern = "\\b\(name)\\s*:\\s*(0x[0-9A-Fa-f]+|\\d+)"
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(
                in: source, range: NSRange(source.startIndex..., in: source)
              ),
              let range = Range(match.range(at: 1), in: source) else { return nil }
        let text = String(source[range])
        if text.hasPrefix("0x") { return Int(text.dropFirst(2), radix: 16) }
        return Int(text)
    }

    private func assertParity(
        _ name: String,
        _ expected: Int,
        in source: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        guard let actual = constant(name, in: source) else {
            return XCTFail("JS/webgpu.js has no \(name) constant", file: file, line: line)
        }
        XCTAssertEqual(
            actual, expected,
            "\(name): JS=\(actual) Swift=\(expected) — the two sides of the command stream disagree",
            file: file, line: line
        )
    }

    func test_bufferUsageFlagsMatchJS() throws {
        let source = try loadShim()
        assertParity("MAP_READ", WGPUBufferUsage.mapRead.rawValue, in: source)
        assertParity("MAP_WRITE", WGPUBufferUsage.mapWrite.rawValue, in: source)
        assertParity("COPY_SRC", WGPUBufferUsage.copySrc.rawValue, in: source)
        assertParity("COPY_DST", WGPUBufferUsage.copyDst.rawValue, in: source)
        assertParity("INDEX", WGPUBufferUsage.index.rawValue, in: source)
        assertParity("VERTEX", WGPUBufferUsage.vertex.rawValue, in: source)
        assertParity("UNIFORM", WGPUBufferUsage.uniform.rawValue, in: source)
        assertParity("STORAGE", WGPUBufferUsage.storage.rawValue, in: source)
        assertParity("INDIRECT", WGPUBufferUsage.indirect.rawValue, in: source)
        assertParity("QUERY_RESOLVE", WGPUBufferUsage.queryResolve.rawValue, in: source)
    }

    func test_textureUsageFlagsMatchJS() throws {
        // Look only at the GPUTextureUsage block — names such as COPY_SRC collide with the buffer side.
        let source = try loadShim()
        guard let start = source.range(of: "export const GPUTextureUsage = {"),
              let end = source.range(of: "};", range: start.upperBound..<source.endIndex) else {
            return XCTFail("could not find the GPUTextureUsage block")
        }
        let block = String(source[start.upperBound..<end.lowerBound])

        assertParity("COPY_SRC", WGPUTextureUsage.copySrc.rawValue, in: block)
        assertParity("COPY_DST", WGPUTextureUsage.copyDst.rawValue, in: block)
        assertParity("TEXTURE_BINDING", WGPUTextureUsage.textureBinding.rawValue, in: block)
        assertParity("STORAGE_BINDING", WGPUTextureUsage.storageBinding.rawValue, in: block)
        assertParity("RENDER_ATTACHMENT", WGPUTextureUsage.renderAttachment.rawValue, in: block)
    }

    func test_shaderStageAndColorMaskMatchJS() throws {
        let source = try loadShim()
        guard let stageStart = source.range(of: "export const GPUShaderStage = {"),
              let stageEnd = source.range(of: "};", range: stageStart.upperBound..<source.endIndex) else {
            return XCTFail("could not find the GPUShaderStage block")
        }
        let stages = String(source[stageStart.upperBound..<stageEnd.lowerBound])
        assertParity("VERTEX", WGPUShaderStage.vertex.rawValue, in: stages)
        assertParity("FRAGMENT", WGPUShaderStage.fragment.rawValue, in: stages)
        assertParity("COMPUTE", WGPUShaderStage.compute.rawValue, in: stages)

        assertParity("RED", WGPUColorWriteMask.red.rawValue, in: source)
        assertParity("GREEN", WGPUColorWriteMask.green.rawValue, in: source)
        assertParity("BLUE", WGPUColorWriteMask.blue.rawValue, in: source)
        assertParity("ALPHA", WGPUColorWriteMask.alpha.rawValue, in: source)
        assertParity("ALL", WGPUColorWriteMask.all.rawValue, in: source)
    }
}
