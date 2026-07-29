import XCTest
@testable import LynxWebGPUCore

/// JS 클라이언트(`JS/webgpu.js`)의 상수와 Swift `OptionSet` 값이 어긋나지 않는지 본다.
///
/// 이 둘은 커맨드 스트림을 사이에 두고 **숫자로만** 만난다 — 한쪽만 고쳐도 컴파일은 되고
/// 테스트도 통과하지만 런타임에 엉뚱한 사용 플래그가 적용된다. 그 조용한 드리프트를 막는 장치다.
final class JSConstantParityTests: XCTestCase {
    /// 저장소 루트의 `JS/webgpu.js`. 소스 배포가 아닌 환경에서는 테스트를 건너뛴다.
    private func loadShim() throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // LynxWebGPUCoreTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // 저장소 루트
        let url = root.appendingPathComponent("JS/webgpu.js")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("JS/webgpu.js 를 찾을 수 없다 (\(url.path))")
        }
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// `NAME: 0x0020,` 형태에서 값을 뽑는다.
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
            return XCTFail("JS/webgpu.js 에 \(name) 상수가 없다", file: file, line: line)
        }
        XCTAssertEqual(
            actual, expected,
            "\(name): JS=\(actual) Swift=\(expected) — 커맨드 스트림 양쪽이 어긋났다",
            file: file, line: line
        )
    }

    func test_버퍼_사용플래그가_JS와_일치한다() throws {
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

    func test_텍스처_사용플래그가_JS와_일치한다() throws {
        // GPUTextureUsage 블록만 잘라서 본다 — COPY_SRC 등 이름이 버퍼 쪽과 겹친다.
        let source = try loadShim()
        guard let start = source.range(of: "export const GPUTextureUsage = {"),
              let end = source.range(of: "};", range: start.upperBound..<source.endIndex) else {
            return XCTFail("GPUTextureUsage 블록을 찾을 수 없다")
        }
        let block = String(source[start.upperBound..<end.lowerBound])

        assertParity("COPY_SRC", WGPUTextureUsage.copySrc.rawValue, in: block)
        assertParity("COPY_DST", WGPUTextureUsage.copyDst.rawValue, in: block)
        assertParity("TEXTURE_BINDING", WGPUTextureUsage.textureBinding.rawValue, in: block)
        assertParity("STORAGE_BINDING", WGPUTextureUsage.storageBinding.rawValue, in: block)
        assertParity("RENDER_ATTACHMENT", WGPUTextureUsage.renderAttachment.rawValue, in: block)
    }

    func test_셰이더스테이지와_컬러마스크가_JS와_일치한다() throws {
        let source = try loadShim()
        guard let stageStart = source.range(of: "export const GPUShaderStage = {"),
              let stageEnd = source.range(of: "};", range: stageStart.upperBound..<source.endIndex) else {
            return XCTFail("GPUShaderStage 블록을 찾을 수 없다")
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
