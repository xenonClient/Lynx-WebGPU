import XCTest
import LynxWebGPUCore
@testable import LynxWebGPUShader

/// 외부 WGSL 코퍼스 호환성 리포트.
///
/// 직접 쓴 테스트만으로는 "우리가 생각한 문법"만 검증하게 된다. 실제 세상의 셰이더가
/// 얼마나 그대로 통과하는지 재려면 남의 코퍼스가 필요하다 — 대표적으로 공식
/// [webgpu-samples](https://github.com/webgpu/webgpu-samples).
///
/// ```zsh
/// git clone --depth 1 https://github.com/webgpu/webgpu-samples.git /tmp/webgpu-samples
/// LYNXWEBGPU_WGSL_CORPUS=/tmp/webgpu-samples/sample swift test --filter SampleCorpus
/// ```
///
/// 환경변수가 없으면 건너뛴다 (기본 테스트 실행을 외부 저장소에 묶지 않는다).
final class SampleCorpusTests: XCTestCase {
    private enum Outcome {
        case ok
        case noEntryPoints
        /// 호스트가 `constants`로 값을 줘야 컴파일되는 셰이더 — 명세상 정상 동작이다.
        case needsPipelineConstants(String)
        case parseFailure(String)
        case transpileFailure(entryPoint: String, message: String)
        case mslFailure(entryPoint: String, diagnostic: String)
    }

    func test_외부_WGSL_코퍼스_호환성_리포트() throws {
        guard let root = ProcessInfo.processInfo.environment["LYNXWEBGPU_WGSL_CORPUS"] else {
            throw XCTSkip("LYNXWEBGPU_WGSL_CORPUS 미지정 — 외부 코퍼스 리포트를 건너뛴다")
        }

        let files = try shaderFiles(under: URL(fileURLWithPath: root))
        XCTAssertFalse(files.isEmpty, "코퍼스에 .wgsl 파일이 없다: \(root)")

        var results: [(name: String, outcome: Outcome)] = []
        for file in files {
            let name = file.path.replacingOccurrences(of: root, with: "").trimmingCharacters(in: ["/"])
            results.append((name, evaluate(file)))
        }

        report(results, root: root)
    }

    // MARK: - 평가

    private func evaluate(_ file: URL) -> Outcome {
        guard let source = try? String(contentsOf: file, encoding: .utf8) else {
            return .parseFailure("파일을 읽을 수 없다")
        }

        let module: WGSLShaderModule
        do {
            module = try WGSLShaderModule(source: source)
        } catch {
            return .parseFailure(shortMessage(error))
        }

        let entryPoints = module.reflection.entryPoints
        guard !entryPoints.isEmpty else { return .noEntryPoints }

        for entryPoint in entryPoints {
            let msl: String
            do {
                let groups = module.autoBindGroupLayouts(entryPoints: [entryPoint.name])
                let bindings = try WGSLBindingAssigner.assign(groups: groups)
                msl = try module.translateToMSL(entryPoints: [entryPoint.name], bindings: bindings)
            } catch {
                let message = shortMessage(error)
                if message.contains("override") {
                    return .needsPipelineConstants(message)
                }
                return .transpileFailure(entryPoint: entryPoint.name, message: message)
            }
            // LYNXWEBGPU_WGSL_DUMP=<dir> 를 주면 생성된 MSL을 떨군다 (번역 결과를 눈으로 볼 때).
            if let dump = ProcessInfo.processInfo.environment["LYNXWEBGPU_WGSL_DUMP"] {
                let name = "\(file.deletingPathExtension().lastPathComponent).\(entryPoint.name).metal"
                try? msl.write(
                    to: URL(fileURLWithPath: dump).appendingPathComponent(name),
                    atomically: true, encoding: .utf8
                )
            }
            if let result = MetalCompilerHarness.compile(msl), !result.succeeded {
                return .mslFailure(entryPoint: entryPoint.name, diagnostic: firstDiagnostic(result.diagnostics))
            }
        }
        return .ok
    }

    // MARK: - 리포트

    private func report(_ results: [(name: String, outcome: Outcome)], root: String) {
        var ok: [String] = []
        var fragments: [String] = []
        var needsConstants: [String] = []
        var failures: [(name: String, reason: String)] = []

        for result in results {
            switch result.outcome {
            case .ok:
                ok.append(result.name)
            case .noEntryPoints:
                fragments.append(result.name)
            case .needsPipelineConstants:
                needsConstants.append(result.name)
            case .parseFailure(let message):
                failures.append((result.name, "파싱: \(message)"))
            case .transpileFailure(let entryPoint, let message):
                failures.append((result.name, "번역[\(entryPoint)]: \(message)"))
            case .mslFailure(let entryPoint, let diagnostic):
                failures.append((result.name, "MSL[\(entryPoint)]: \(diagnostic)"))
            }
        }

        let translatable = ok.count + failures.count + needsConstants.count
        let rate = translatable > 0 ? Int((Double(ok.count) / Double(translatable)) * 100) : 0

        var lines: [String] = []
        lines.append("")
        lines.append("┌─ WGSL 코퍼스 호환성 리포트 ─────────────────────────────")
        lines.append("│ 코퍼스: \(root)")
        lines.append("│ 파일 \(results.count)개 (번역 대상 \(translatable)개, 조각 파일 \(fragments.count)개)")
        lines.append("│ 그대로 통과: \(ok.count)/\(translatable)  (\(rate)%)")
        lines.append("│ 호스트가 constants를 줘야 하는 것: \(needsConstants.count)건 — \(needsConstants.joined(separator: ", "))")
        lines.append("├─ 실패 \(failures.count)건 ───────────────────────────────")
        for failure in failures.sorted(by: { $0.name < $1.name }) {
            lines.append("│ ✗ \(failure.name)")
            lines.append("│     \(failure.reason)")
        }
        lines.append("└────────────────────────────────────────────────────────")
        print(lines.joined(separator: "\n"))

        // 리포트가 목적이므로 실패해도 테스트를 깨지 않는다 — 수치를 눈으로 보는 것이 결과물이다.
        XCTAssertGreaterThan(ok.count, 0, "코퍼스에서 단 하나도 번역되지 않았다")
    }

    // MARK: - 보조

    private func shaderFiles(under root: URL) throws -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
        ) else { return [] }
        return enumerator.compactMap { $0 as? URL }
            .filter { $0.pathExtension == "wgsl" }
            .sorted { $0.path < $1.path }
    }

    private func shortMessage(_ error: Error) -> String {
        let message = (error as? WGPUError)?.message ?? error.localizedDescription
        return message.split(separator: "\n").first.map(String.init) ?? message
    }

    private func firstDiagnostic(_ diagnostics: String) -> String {
        diagnostics.split(separator: "\n")
            .first { $0.contains("error:") }
            .map { line -> String in
                // "…/shader.metal:12:5: error: …" 에서 파일 경로를 떼어낸다.
                guard let range = line.range(of: "error:") else { return String(line) }
                return String(line[range.lowerBound...])
            } ?? "(진단 없음)"
    }
}
