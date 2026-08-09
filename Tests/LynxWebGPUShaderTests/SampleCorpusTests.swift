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
        /// 같은 폴더의 조각(`common.wgsl` 류)을 붙이니 통과 — 실제 앱이 하는 것과 같은 조립이다.
        case okComposed(with: String)
        case noEntryPoints
        /// `{OUTPUT_FORMAT}` 같은 자리표시자가 든 **템플릿** — 호스트가 치환하기 전에는 WGSL이 아니다.
        case template(String)
        /// 호스트가 `constants`로 값을 줘야 컴파일되는 셰이더 — 명세상 정상 동작이다.
        case needsPipelineConstants(String)
        case parseFailure(String)
        case transpileFailure(entryPoint: String, message: String)
        case mslFailure(entryPoint: String, diagnostic: String)
    }

    func test_externalWGSLCorpusCompatibilityReport() throws {
        guard let root = ProcessInfo.processInfo.environment["LYNXWEBGPU_WGSL_CORPUS"] else {
            throw XCTSkip("LYNXWEBGPU_WGSL_CORPUS 미지정 — 외부 코퍼스 리포트를 건너뛴다")
        }

        let files = try shaderFiles(under: URL(fileURLWithPath: root))
        XCTAssertFalse(files.isEmpty, "코퍼스에 .wgsl 파일이 없다: \(root)")

        var results: [(name: String, outcome: Outcome)] = []
        for file in files {
            let name = file.path.replacingOccurrences(of: root, with: "").trimmingCharacters(in: ["/"])
            results.append((name, evaluate(file, siblings: files)))
        }

        report(results, root: root)
    }

    // MARK: - 평가

    /// 파일 하나를 평가한다. **실제 앱이 하는 조립까지 따라 해 본다.**
    ///
    /// 코퍼스의 셰이더가 모두 완성된 모듈인 것은 아니다 — `rasterizerWGSL + common.wgsl`처럼
    /// 같은 폴더의 조각을 붙여 쓰는 것이 흔하다. 그것을 그대로 재면 "우리 트랜스파일러가 못 한 것"과
    /// "애초에 완성된 모듈이 아닌 것"이 섞여, 수치가 실제 호환성을 말하지 않게 된다.
    private func evaluate(_ file: URL, siblings: [URL]) -> Outcome {
        guard let source = try? String(contentsOf: file, encoding: .utf8) else {
            return .parseFailure("파일을 읽을 수 없다")
        }
        if let placeholder = templatePlaceholder(in: source) {
            return .template(placeholder)
        }

        let direct = translate(source, dumpingAs: file)
        if case .parseFailure = direct {
            return direct
        }
        if case .ok = direct { return direct }
        if case .noEntryPoints = direct { return direct }
        if case .needsPipelineConstants = direct { return direct }

        // 실패했다면 같은 폴더의 **조각 파일**을 하나씩 붙여 다시 해 본다 (앱이 하는 조립).
        for fragment in fragmentSiblings(of: file, in: siblings) {
            guard let extra = try? String(contentsOf: fragment, encoding: .utf8) else { continue }
            if case .ok = translate(source + "\n" + extra) {
                return .okComposed(with: fragment.lastPathComponent)
            }
        }
        return direct
    }

    /// 소스 하나를 진입점마다 번역하고 Metal로 컴파일한다.
    private func translate(_ source: String, dumpingAs file: URL? = nil) -> Outcome {
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
            if let dump = ProcessInfo.processInfo.environment["LYNXWEBGPU_WGSL_DUMP"], let file {
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
        var composed: [String] = []
        var fragments: [String] = []
        var templates: [String] = []
        var needsConstants: [String] = []
        var failures: [(name: String, reason: String)] = []

        for result in results {
            switch result.outcome {
            case .ok:
                ok.append(result.name)
            case .okComposed(let fragment):
                composed.append("\(result.name) (+\(fragment))")
            case .noEntryPoints:
                fragments.append(result.name)
            case .template(let placeholder):
                templates.append("\(result.name) \(placeholder)")
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

        // 통과율의 분모는 **완성된 모듈**이다. 조각 파일·템플릿은 그대로는 WGSL이 아니므로
        // 여기 넣으면 "우리가 못 한 것"과 "애초에 완성본이 아닌 것"이 섞여 수치가 뜻을 잃는다.
        // 대신 전부 이름까지 찍는다 — 분모에서 뺀 것을 숨기지 않기 위해서다.
        let passed = ok.count + composed.count
        let translatable = passed + failures.count + needsConstants.count
        let rate = translatable > 0 ? Int((Double(passed) / Double(translatable)) * 100) : 0

        var lines: [String] = []
        lines.append("")
        lines.append("┌─ WGSL 코퍼스 호환성 리포트 ─────────────────────────────")
        lines.append("│ 코퍼스: \(root)")
        lines.append("│ 파일 \(results.count)개 → 완성 모듈 \(translatable)개")
        lines.append("│ 통과: \(passed)/\(translatable)  (\(rate)%)  — 그대로 \(ok.count) + 조립 \(composed.count)")
        if !composed.isEmpty {
            lines.append("├─ 같은 폴더 조각을 붙여 통과 \(composed.count)건 ──────────")
            for name in composed.sorted() { lines.append("│ ✓ \(name)") }
        }
        lines.append("├─ 분모에서 뺀 것 ───────────────────────────────")
        lines.append("│ 조각 파일(진입점 없음) \(fragments.count)개: \(fragments.joined(separator: ", "))")
        lines.append("│ 템플릿(호스트 치환 전) \(templates.count)개: \(templates.joined(separator: ", "))")
        lines.append("│ constants를 줘야 하는 것 \(needsConstants.count)개: \(needsConstants.joined(separator: ", "))")
        lines.append("├─ 실패 \(failures.count)건 ───────────────────────────────")
        for failure in failures.sorted(by: { $0.name < $1.name }) {
            lines.append("│ ✗ \(failure.name)")
            lines.append("│     \(failure.reason)")
        }
        lines.append("└────────────────────────────────────────────────────────")
        print(lines.joined(separator: "\n"))

        // 리포트가 목적이므로 실패해도 테스트를 깨지 않는다 — 수치를 눈으로 보는 것이 결과물이다.
        XCTAssertGreaterThan(passed, 0, "코퍼스에서 단 하나도 번역되지 않았다")
    }

    // MARK: - 보조

    /// `texture_storage_2d<{OUTPUT_FORMAT}, write>` 같은 **호스트 치환 자리표시자**.
    ///
    /// 대문자·밑줄만 든 중괄호는 WGSL 어디에도 올 수 없다 (블록은 `{` 뒤에 문장이 온다).
    /// 이런 파일은 치환 전에는 WGSL이 아니므로 "실패"가 아니라 템플릿으로 분류한다.
    private func templatePlaceholder(in source: String) -> String? {
        var current = ""
        var inBraces = false
        for character in source {
            if character == "{" {
                inBraces = true
                current = ""
            } else if character == "}" {
                if inBraces, !current.isEmpty,
                   current.allSatisfy({ $0.isUppercase || $0 == "_" || $0.isNumber }) {
                    return "{\(current)}"
                }
                inBraces = false
            } else if inBraces {
                if character.isNewline { inBraces = false } else { current.append(character) }
            }
        }
        return nil
    }

    /// 같은 폴더에 있는 **조각 파일**들 (진입점이 없는 `.wgsl` — `common.wgsl` 류).
    private func fragmentSiblings(of file: URL, in all: [URL]) -> [URL] {
        let directory = file.deletingLastPathComponent().path
        return all.filter { candidate in
            guard candidate != file, candidate.deletingLastPathComponent().path == directory else {
                return false
            }
            guard let source = try? String(contentsOf: candidate, encoding: .utf8),
                  let module = try? WGSLShaderModule(source: source) else { return false }
            return module.reflection.entryPoints.isEmpty
        }
    }

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
