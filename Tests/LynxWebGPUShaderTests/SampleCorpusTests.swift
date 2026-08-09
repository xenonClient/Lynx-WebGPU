import XCTest
import LynxWebGPUCore
@testable import LynxWebGPUShader

/// External WGSL corpus compatibility report.
///
/// Hand-written tests alone only verify "the grammar we thought of". Measuring how much real-world
/// shader code passes unchanged needs someone else's corpus — most notably the official
/// [webgpu-samples](https://github.com/webgpu/webgpu-samples).
///
/// ```zsh
/// git clone --depth 1 https://github.com/webgpu/webgpu-samples.git /tmp/webgpu-samples
/// LYNXWEBGPU_WGSL_CORPUS=/tmp/webgpu-samples/sample swift test --filter SampleCorpus
/// ```
///
/// Skipped without the environment variable (the default test run is not tied to an external repository).
final class SampleCorpusTests: XCTestCase {
    private enum Outcome {
        case ok
        /// Passed once same-folder fragments (`common.wgsl` and the like) were appended — the same assembly a real app does.
        case okComposed(with: String)
        case noEntryPoints
        /// A **template** holding placeholders such as `{OUTPUT_FORMAT}` — not WGSL until the host substitutes them.
        case template(String)
        /// A shader that compiles only once the host supplies `constants` — normal behaviour per spec.
        case needsPipelineConstants(String)
        case parseFailure(String)
        case transpileFailure(entryPoint: String, message: String)
        case mslFailure(entryPoint: String, diagnostic: String)
    }

    func test_externalWGSLCorpusCompatibilityReport() throws {
        guard let root = ProcessInfo.processInfo.environment["LYNXWEBGPU_WGSL_CORPUS"] else {
            throw XCTSkip("LYNXWEBGPU_WGSL_CORPUS not set — skipping the external corpus report")
        }

        let files = try shaderFiles(under: URL(fileURLWithPath: root))
        XCTAssertFalse(files.isEmpty, "no .wgsl files in the corpus: \(root)")

        var results: [(name: String, outcome: Outcome)] = []
        for file in files {
            let name = file.path.replacingOccurrences(of: root, with: "").trimmingCharacters(in: ["/"])
            results.append((name, evaluate(file, siblings: files)))
        }

        report(results, root: root)
    }

    // MARK: - Evaluation

    /// Evaluates one file. **It imitates the assembly a real app does, too.**
    ///
    /// Not every shader in the corpus is a complete module — using same-folder fragments, as with
    /// `rasterizerWGSL + common.wgsl`, is common. Measuring that as-is mixes "what our transpiler could
    /// not do" with "what was never a complete module", so the number stops describing real compatibility.
    private func evaluate(_ file: URL, siblings: [URL]) -> Outcome {
        guard let source = try? String(contentsOf: file, encoding: .utf8) else {
            return .parseFailure("could not read the file")
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

        // On failure, append the folder's **fragment files** one at a time and retry (the assembly an app does).
        for fragment in fragmentSiblings(of: file, in: siblings) {
            guard let extra = try? String(contentsOf: fragment, encoding: .utf8) else { continue }
            if case .ok = translate(source + "\n" + extra) {
                return .okComposed(with: fragment.lastPathComponent)
            }
        }
        return direct
    }

    /// Translates one source per entry point and compiles it with Metal.
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
            // Setting LYNXWEBGPU_WGSL_DUMP=<dir> dumps the generated MSL (for eyeballing the translation).
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

    // MARK: - Report

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
                failures.append((result.name, "parse: \(message)"))
            case .transpileFailure(let entryPoint, let message):
                failures.append((result.name, "translate[\(entryPoint)]: \(message)"))
            case .mslFailure(let entryPoint, let diagnostic):
                failures.append((result.name, "MSL[\(entryPoint)]: \(diagnostic)"))
            }
        }

        // The denominator of the pass rate is **complete modules**. Fragments and templates are not WGSL
        // as they stand, so including them mixes "what we could not do" with "what was never complete"
        // and the number loses meaning. Instead every one is named — so nothing removed from the denominator is hidden.
        let passed = ok.count + composed.count
        let translatable = passed + failures.count + needsConstants.count
        let rate = translatable > 0 ? Int((Double(passed) / Double(translatable)) * 100) : 0

        var lines: [String] = []
        lines.append("")
        lines.append("┌─ WGSL corpus compatibility report ─────────────────────")
        lines.append("│ corpus: \(root)")
        lines.append("│ \(results.count) files → \(translatable) complete modules")
        lines.append("│ passed: \(passed)/\(translatable)  (\(rate)%)  — as-is \(ok.count) + assembled \(composed.count)")
        if !composed.isEmpty {
            lines.append("├─ passed after appending same-folder fragments: \(composed.count) ──────")
            for name in composed.sorted() { lines.append("│ ✓ \(name)") }
        }
        lines.append("├─ removed from the denominator ─────────────────")
        lines.append("│ fragment files (no entry point): \(fragments.count) — \(fragments.joined(separator: ", "))")
        lines.append("│ templates (before host substitution): \(templates.count) — \(templates.joined(separator: ", "))")
        lines.append("│ needing constants: \(needsConstants.count) — \(needsConstants.joined(separator: ", "))")
        lines.append("├─ failures: \(failures.count) ───────────────────────────")
        for failure in failures.sorted(by: { $0.name < $1.name }) {
            lines.append("│ ✗ \(failure.name)")
            lines.append("│     \(failure.reason)")
        }
        lines.append("└────────────────────────────────────────────────────────")
        print(lines.joined(separator: "\n"))

        // The report is the point, so a failure does not break the test — seeing the numbers is the deliverable.
        XCTAssertGreaterThan(passed, 0, "not a single file in the corpus translated")
    }

    // MARK: - Helpers

    /// **Host substitution placeholders** such as `texture_storage_2d<{OUTPUT_FORMAT}, write>`.
    ///
    /// Braces holding only uppercase and underscores cannot appear anywhere in WGSL (a block has
    /// statements after `{`). Such a file is not WGSL before substitution, so it is classified as a
    /// template rather than a failure.
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

    /// The **fragment files** in the same folder (`.wgsl` with no entry point — `common.wgsl` and the like).
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
                // Strip the file path from "…/shader.metal:12:5: error: …".
                guard let range = line.range(of: "error:") else { return String(line) }
                return String(line[range.lowerBound...])
            } ?? "(no diagnostics)"
    }
}
