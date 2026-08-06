import Foundation
import LynxWebGPUCore

/// 커맨드 스트림 계약의 **적합성 스위트**.
///
/// ## 왜 라이브러리인가 (테스트 타깃이 아니라)
///
/// 이 스위트가 판정하는 것은 "이 저장소의 구현이 맞는가"가 아니라 **"이 런타임이 계약을
/// 지키는가"**다. 그래서 XCTest에 묶지 않았다 — SPM의 테스트 타깃은 다른 패키지가 가져다
/// 쓸 수 없기 때문이다. 라이브러리로 두면 이 저장소 밖에서 만든 런타임
/// (예: [Dawn](https://github.com/google/dawn) 위에 얹은 구현)이 **같은 스위트를 그대로**
/// 돌려 자신을 증명할 수 있다:
///
/// ```swift
/// let outcomes = WebGPUConformance.run(on: try DawnWebGPURuntime())
/// for outcome in outcomes where outcome.status == .failed {
///     print(outcome.name, outcome.detail)
/// }
/// ```
///
/// 검사는 전부 **커맨드 스트림 · `readCanvasPixels` · `readBuffer` · `adapterInfo`**만 쓴다.
/// 백엔드 내부를 들여다보는 검사는 여기 들어올 수 없다 — 그러면 다른 런타임에서 못 돈다.
/// Metal 내부(인자 테이블 배정, 스테이징 풀, 드로어블 회계)는 `Tests/LynxWebGPUTests`의
/// 해당 파일들이 따로 본다.
///
/// 계약 자체는 `docs/COMMAND-STREAM.md`에 있다.
public enum WebGPUConformance {

    /// 검사 하나.
    public struct Check {
        /// 커맨드 스트림 계약에서 이 검사가 보는 자리.
        public let name: String
        /// 이 기능을 광고하지 않는 어댑터에서는 건너뛴다 (`adapter.features`의 이름).
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
        /// 실패·건너뜀의 이유. 통과면 빈 문자열이다.
        public let detail: String
    }

    /// 런타임 하나를 스위트 전체에 걸어 본다.
    ///
    /// **검사마다 런타임을 초기화한다** (`reset()` + 캔버스 재부착) — 앞 검사가 남긴 객체나
    /// 열린 오류 스코프가 다음 검사의 결과를 바꾸지 않게 한다.
    ///
    /// - Parameter names: 주면 그 이름의 검사만 돈다 (디버깅용).
    public static func run(on runtime: WebGPURuntime, only names: Set<String>? = nil) -> [Outcome] {
        checks
            .filter { names?.contains($0.name) ?? true }
            .map { check in
                do {
                    let harness = try ConformanceHarness(runtime: runtime)
                    if let feature = check.requiresFeature, !harness.advertises(feature: feature) {
                        return Outcome(
                            name: check.name, status: .skipped,
                            detail: "어댑터가 '\(feature)'을(를) 광고하지 않는다"
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

    /// 한 줄 요약 — 로그·CI 출력용.
    public static func summary(_ outcomes: [Outcome]) -> String {
        let passed = outcomes.filter { $0.status == .passed }.count
        let skipped = outcomes.filter { $0.status == .skipped }.count
        let failed = outcomes.filter { $0.status == .failed }.count
        return "적합성 \(passed)/\(outcomes.count) 통과"
            + (skipped > 0 ? " · \(skipped) 건너뜀" : "")
            + (failed > 0 ? " · \(failed) 실패" : "")
    }
}
