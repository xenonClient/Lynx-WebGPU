import Foundation
import LynxWebGPUConformance

// 검증하는 것:
// 1. Core+Conformance product만으로 `WebGPURuntime` 구현이 **컴파일**된다 (StubRuntime).
// 2. 적합성 스위트가 외부 런타임 위에서 **크래시 없이 전 검사를 판정**한다.
//    스텁이라 GPU 검사는 실패가 정상이다 — 여기서 재는 것은 통과율이 아니라
//    "판정이 성립하는가"다. 실제 백엔드(Dawn 등)는 이 자리에서 통과율을 잰다.
let outcomes = WebGPUConformance.run(on: StubRuntime())
let expected = WebGPUConformance.checks.count

print(WebGPUConformance.summary(outcomes))
for outcome in outcomes where outcome.status == .passed {
    print("  ✓ \(outcome.name)")
}
for outcome in outcomes where outcome.status == .failed {
    print("  ✗ \(outcome.name) — \(outcome.detail)")
}

guard outcomes.count == expected else {
    print("검사 수가 어긋난다 — \(outcomes.count)/\(expected). 스위트가 중간에 끊겼다.")
    exit(1)
}
print("외부 주입 검증 통과 — Core+Conformance만 링크해 런타임을 만들고 \(expected)개 검사를 전부 판정했다.")
