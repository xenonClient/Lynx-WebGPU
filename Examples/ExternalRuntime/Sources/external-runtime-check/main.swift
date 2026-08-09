import Foundation
import LynxWebGPUConformance

// What this verifies:
// 1. A `WebGPURuntime` implementation **compiles** against the Core+Conformance products alone (StubRuntime).
// 2. The conformance suite **judges every check without crashing** on an external runtime.
//    Being a stub, GPU checks failing is normal — what is measured here is not the pass rate but
//    "does the judgement hold up". A real backend (Dawn and the like) measures its pass rate in this slot.
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
    print("the check count disagrees — \(outcomes.count)/\(expected). The suite was cut short.")
    exit(1)
}
print("external injection verified — a runtime was built linking only Core+Conformance and judged all \(expected) checks.")
