import XCTest
import LynxWebGPUCore

/// Tick backpressure policy — verified with no display link and no JS thread.
///
/// The contract under test: at most one `webgpu:frame` event is ever waiting in the JS queue
/// (rAF semantics), and a lost ack degrades to timeout pacing instead of killing the loop.
final class WGPUFrameTickGateTests: XCTestCase {

    func test_theSecondTickWaitsForTheAck() {
        let gate = WGPUFrameTickGate()
        XCTAssertTrue(gate.shouldSend(at: 0))
        XCTAssertFalse(gate.shouldSend(at: 0.016), "the previous tick has not been acknowledged")
        XCTAssertFalse(gate.shouldSend(at: 0.033))
        gate.acknowledge()
        XCTAssertTrue(gate.shouldSend(at: 0.050))
        XCTAssertEqual(gate.skippedTickCount, 2)
    }

    func test_ackingEveryTickNeverSkips() {
        let gate = WGPUFrameTickGate()
        for frame in 0..<10 {
            XCTAssertTrue(gate.shouldSend(at: Double(frame) / 60))
            gate.acknowledge()
        }
        XCTAssertEqual(gate.skippedTickCount, 0)
        XCTAssertEqual(gate.expiredTickCount, 0)
    }

    func test_aLostAckExpiresAtTheTimeoutInsteadOfKillingTheLoop() {
        let gate = WGPUFrameTickGate(timeout: 1.0)
        XCTAssertTrue(gate.shouldSend(at: 0))
        XCTAssertFalse(gate.shouldSend(at: 0.999), "inside the timeout the debt still holds")
        XCTAssertTrue(gate.shouldSend(at: 1.0), "at the timeout the pending tick is written off")
        XCTAssertEqual(gate.expiredTickCount, 1)
    }

    /// A stale ack (for a tick the timeout already wrote off) clears the current debt once —
    /// that lets one extra event into the queue, bounded at that, and accounting recovers.
    func test_aStaleAckOpensTheGateOnlyOnce() {
        let gate = WGPUFrameTickGate(timeout: 1.0)
        XCTAssertTrue(gate.shouldSend(at: 0))
        XCTAssertTrue(gate.shouldSend(at: 1.5), "expired — resent")
        gate.acknowledge()   // the ack the first tick owed, arriving late
        XCTAssertTrue(gate.shouldSend(at: 1.6))
        XCTAssertFalse(gate.shouldSend(at: 1.7), "the debt re-arms after one tick")
    }

    func test_resetForgetsThePendingTick() {
        let gate = WGPUFrameTickGate()
        XCTAssertTrue(gate.shouldSend(at: 0))
        gate.reset()
        XCTAssertTrue(gate.shouldSend(at: 0.016), "a restarted loop must not wait out the old debt")
    }

    func test_anAckWithNothingPendingIsANoOp() {
        let gate = WGPUFrameTickGate()
        gate.acknowledge()
        XCTAssertTrue(gate.shouldSend(at: 0))
        XCTAssertFalse(gate.shouldSend(at: 0.016), "the spurious ack must not pre-open the gate")
    }
}
