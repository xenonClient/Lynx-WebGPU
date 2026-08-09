import XCTest
import LynxWebGPUCore

/// Frame boundary policy — verified **with no GPU**.
///
/// Why this policy moved outside the backend shows here: the rule avoiding drawable pool exhaustion
/// is the same for Metal and Dawn, and all of it can be checked without a Metal device.
final class WGPUFrameCoordinatorTests: XCTestCase {

    // MARK: - In-flight accounting

    func test_reachingTheLimitBlocksUntilACompletionReturns() {
        let coordinator = WGPUFrameCoordinator()
        coordinator.track(canvas: "main")
        XCTAssertTrue(coordinator.isReadyForNextFrame)

        coordinator.noteCommitted(canvas: "main")
        coordinator.noteCommitted(canvas: "main")
        XCTAssertTrue(coordinator.isReadyForNextFrame, "up to 2 frames are accepted")

        coordinator.noteCommitted(canvas: "main")
        XCTAssertFalse(coordinator.isReadyForNextFrame, "falling behind by the drawable pool size (3) drops the tick")

        coordinator.noteCompleted(canvas: "main")
        XCTAssertTrue(coordinator.isReadyForNextFrame)
    }

    func test_moreCompletionsThanCommitsNeverGoesNegative() {
        let coordinator = WGPUFrameCoordinator()
        coordinator.track(canvas: "main")
        coordinator.noteCompleted(canvas: "main")
        coordinator.noteCompleted(canvas: "main")
        XCTAssertEqual(coordinator.framesInFlight(canvas: "main"), 0)
    }

    /// An untracked canvas (an offscreen surface) cannot fall behind — it is not counted either.
    func test_anUntrackedCanvasIsAlwaysReady() {
        let coordinator = WGPUFrameCoordinator()
        for _ in 0..<10 { coordinator.noteCommitted(canvas: "offscreen") }
        XCTAssertTrue(coordinator.isReady(canvas: "offscreen"))
        XCTAssertTrue(coordinator.isReadyForNextFrame)
        XCTAssertEqual(coordinator.framesInFlight(canvas: "offscreen"), 0)
        XCTAssertEqual(coordinator.trackedCanvases, [])
    }

    func test_oneSaturatedCanvasMakesEverythingNotReady() {
        let coordinator = WGPUFrameCoordinator()
        coordinator.track(canvas: "a")
        coordinator.track(canvas: "b")
        for _ in 0..<WGPUFrameCoordinator.defaultMaxFramesInFlight {
            coordinator.noteCommitted(canvas: "b")
        }
        XCTAssertTrue(coordinator.isReady(canvas: "a"))
        XCTAssertFalse(coordinator.isReady(canvas: "b"))
        XCTAssertFalse(coordinator.isReadyForNextFrame, "there is one frame event per page, so it cannot be split")
    }

    /// **Leaving a dead canvas's counter behind blocks ticks forever.**
    /// The kind of bug where the screen quietly stops and the cause disappears.
    func test_aForgottenCanvasNoLongerBlocksTicks() {
        let coordinator = WGPUFrameCoordinator()
        coordinator.track(canvas: "gone")
        for _ in 0..<WGPUFrameCoordinator.defaultMaxFramesInFlight {
            coordinator.noteCommitted(canvas: "gone")
        }
        XCTAssertFalse(coordinator.isReadyForNextFrame)

        coordinator.forget(canvas: "gone")
        XCTAssertTrue(coordinator.isReadyForNextFrame)
        XCTAssertEqual(coordinator.trackedCanvases, [])
    }

    func test_theLimitCanBeChanged() {
        let coordinator = WGPUFrameCoordinator(maxFramesInFlight: 1)
        coordinator.track(canvas: "main")
        coordinator.noteCommitted(canvas: "main")
        XCTAssertFalse(coordinator.isReadyForNextFrame)
    }

    /// Zero or negative would make a coordinator **no frame can ever pass** — it is raised to 1.
    func test_aLimitAtOrBelowZeroIsRaisedToOne() {
        XCTAssertEqual(WGPUFrameCoordinator(maxFramesInFlight: 0).maxFramesInFlight, 1)
        XCTAssertEqual(WGPUFrameCoordinator(maxFramesInFlight: -3).maxFramesInFlight, 1)
    }

    func test_concurrentNotificationsKeepTheCounterConsistent() {
        let coordinator = WGPUFrameCoordinator(maxFramesInFlight: 1_000_000)
        coordinator.track(canvas: "main")
        DispatchQueue.concurrentPerform(iterations: 500) { _ in
            coordinator.noteCommitted(canvas: "main")
        }
        XCTAssertEqual(coordinator.framesInFlight(canvas: "main"), 500)
        DispatchQueue.concurrentPerform(iterations: 500) { _ in
            coordinator.noteCompleted(canvas: "main")
        }
        XCTAssertEqual(coordinator.framesInFlight(canvas: "main"), 0)
    }

    // MARK: - Frame boundary

    func test_presentFalseIsAMidFrameSubmit() {
        let boundary = WGPUFrameBoundary(requestedPresent: false, commandCount: 5)
        XCTAssertFalse(boundary.presents, "it does not put the drawable out")
        XCTAssertFalse(boundary.closesFrame)
    }

    /// A batch that only presents with no commands = the tick's close. It must put the drawable out even with no command buffer.
    func test_aPresentBatchWithNoCommandsClosesTheTick() {
        let boundary = WGPUFrameBoundary(requestedPresent: true, commandCount: 0)
        XCTAssertTrue(boundary.presents)
        XCTAssertTrue(boundary.closesFrame)
    }

    /// **Narrowing the condition to "no commands" matters** — presenting a batch that has commands but
    /// produced no command buffer (one that only acquired a drawable) ships a frame that was never drawn.
    func test_aPresentBatchWithCommandsIsNotAClosingBatch() {
        let boundary = WGPUFrameBoundary(requestedPresent: true, commandCount: 1)
        XCTAssertTrue(boundary.presents)
        XCTAssertFalse(boundary.closesFrame)
    }

    func test_withPresentFalseNoCommandsStillDoesNotClose() {
        XCTAssertFalse(WGPUFrameBoundary(requestedPresent: false, commandCount: 0).closesFrame)
    }
}
