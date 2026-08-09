import XCTest
import LynxWebGPUCore

/// 프레임 경계 정책 — **GPU 없이** 검증한다.
///
/// 이 정책이 백엔드 밖으로 나온 이유가 여기서 드러난다: 드로어블 풀 고갈을 피하는 규칙은
/// Metal이든 Dawn이든 같고, Metal 디바이스 없이도 전부 확인할 수 있다.
final class WGPUFrameCoordinatorTests: XCTestCase {

    // MARK: - in-flight 회계

    func test_reachingTheLimitBlocksUntilACompletionReturns() {
        let coordinator = WGPUFrameCoordinator()
        coordinator.track(canvas: "main")
        XCTAssertTrue(coordinator.isReadyForNextFrame)

        coordinator.noteCommitted(canvas: "main")
        coordinator.noteCommitted(canvas: "main")
        XCTAssertTrue(coordinator.isReadyForNextFrame, "2프레임까지는 받아들인다")

        coordinator.noteCommitted(canvas: "main")
        XCTAssertFalse(coordinator.isReadyForNextFrame, "드로어블 풀 크기(3)만큼 밀리면 거른다")

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

    /// 추적하지 않는 캔버스(오프스크린 표면)는 밀릴 일이 없다 — 세지도 않는다.
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
        XCTAssertFalse(coordinator.isReadyForNextFrame, "프레임 이벤트는 페이지에 하나뿐이라 나뉘지 않는다")
    }

    /// **죽은 캔버스의 카운터를 남겨 두면 그것이 영원히 틱을 막는다.**
    /// 화면이 조용히 멈추고 원인은 사라지는 종류의 버그다.
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

    /// 0이나 음수를 주면 **아무 프레임도 통과하지 못하는** 코디네이터가 된다 — 1로 올린다.
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

    // MARK: - 프레임 경계

    func test_present_false는_프레임_중간_제출이다() {
        let boundary = WGPUFrameBoundary(requestedPresent: false, commandCount: 5)
        XCTAssertFalse(boundary.presents, "드로어블을 내보내지 않는다")
        XCTAssertFalse(boundary.closesFrame)
    }

    /// 명령 없이 present만 하는 배치 = 틱의 마무리. 커맨드 버퍼가 없어도 드로어블을 내보내야 한다.
    func test_aPresentBatchWithNoCommandsClosesTheTick() {
        let boundary = WGPUFrameBoundary(requestedPresent: true, commandCount: 0)
        XCTAssertTrue(boundary.presents)
        XCTAssertTrue(boundary.closesFrame)
    }

    /// **조건이 "명령이 비었을 때"로 좁혀져 있는 것이 중요하다** — 명령은 있는데 커맨드 버퍼가
    /// 안 생긴 배치(드로어블만 얻고 끝난 경우)까지 내보내면 그리지도 않은 화면이 나간다.
    func test_aPresentBatchWithCommandsIsNotAClosingBatch() {
        let boundary = WGPUFrameBoundary(requestedPresent: true, commandCount: 1)
        XCTAssertTrue(boundary.presents)
        XCTAssertFalse(boundary.closesFrame)
    }

    func test_present가_false면_명령이_없어도_마무리가_아니다() {
        XCTAssertFalse(WGPUFrameBoundary(requestedPresent: false, commandCount: 0).closesFrame)
    }
}
