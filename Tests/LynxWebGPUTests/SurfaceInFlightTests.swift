import XCTest
import Metal
import QuartzCore
@testable import LynxWebGPUCore
@testable import LynxWebGPU

/// in-flight 프레임 회계의 **배선** — 드로어블을 실은 커맨드 버퍼의 커밋/완료가
/// 코디네이터에 통지되는지 본다.
///
/// 회계 규칙 자체(한도·음수 방지·해제)는 GPU 없이 `WGPUFrameCoordinatorTests`가 검증한다.
/// 여기서 볼 것은 Metal 경로가 그 규칙에 **제대로 물려 있는가**뿐이다.
final class SurfaceInFlightTests: XCTestCase {
    private var device: MTLDevice!

    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
        try XCTSkipIf(device == nil, "Metal 디바이스 없음")
    }

    // MARK: - 등록 배선

    func test_스왑체인_표면만_페이싱_대상으로_등록된다() throws {
        let context = try LynxWebGPUContext(device: device)
        context.registerSurface(WGPUOffscreenSurface(
            identifier: "off", size: CGSize(width: 8, height: 8), device: device
        ))
        XCTAssertEqual(context.frameCoordinator.trackedCanvases, [], "오프스크린은 밀릴 일이 없다")

        context.registerSurface(WGPUMetalLayerSurface(identifier: "screen", layer: CAMetalLayer()))
        XCTAssertEqual(context.frameCoordinator.trackedCanvases, ["screen"])
    }

    func test_컨텍스트는_포화된_캔버스가_하나라도_있으면_준비_안됨이다() throws {
        let context = try LynxWebGPUContext(device: device)
        context.registerSurface(WGPUMetalLayerSurface(identifier: "sat", layer: CAMetalLayer()))
        XCTAssertTrue(context.isReadyForNextFrame)

        for _ in 0..<WGPUFrameCoordinator.defaultMaxFramesInFlight {
            context.frameCoordinator.noteCommitted(canvas: "sat")
        }
        XCTAssertFalse(context.isReadyForNextFrame)

        // 해제하면 죽은 카운터가 남지 않는다 — 남으면 화면이 조용히 멈춘다.
        context.unregisterSurface(identifier: "sat")
        XCTAssertTrue(context.isReadyForNextFrame)
    }

    // MARK: - 해석기 배선

    /// 드로어블을 실은 배치가 커밋 1회·완료 1회를 통지하는지 — 같은 표면에서 텍스처를
    /// 여러 번 얻어도 프레임은 하나이므로 통지도 한 번이어야 한다.
    func test_드로어블을_실은_배치는_커밋과_완료를_한_번씩_알린다() throws {
        let coordinator = CountingCoordinator()
        let context = try LynxWebGPUContext(device: device, frameCoordinator: coordinator)
        // 오프스크린 표면을 쓰되 페이싱 대상으로 직접 등록한다 — 헤드리스에서도 드로어블이
        // 반드시 나오게 하면서, 통지 경로는 화면 표면과 같은 것을 지난다.
        context.registerSurface(WGPUOffscreenSurface(
            identifier: "count", size: CGSize(width: 8, height: 8), device: device
        ))
        coordinator.track(canvas: "count")

        let result = context.execute(commands: [
            ["op": "configureCanvas", "canvas": "count", "format": "rgba8unorm"],
            ["op": "getCurrentTexture", "id": 1, "canvas": "count"],
            ["op": "createTextureView", "id": 2, "texture": 1],
            ["op": "getCurrentTexture", "id": 3, "canvas": "count"],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 2, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "endPass"],
        ])
        XCTAssertEqual(result["ok"] as? Bool, true, "\(result)")

        XCTAssertEqual(coordinator.committedCount, 1, "캔버스당 프레임 회계는 한 번이다")
        XCTAssertTrue(waitUntil { coordinator.completedCount == 1 }, "GPU 완료가 돌아와야 한다")
    }

    func test_드로어블이_없는_배치는_알리지_않는다() throws {
        let coordinator = CountingCoordinator()
        let context = try LynxWebGPUContext(device: device, frameCoordinator: coordinator)
        coordinator.track(canvas: "idle")

        let result = context.execute(commands: [
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.copyDst],
            ["op": "writeBuffer", "buffer": 1, "data": [Float](repeating: 0, count: 4).base64],
        ])
        XCTAssertEqual(result["ok"] as? Bool, true, "\(result)")

        XCTAssertTrue(waitUntil { context.stagingPool.pooledBufferCount == 1 }, "배치는 실제로 실행됐다")
        XCTAssertEqual(coordinator.committedCount, 0)
        XCTAssertEqual(coordinator.completedCount, 0)
    }

    /// 프레임 **중간**의 내부 제출(`present: false`)은 프레임이 아니므로 세지 않는다 —
    /// 세면 popErrorScope 한 번에 티커가 막힌다.
    func test_present_false_배치는_회계에_잡히지_않는다() throws {
        let coordinator = CountingCoordinator()
        let context = try LynxWebGPUContext(device: device, frameCoordinator: coordinator)
        context.registerSurface(WGPUOffscreenSurface(
            identifier: "mid", size: CGSize(width: 8, height: 8), device: device
        ))
        coordinator.track(canvas: "mid")

        let result = context.execute([
            "present": false,
            "commands": [
                ["op": "configureCanvas", "canvas": "mid", "format": "rgba8unorm"],
                ["op": "getCurrentTexture", "id": 1, "canvas": "mid"],
                ["op": "createTextureView", "id": 2, "texture": 1],
                ["op": "beginRenderPass", "colorAttachments": [[
                    "view": 2, "loadOp": "clear", "storeOp": "store",
                    "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
                ]]],
                ["op": "endPass"],
            ],
        ])
        XCTAssertEqual(result["ok"] as? Bool, true, "\(result)")
        XCTAssertEqual(coordinator.committedCount, 0, "프레임 중간 제출은 프레임이 아니다")
    }

    // MARK: - CAMetalLayer 왕복 (헤드리스에서 드로어블이 나오는 환경에서만)

    func test_CAMetalLayer_표면도_프레임_왕복_후_카운터가_0으로_돌아온다() throws {
        let context = try LynxWebGPUContext(device: device)
        let surface = WGPUMetalLayerSurface(identifier: "layer", layer: CAMetalLayer())
        surface.updateDrawableSize(CGSize(width: 32, height: 32))
        context.registerSurface(surface)

        let result = context.execute(commands: [
            ["op": "configureCanvas", "canvas": "layer", "format": "bgra8unorm"],
            ["op": "getCurrentTexture", "id": 1, "canvas": "layer"],
            ["op": "createTextureView", "id": 2, "texture": 1],
            ["op": "beginRenderPass", "colorAttachments": [[
                "view": 2, "loadOp": "clear", "storeOp": "store",
                "clearValue": ["r": 0, "g": 0, "b": 0, "a": 1],
            ]]],
            ["op": "endPass"],
        ])
        try XCTSkipIf((result["ok"] as? Bool) != true, "헤드리스에서 드로어블을 얻지 못했다: \(result)")

        XCTAssertTrue(
            waitUntil { context.frameCoordinator.framesInFlight(canvas: "layer") == 0 },
            "완료 후 카운터가 돌아와야 한다"
        )
        XCTAssertTrue(context.isReadyForNextFrame)
    }

    /// 크기가 NaN·무한대·음수로 와도 **프로세스가 죽지 않아야 한다.**
    ///
    /// 크기는 UI 레이아웃에서 온다(`bounds × pixelRatio`) — 측정 전 프레임이나 이상한
    /// pixelRatio가 NaN을 흘리는 순간이 있다. 그대로 내려가면 오프스크린 표면의
    /// `Int(size.width)`가 **Swift 런타임 트랩**으로 프로세스를 죽인다. 검증 오류가 아니라
    /// 즉사라 로그도 남지 않는다.
    func test_이상한_크기의_resize는_무시하고_크래시하지_않는다() throws {
        let context = try LynxWebGPUContext(device: device)
        try context.attachOffscreenCanvas(identifier: "odd", size: CGSize(width: 8, height: 8))

        for bad in [
            CGSize(width: CGFloat.nan, height: -5),
            CGSize(width: 8, height: CGFloat.nan),
            CGSize(width: CGFloat.infinity, height: 8),
            CGSize(width: -16, height: -16),
        ] {
            context.resizeCanvas(identifier: "odd", drawableSize: bad)
        }

        // 표면이 오염되지 않고 원래 크기 그대로 살아 있다.
        let info = context.canvasInfo(identifier: "odd")
        XCTAssertEqual(info["ok"] as? Bool, true, "\(info)")
        XCTAssertEqual(info["width"] as? Int, 8)
        XCTAssertEqual(info["height"] as? Int, 8)

        // 정상 크기는 그대로 반영된다 — 무시가 표면을 잠가 버리지 않았다.
        context.resizeCanvas(identifier: "odd", drawableSize: CGSize(width: 16, height: 16))
        XCTAssertEqual(context.canvasInfo(identifier: "odd")["width"] as? Int, 16)
    }

    /// 붙일 때부터 이상한 크기면 **검증 오류로 거부한다** (여기는 돌려줄 통로가 있다).
    func test_이상한_크기의_오프스크린_부착은_거부된다() throws {
        let context = try LynxWebGPUContext(device: device)
        XCTAssertThrowsError(
            try context.attachOffscreenCanvas(
                identifier: "nan", size: CGSize(width: CGFloat.nan, height: 8)
            )
        )
    }

    private func waitUntil(timeout: TimeInterval = 2, _ condition: () -> Bool) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            Thread.sleep(forTimeInterval: 0.005)
        }
        return condition()
    }
}

/// 통지 횟수를 기록하는 코디네이터 — 해석기가 회계를 **부르는지**만 본다.
private final class CountingCoordinator: WGPUFrameCoordinator {
    private let counterLock = NSLock()
    private var committed = 0
    private var completed = 0

    override func noteCommitted(canvas: String) {
        counterLock.lock()
        committed += 1
        counterLock.unlock()
        super.noteCommitted(canvas: canvas)
    }

    override func noteCompleted(canvas: String) {
        counterLock.lock()
        completed += 1
        counterLock.unlock()
        super.noteCompleted(canvas: canvas)
    }

    var committedCount: Int {
        counterLock.lock()
        defer { counterLock.unlock() }
        return committed
    }

    var completedCount: Int {
        counterLock.lock()
        defer { counterLock.unlock() }
        return completed
    }
}
