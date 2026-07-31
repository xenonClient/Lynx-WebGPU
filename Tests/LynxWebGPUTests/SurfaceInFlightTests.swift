import XCTest
import Metal
import QuartzCore
import LynxWebGPUCore
@testable import LynxWebGPU

/// in-flight 프레임 회계 — 드로어블을 실은 커맨드 버퍼의 커밋/완료가 표면에 통지되고,
/// 포화된 표면이 "다음 프레임 준비 안 됨"을 보고하는 계약.
final class SurfaceInFlightTests: XCTestCase {
    private var device: MTLDevice!

    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
        try XCTSkipIf(device == nil, "Metal 디바이스 없음")
    }

    // MARK: - 카운터 계약 (CAMetalLayer 표면)

    func test_in_flight가_3에_닿으면_준비_안됨이_되고_완료가_돌아오면_풀린다() {
        let surface = WGPUMetalLayerSurface(identifier: "s", layer: CAMetalLayer())
        XCTAssertTrue(surface.isReadyForNextFrame)

        surface.noteFrameCommitted()
        surface.noteFrameCommitted()
        XCTAssertTrue(surface.isReadyForNextFrame, "2프레임까지는 받아들인다")
        surface.noteFrameCommitted()
        XCTAssertFalse(surface.isReadyForNextFrame, "드로어블 풀 크기(3)만큼 밀리면 거른다")

        surface.noteFrameCompleted()
        XCTAssertTrue(surface.isReadyForNextFrame)
    }

    func test_완료가_커밋보다_많아져도_음수로_내려가지_않는다() {
        let surface = WGPUMetalLayerSurface(identifier: "s", layer: CAMetalLayer())
        surface.noteFrameCompleted()
        XCTAssertEqual(surface.currentFramesInFlight, 0)
    }

    func test_컨텍스트는_포화된_표면이_하나라도_있으면_준비_안됨이다() throws {
        let context = try LynxWebGPUContext(device: device)
        let offscreen = WGPUOffscreenSurface(identifier: "off", size: CGSize(width: 8, height: 8), device: device)
        let saturated = WGPUMetalLayerSurface(identifier: "sat", layer: CAMetalLayer())
        for _ in 0..<WGPUMetalLayerSurface.maxFramesInFlight { saturated.noteFrameCommitted() }

        context.registerSurface(offscreen)
        XCTAssertTrue(context.isReadyForNextFrame, "오프스크린 표면은 항상 준비 상태다")

        context.registerSurface(saturated)
        XCTAssertFalse(context.isReadyForNextFrame)

        context.unregisterSurface(identifier: "sat")
        XCTAssertTrue(context.isReadyForNextFrame)
    }

    // MARK: - 해석기 배선 (기록용 표면 더블)

    /// 드로어블을 실은 배치가 커밋 1회·완료 1회를 통지하는지 — 같은 표면에서 텍스처를
    /// 여러 번 얻어도 프레임은 하나이므로 통지도 한 번이어야 한다.
    func test_드로어블을_실은_배치는_표면에_커밋과_완료를_한_번씩_알린다() throws {
        let context = try LynxWebGPUContext(device: device)
        let surface = CountingSurface(identifier: "count", size: CGSize(width: 8, height: 8), device: device)
        context.registerSurface(surface)

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

        XCTAssertEqual(surface.committedCount, 1, "표면당 프레임 회계는 한 번이다")
        XCTAssertTrue(waitUntil { surface.completedCount == 1 }, "GPU 완료가 표면으로 돌아와야 한다")
    }

    func test_드로어블이_없는_배치는_표면에_알리지_않는다() throws {
        let context = try LynxWebGPUContext(device: device)
        let surface = CountingSurface(identifier: "idle", size: CGSize(width: 8, height: 8), device: device)
        context.registerSurface(surface)

        let result = context.execute(commands: [
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.copyDst],
            ["op": "writeBuffer", "buffer": 1, "data": [Float](repeating: 0, count: 4).base64],
        ])
        XCTAssertEqual(result["ok"] as? Bool, true, "\(result)")

        XCTAssertTrue(waitUntil { context.stagingPool.pooledBufferCount == 1 }, "배치는 실제로 실행됐다")
        XCTAssertEqual(surface.committedCount, 0)
        XCTAssertEqual(surface.completedCount, 0)
    }

    // MARK: - CAMetalLayer 왕복 (헤드리스에서 드로어블이 나오는 환경에서만)

    func test_CAMetalLayer_표면도_프레임_왕복_후_카운터가_0으로_돌아온다() throws {
        let context = try LynxWebGPUContext(device: device)
        let layer = CAMetalLayer()
        let surface = WGPUMetalLayerSurface(identifier: "layer", layer: layer)
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

        XCTAssertTrue(waitUntil { surface.currentFramesInFlight == 0 }, "완료 후 카운터가 돌아와야 한다")
        XCTAssertTrue(surface.isReadyForNextFrame)
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

/// 통지 횟수를 기록하는 표면 더블 — 오프스크린 표면에 in-flight 훅만 얹는다.
private final class CountingSurface: WGPUSurface {
    let identifier: String
    private let inner: WGPUOffscreenSurface
    private let lock = NSLock()
    private var committed = 0
    private var completed = 0

    init(identifier: String, size: CGSize, device: MTLDevice) {
        self.identifier = identifier
        self.inner = WGPUOffscreenSurface(identifier: identifier, size: size, device: device)
    }

    var pixelSize: CGSize { inner.pixelSize }
    var configuredFormat: WGPUTextureFormat { inner.configuredFormat }
    func configure(_ configuration: WGPUCanvasConfiguration, device: MTLDevice) throws {
        try inner.configure(configuration, device: device)
    }
    func nextDrawable() -> WGPUDrawable? { inner.nextDrawable() }

    func noteFrameCommitted() {
        lock.lock()
        committed += 1
        lock.unlock()
    }

    func noteFrameCompleted() {
        lock.lock()
        completed += 1
        lock.unlock()
    }

    var committedCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return committed
    }

    var completedCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return completed
    }
}
