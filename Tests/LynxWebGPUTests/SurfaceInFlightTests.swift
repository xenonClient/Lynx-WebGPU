import XCTest
import Metal
import QuartzCore
@testable import LynxWebGPUCore
@testable import LynxWebGPU

/// The **wiring** of in-flight frame accounting — whether commit and completion of a command buffer
/// carrying a drawable are reported to the coordinator.
///
/// The accounting rules themselves (the cap, no-negative, forgetting) are verified with no GPU by
/// `WGPUFrameCoordinatorTests`. What matters here is only whether the Metal path **is wired into them properly**.
final class SurfaceInFlightTests: XCTestCase {
    private var device: MTLDevice!

    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
        try XCTSkipIf(device == nil, "no Metal device")
    }

    // MARK: - Registration wiring

    func test_onlySwapchainSurfacesAreRegisteredForPacing() throws {
        let context = try LynxWebGPUContext(device: device)
        context.registerSurface(WGPUOffscreenSurface(
            identifier: "off", size: CGSize(width: 8, height: 8), device: device
        ))
        XCTAssertEqual(context.frameCoordinator.trackedCanvases, [], "an offscreen surface cannot fall behind")

        context.registerSurface(WGPUMetalLayerSurface(identifier: "screen", layer: CAMetalLayer()))
        XCTAssertEqual(context.frameCoordinator.trackedCanvases, ["screen"])
    }

    func test_theContextIsNotReadyWhenAnyCanvasIsSaturated() throws {
        let context = try LynxWebGPUContext(device: device)
        context.registerSurface(WGPUMetalLayerSurface(identifier: "sat", layer: CAMetalLayer()))
        XCTAssertTrue(context.isReadyForNextFrame)

        for _ in 0..<WGPUFrameCoordinator.defaultMaxFramesInFlight {
            context.frameCoordinator.noteCommitted(canvas: "sat")
        }
        XCTAssertFalse(context.isReadyForNextFrame)

        // Removing it leaves no dead counter behind — one left behind quietly stops the screen.
        context.unregisterSurface(identifier: "sat")
        XCTAssertTrue(context.isReadyForNextFrame)
    }

    // MARK: - Interpreter wiring

    /// Whether a batch carrying a drawable reports one commit and one completion — obtaining the texture
    /// several times from one surface is still one frame, so it must report once.
    func test_aBatchCarryingADrawableNotifiesCommitAndCompletionOnce() throws {
        let coordinator = CountingCoordinator()
        let context = try LynxWebGPUContext(device: device, frameCoordinator: coordinator)
        // Use an offscreen surface but register it for pacing directly — this guarantees a drawable even
        // headless while the notification path goes through the same code as a screen surface.
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

        XCTAssertEqual(coordinator.committedCount, 1, "frame accounting is once per canvas")
        XCTAssertTrue(waitUntil { coordinator.completedCount == 1 }, "the GPU completion must come back")
    }

    func test_aBatchWithNoDrawableNotifiesNothing() throws {
        let coordinator = CountingCoordinator()
        let context = try LynxWebGPUContext(device: device, frameCoordinator: coordinator)
        coordinator.track(canvas: "idle")

        let result = context.execute(commands: [
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.copyDst],
            ["op": "writeBuffer", "buffer": 1, "data": [Float](repeating: 0, count: 4).base64],
        ])
        XCTAssertEqual(result["ok"] as? Bool, true, "\(result)")

        XCTAssertTrue(waitUntil { context.stagingPool.pooledBufferCount == 1 }, "the batch really did run")
        XCTAssertEqual(coordinator.committedCount, 0)
        XCTAssertEqual(coordinator.completedCount, 0)
    }

    /// An internal **mid-frame** submit (`present: false`) is not a frame and is not counted —
    /// counting it would block the ticker on a single popErrorScope.
    func test_aPresentFalseBatchIsNotCountedInTheAccounting() throws {
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
        XCTAssertEqual(coordinator.committedCount, 0, "a mid-frame submit is not a frame")
    }

    // MARK: - CAMetalLayer round trip (only where a drawable comes out headless)

    func test_aCAMetalLayerSurfaceAlsoReturnsToZeroAfterAFrameRoundTrip() throws {
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
        try XCTSkipIf((result["ok"] as? Bool) != true, "could not obtain a drawable headless: \(result)")

        XCTAssertTrue(
            waitUntil { context.frameCoordinator.framesInFlight(canvas: "layer") == 0 },
            "the counter must come back after completion"
        )
        XCTAssertTrue(context.isReadyForNextFrame)
    }

    /// **The process must not die** even when the size arrives as NaN, infinity or negative.
    ///
    /// The size comes from UI layout (`bounds × pixelRatio`) — a frame before measurement or a strange
    /// pixelRatio leaks a NaN at some point. Passed straight down, the offscreen surface's
    /// `Int(size.width)` kills the process with **a Swift runtime trap**. It is instant death rather
    /// than a validation error, so not even a log survives.
    func test_aStrangeResizeIsIgnoredWithoutCrashing() throws {
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

        // The surface is uncorrupted and still at its original size.
        let info = context.canvasInfo(identifier: "odd")
        XCTAssertEqual(info["ok"] as? Bool, true, "\(info)")
        XCTAssertEqual(info["width"] as? Int, 8)
        XCTAssertEqual(info["height"] as? Int, 8)

        // A normal size still applies — ignoring did not lock the surface.
        context.resizeCanvas(identifier: "odd", drawableSize: CGSize(width: 16, height: 16))
        XCTAssertEqual(context.canvasInfo(identifier: "odd")["width"] as? Int, 16)
    }

    /// A strange size at attach time is **rejected as a validation error** (here there is a channel to return one).
    func test_attachingAnOffscreenCanvasWithAStrangeSizeIsRejected() throws {
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

/// A coordinator recording notification counts — it only checks whether the interpreter **calls** the accounting.
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
