import Foundation

/// What one batch must do at the frame boundary.
///
/// The spec expires the current texture not at `submit()` but at **"updating the rendering"**, i.e.
/// the end of the frame. So a single frame may be split across several batches, and only the client
/// (the shim) knows which batch ends the frame — it says so with the `present` flag
/// (`docs/COMMAND-STREAM.md` §1).
///
/// This rule is **backend-independent** — Metal or Dawn, presenting at the same moment is what
/// makes the picture match a browser. So it is factored into one value both sides read.
public struct WGPUFrameBoundary: Equatable {
    /// Whether to send the acquired drawable to screen and expire frame-scoped handles.
    public let presents: Bool
    /// Whether this batch **only presents**, with no commands (the tail of a frame-loop callback).
    ///
    /// Such a batch must put the drawable out even without a command buffer — pass over it and the
    /// screen freezes with nothing said. **Narrowing the condition to "no commands" matters**: a
    /// batch that has commands but produced no command buffer (it only acquired a drawable, say)
    /// would otherwise present a frame that was never drawn.
    public let closesFrame: Bool

    /// - Parameter requestedPresent: `execute({present})`. False means an internal submit from the
    ///   **middle** of a frame — a batch the shim flushed to collect a `popErrorScope`/`mapAsync`
    ///   result.
    public init(requestedPresent: Bool, commandCount: Int) {
        presents = requestedPresent
        closesFrame = requestedPresent && commandCount == 0
    }
}

/// Per-canvas in-flight frame accounting.
///
/// ## Why it is needed
///
/// When the GPU falls behind, the swapchain's drawable pool drains and acquiring a drawable
/// **stalls the entire JS thread for up to a second** — the worst kind of backpressure, freezing not
/// just the canvas but that page's touch handlers, timers and network callbacks with it.
///
/// So we count committed frames, and when a canvas hits the limit **the frame ticker skips the tick
/// entirely.** JS never wakes up and therefore cannot block; once the GPU returns a completion,
/// things resume from the next tick. On screen it looks like a dropped frame — better than queuing
/// frames up and drawing them in a burst.
///
/// ## Why it lives outside the backend
///
/// **Dawn does not do this for you.** `wgpuSurfacePresent` offers no way to ask "would presenting
/// block right now", so whichever backend you use you must count submissions and completions
/// yourself. Hence it sits here, knowing nothing about GPU types — a new runtime only has to call
/// `noteCommitted(canvas:)` on commit and `noteCompleted(canvas:)` from the completion handler.
///
/// ## Threading
///
/// `noteCommitted` comes from the JS thread, `noteCompleted` from a GPU completion handler
/// (arbitrary thread), and `isReadyForNextFrame` from the main thread (the frame ticker). All of
/// them are wrapped in the lock. (It is not `final` because tests subclass it to count
/// notifications — external modules cannot subclass it.)
public class WGPUFrameCoordinator {
    /// Cap on frames queued on the GPU at once. Matches `CAMetalLayer`'s drawable pool size
    /// (3 by default) — fall further behind and acquisition stalls the JS thread, so we drop a
    /// frame before that happens.
    public static let defaultMaxFramesInFlight = 3

    public let maxFramesInFlight: Int

    /// In-flight count per paced canvas. **A canvas absent from here is always ready** — an
    /// offscreen surface has no drawable pool and cannot fall behind.
    private var inFlight: [String: Int] = [:]
    private let lock = NSLock()

    public init(maxFramesInFlight: Int = WGPUFrameCoordinator.defaultMaxFramesInFlight) {
        self.maxFramesInFlight = max(maxFramesInFlight, 1)
    }

    // MARK: - Registration

    /// Starts pacing this canvas (a surface with a swapchain).
    public func track(canvas: String) {
        lock.lock()
        if inFlight[canvas] == nil { inFlight[canvas] = 0 }
        lock.unlock()
    }

    /// The canvas is gone. **Leave it behind and a dead canvas's counter blocks ticks forever.**
    public func forget(canvas: String) {
        lock.lock()
        inFlight.removeValue(forKey: canvas)
        lock.unlock()
    }

    public func reset() {
        lock.lock()
        inFlight.removeAll()
        lock.unlock()
    }

    // MARK: - Accounting

    /// Work carrying this canvas's drawable was submitted to the GPU.
    ///
    /// Does nothing for an untracked canvas — counting it would go unused.
    public func noteCommitted(canvas: String) {
        lock.lock()
        if let count = inFlight[canvas] { inFlight[canvas] = count + 1 }
        lock.unlock()
    }

    /// That work finished on the GPU (arbitrary thread).
    public func noteCompleted(canvas: String) {
        lock.lock()
        if let count = inFlight[canvas] { inFlight[canvas] = max(count - 1, 0) }
        lock.unlock()
    }

    // MARK: - Queries

    public func framesInFlight(canvas: String) -> Int {
        lock.lock()
        defer { lock.unlock() }
        return inFlight[canvas] ?? 0
    }

    public func isReady(canvas: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard let count = inFlight[canvas] else { return true }
        return count < maxFramesInFlight
    }

    /// Whether **every** tracked canvas can take a new frame.
    ///
    /// One saturated canvas skips the whole tick. Filtering per canvas would be possible, but there
    /// is only one frame event for the entire page, so it cannot be split anyway.
    public var isReadyForNextFrame: Bool {
        lock.lock()
        defer { lock.unlock() }
        return inFlight.values.allSatisfy { $0 < maxFramesInFlight }
    }

    /// Canvases currently tracked (tests and diagnostics).
    public var trackedCanvases: [String] {
        lock.lock()
        defer { lock.unlock() }
        return inFlight.keys.sorted()
    }
}
