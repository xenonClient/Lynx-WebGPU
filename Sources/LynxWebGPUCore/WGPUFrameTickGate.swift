import Foundation

/// rAF-style backpressure between the display link and the JS thread.
///
/// ## Why it is needed
///
/// `WGPUFrameCoordinator` closes the tick gate when the **GPU** falls behind. But when the **JS
/// thread** is the slow side — command recording, bridge conversion and native encoding all run
/// there — that gate stays open, and the display link keeps stuffing `webgpu:frame` events into
/// the JS message queue at refresh rate. The queue then grows without bound, and every touch
/// event queues **behind** the stale frame events: input lag builds up to seconds and keeps
/// growing for as long as the scene stays heavy.
///
/// A browser never has this failure mode because `requestAnimationFrame` is a pull model — a
/// callback is scheduled again only after the previous one finished, so at most one frame task is
/// ever pending and input events interleave between frames. This gate reproduces that: a tick is
/// sent only after the previous one has been **acknowledged** (the shim acks at the end of its
/// frame callback), so at most one frame event waits in the JS queue.
///
/// ## The timeout
///
/// An ack can get lost — the frame listener was removed while the event was in flight, the bundle
/// was torn down mid-tick, or an old shim without the ack call is running. Without a failsafe the
/// loop would then stay blocked forever, so after `timeout` the pending tick is written off and
/// sending resumes. A legitimately slow frame that overruns the timeout only degrades to one
/// extra queued event per timeout period — still bounded, and the loop cannot die.
///
/// ## Threading
///
/// `shouldSend` comes from the main thread (the display link), `acknowledge` from the JS thread
/// (the bridge ack call), `reset` from wherever the frame loop is started or stopped. All of them
/// are wrapped in the lock.
public final class WGPUFrameTickGate {
    /// Long enough that no ordinary frame trips it by accident, short enough that a lost ack
    /// does not read as a frozen page.
    public static let defaultTimeout: TimeInterval = 1.0

    private let timeout: TimeInterval
    /// Timestamp of the tick that went out and has not been acknowledged. `nil` means the JS side
    /// owes nothing and the next tick may go.
    private var pendingSince: TimeInterval?
    private var skippedTicks = 0
    private var expiredTicks = 0
    private let lock = NSLock()

    public init(timeout: TimeInterval = WGPUFrameTickGate.defaultTimeout) {
        self.timeout = timeout
    }

    /// Whether the tick at `timestamp` may be sent to JS. Saying yes **records the debt** — the
    /// next tick is refused until `acknowledge()` (or the timeout) clears it.
    ///
    /// - Parameter timestamp: the display link timestamp (monotonic seconds). Time comes in as a
    ///   parameter rather than being read here, so the decision is reproducible in tests.
    public func shouldSend(at timestamp: TimeInterval) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if let since = pendingSince {
            if timestamp - since < timeout {
                skippedTicks += 1
                return false
            }
            expiredTicks += 1
        }
        pendingSince = timestamp
        return true
    }

    /// The JS side finished handling the frame event. An ack with no pending tick (a stale ack
    /// arriving after the timeout wrote its tick off) is a no-op.
    public func acknowledge() {
        lock.lock()
        pendingSince = nil
        lock.unlock()
    }

    /// Forget the pending tick — the frame loop was started, stopped, or the page detached.
    /// The listener that owes the ack may not exist anymore, so waiting for it would stall the
    /// first tick of the next loop by a full timeout.
    public func reset() {
        lock.lock()
        pendingSince = nil
        lock.unlock()
    }

    // MARK: - Diagnostics

    /// Ticks refused because the previous one was still unacknowledged. A high number is not a
    /// bug — it is the gate doing its job while frames cost more than a refresh period.
    public var skippedTickCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return skippedTicks
    }

    /// Pending ticks written off by the timeout. Nonzero means acks are getting lost — a torn
    /// down listener, or a shim without the ack call.
    public var expiredTickCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return expiredTicks
    }
}
