import Foundation

/// Deferred reporting queue for asynchronous GPU failures — **filled by the completion handler
/// (arbitrary thread) and drained by the next batch.**
///
/// A batch collects its errors before the commit, so GPU-side failures (out of memory, timeout,
/// device removal) structurally cannot be caught at that point. The completion handler gathers them
/// here instead and they ride out on the **next batch's result** — without this they would surface
/// nowhere at all and stay silent.
///
/// The deferred-delivery model is itself part of the wire contract (`docs/COMMAND-STREAM.md` §2)
/// and every backend needs the same channel, which is why it lives here in Core — under Dawn this
/// is what the uncaptured-error callback fills.
public final class WGPUDeferredErrorQueue {
    private let lock = NSLock()
    private var pending: [WGPUError] = []

    public init() {}

    /// Records a failure. Safe from any thread.
    public func report(_ error: WGPUError) {
        lock.lock()
        pending.append(error)
        lock.unlock()
    }

    /// Takes the accumulated failures and clears them (the caller feeds them into this batch's
    /// error collection).
    public func drain() -> [WGPUError] {
        lock.lock()
        defer { lock.unlock() }
        let failures = pending
        pending.removeAll()
        return failures
    }
}
