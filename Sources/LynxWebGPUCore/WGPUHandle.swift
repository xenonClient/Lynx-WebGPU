import Foundation

/// Integer handle pointing at a GPU object.
///
/// Handles are **issued by the client (JS)**. Creating an object never waits for a native round
/// trip, so JS can record `createBuffer` → `writeBuffer` → `setVertexBuffer` back to back in a
/// single batch (the same model as the Dawn wire — `docs/ARCHITECTURE.md` §3).
public struct WGPUHandle: Hashable, CustomStringConvertible, Sendable {
    public let rawValue: Int

    public init(_ rawValue: Int) { self.rawValue = rawValue }

    public var description: String { "#\(rawValue)" }
}

/// Handle → GPU object mapping.
///
/// Command interpretation runs on the JS thread while resource release and canvas resizing can run
/// on the main thread, so every access is wrapped in a lock. Inside the locked section we only
/// touch the dictionary — never GPU work.
public final class WGPUObjectRegistry {
    /// Past this count we start warning (and again on every doubling after that).
    /// Handles are plain integers, so the JS GC knows nothing about native lifetimes — the common
    /// web idiom of building a createView/createBindGroup every frame and forgetting destroy piles
    /// up here without bound.
    static let growthWarningFloor = 4096

    private var storage: [WGPUHandle: AnyObject] = [:]
    private var warnedThreshold = 0
    /// How many times a live handle was overwritten — a sign that handle issuance is broken.
    private var displacedCount = 0
    private let lock = NSLock()

    public init() {}

    public func insert(_ object: AnyObject, at handle: WGPUHandle) {
        lock.lock()
        let displaced = storage[handle]
        if displaced != nil { displacedCount += 1 }
        storage[handle] = object
        var crossed: Int?
        let threshold = warnedThreshold == 0 ? Self.growthWarningFloor : warnedThreshold * 2
        if storage.count >= threshold {
            warnedThreshold = threshold
            crossed = threshold
        }
        lock.unlock()

        // Overwriting a live handle means **handle issuance is broken.** JS mints the handles, so
        // we don't reject it here (hand-written command streams keep their freedom), but letting it
        // pass silently leaves only the symptom — "my buffer became someone else's texture" — with
        // the cause gone.
        if let displaced {
            WGPULog.registry.warning(
                """
                Handle \(handle) was already in use (\(type(of: displaced)) → \(type(of: object))). \
                The previous object disappears here — check whether the handle allocator is reusing \
                numbers (a per-device counter makes the second device start over at 1).
                """
            )
        }

        if let crossed {
            WGPULog.registry.warning(
                """
                Live GPU objects passed \(crossed) — destroy() is probably missing. \
                If you build a createView/createBindGroup every frame, either build it once at \
                setup or destroy() it at the end of the frame (the JS GC does not know the native \
                lifetime behind an integer handle).
                """
            )
        }
    }

    public func contains(_ handle: WGPUHandle) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return storage[handle] != nil
    }

    @discardableResult
    public func remove(_ handle: WGPUHandle) -> AnyObject? {
        lock.lock()
        defer { lock.unlock() }
        return storage.removeValue(forKey: handle)
    }

    public func removeAll() {
        lock.lock()
        defer { lock.unlock() }
        storage.removeAll()
        warnedThreshold = 0
        displacedCount = 0
    }

    /// How many times a live handle was overwritten. Non-zero means **the handle allocator is
    /// reusing numbers** — it never surfaces as an error, only as a swapped object, so without this
    /// counter there is no way to track down the cause.
    public var displacedHandleCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return displacedCount
    }

    /// Test observation hook — the threshold we last warned at (0 means never).
    var lastWarnedThreshold: Int {
        lock.lock()
        defer { lock.unlock() }
        return warnedThreshold
    }

    public var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return storage.count
    }

    /// Recovers a handle as the expected type. Missing or mistyped is a validation error.
    public func lookup<T>(_ handle: WGPUHandle, as type: T.Type, kind: String, path: String? = nil) throws -> T {
        lock.lock()
        let object = storage[handle]
        lock.unlock()

        guard let object else {
            throw WGPUError.validation(
                "\(kind) \(handle) does not exist (already destroyed, or never created)", path: path
            )
        }
        guard let typed = object as? T else {
            throw WGPUError.validation(
                "\(handle) is not a \(kind) (actual: \(Swift.type(of: object)))", path: path
            )
        }
        return typed
    }
}
