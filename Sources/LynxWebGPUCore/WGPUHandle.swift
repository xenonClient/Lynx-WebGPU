import Foundation

/// GPU 객체를 가리키는 정수 핸들.
///
/// 핸들은 **클라이언트(JS)가 발급한다**. 객체 생성이 네이티브 왕복을 기다리지 않아도 되므로
/// JS는 `createBuffer` → `writeBuffer` → `setVertexBuffer`를 한 배치에 이어서 기록할 수 있다
/// (Dawn wire와 같은 모델 — `docs/ARCHITECTURE.md` §3).
public struct WGPUHandle: Hashable, CustomStringConvertible, Sendable {
    public let rawValue: Int

    public init(_ rawValue: Int) { self.rawValue = rawValue }

    public var description: String { "#\(rawValue)" }
}

/// 핸들 → GPU 객체 매핑.
///
/// 커맨드 해석은 JS 스레드에서, 리소스 해제·캔버스 리사이즈는 메인 스레드에서 일어날 수 있으므로
/// 모든 접근을 락으로 감싼다. 락 구간에서는 딕셔너리 조작만 하고 GPU 작업은 하지 않는다.
public final class WGPUObjectRegistry {
    /// 이 개수를 넘으면 경고를 남기기 시작한다 (이후 두 배가 될 때마다 반복).
    /// 핸들은 정수라 JS GC가 네이티브 수명을 모른다 — 매 프레임 createView/createBindGroup을
    /// 만들고 destroy를 빼먹는 흔한 웹 관용구가 여기서 무한히 쌓인다.
    static let growthWarningFloor = 4096

    private var storage: [WGPUHandle: AnyObject] = [:]
    private var warnedThreshold = 0
    private let lock = NSLock()

    public init() {}

    public func insert(_ object: AnyObject, at handle: WGPUHandle) {
        lock.lock()
        storage[handle] = object
        var crossed: Int?
        let threshold = warnedThreshold == 0 ? Self.growthWarningFloor : warnedThreshold * 2
        if storage.count >= threshold {
            warnedThreshold = threshold
            crossed = threshold
        }
        lock.unlock()

        if let crossed {
            WGPULog.registry.warning(
                """
                live GPU 객체가 \(crossed)개를 넘었다 — destroy() 누락 가능성. \
                매 프레임 createView/createBindGroup을 만들고 있다면 초기화 때 한 번만 만들거나 \
                프레임 끝에 destroy() 할 것 (JS GC는 정수 핸들의 네이티브 수명을 모른다).
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
    }

    /// 테스트 관찰용 — 마지막으로 경고를 남긴 임계값 (0이면 아직 없음).
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

    /// 핸들을 기대한 타입으로 되찾는다. 없거나 타입이 다르면 validation 오류.
    public func lookup<T>(_ handle: WGPUHandle, as type: T.Type, kind: String, path: String? = nil) throws -> T {
        lock.lock()
        let object = storage[handle]
        lock.unlock()

        guard let object else {
            throw WGPUError.validation("\(kind) \(handle) 이 존재하지 않는다 (이미 destroy 되었거나 생성되지 않음)", path: path)
        }
        guard let typed = object as? T else {
            throw WGPUError.validation("\(handle) 은 \(kind) 가 아니다 (실제: \(Swift.type(of: object)))", path: path)
        }
        return typed
    }
}
