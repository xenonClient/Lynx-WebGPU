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
    private var storage: [WGPUHandle: AnyObject] = [:]
    private let lock = NSLock()

    public init() {}

    public func insert(_ object: AnyObject, at handle: WGPUHandle) {
        lock.lock()
        defer { lock.unlock() }
        storage[handle] = object
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
