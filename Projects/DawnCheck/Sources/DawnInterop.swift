import Foundation
import WebGPU
import LynxWebGPUCore

// webgpu.h C API ↔ Swift 브리징 도구.
//
// C 콜백은 컨텍스트를 캡처할 수 없으므로 (모든 `WGPU*CallbackInfo`가 함수 포인터 +
// `userdata1/2`) 상자를 retained 포인터로 넘기고 콜백 안에서 되찾는다.

/// 콜백 하나가 채우는 결과 상자. `done`은 폴링 루프의 종료 조건이다.
final class DawnBox<Value> {
    var value: Value?
    var message = ""
    var done = false
}

// MARK: - 안전 정수 변환
//
// JS가 실어 보낸 Int는 음수·거대값일 수 있다. `UInt32(value)` 같은 직변환은 범위를 벗어나면
// **Swift 런타임 트랩(프로세스 종료)**이다 — "잘못된 인자로 프로세스를 죽이지 않는다"는
// 이 라이브러리의 계약(`WGPUError`)에 정면으로 어긋난다. GPU 인자 폭으로 옮기는 모든 자리는
// 아래 헬퍼를 거쳐 **validation 오류로 거부**한다.

func dawnU32(_ value: Int, _ field: String) throws -> UInt32 {
    guard let converted = UInt32(exactly: value) else {
        throw WGPUError.validation("\(field) 값 \(value)이(가) u32 범위를 벗어난다")
    }
    return converted
}

func dawnU64(_ value: Int, _ field: String) throws -> UInt64 {
    guard value >= 0 else {
        throw WGPUError.validation("\(field) 값 \(value)이(가) 음수다")
    }
    return UInt64(value)
}

func dawnI32(_ value: Int, _ field: String) throws -> Int32 {
    guard let converted = Int32(exactly: value) else {
        throw WGPUError.validation("\(field) 값 \(value)이(가) i32 범위를 벗어난다")
    }
    return converted
}

func dawnU16(_ value: Int, _ field: String) throws -> UInt16 {
    guard let converted = UInt16(exactly: value) else {
        throw WGPUError.validation("\(field) 값 \(value)이(가) u16 범위를 벗어난다")
    }
    return converted
}

/// CGSize(레이아웃·JS 유래)를 텍스처 크기로 옮긴다 — NaN·음수·0·상한 초과를 전부 거른다.
/// `UInt32(CGFloat.nan)`도 트랩이다.
func dawnTextureDimensions(_ size: CGSize, _ what: String) throws -> (width: UInt32, height: UInt32) {
    guard size.width.isFinite, size.height.isFinite,
          size.width >= 1, size.height >= 1,
          size.width <= 16384, size.height <= 16384 else {
        throw WGPUError.validation("\(what) 크기가 유효하지 않다: \(Int(size.width))×\(Int(size.height))")
    }
    return (UInt32(size.width), UInt32(size.height))
}

extension WGPUStringView {
    /// `WGPU_STRLEN`(SIZE_MAX) — "명시 길이 없음" 센티널. C의 `size_t`가 Swift에는 `Int`로
    /// 들어오므로 **비트 패턴**으로 만들어야 한다 — `Int.max`를 넣으면 Dawn이 2^63 길이의
    /// 문자열을 만들려다 `length_error`로 abort한다 (실제로 겪었다).
    static var noLength: Int { Int(bitPattern: UInt.max) }
}

extension String {
    /// `WGPUStringView` → String. `data`가 nil이면 빈 문자열,
    /// 길이가 `WGPU_STRLEN`(SIZE_MAX 센티널)이면 nul 종료 문자열로 읽는다.
    init(wgpu view: WGPUStringView) {
        guard let data = view.data else {
            self = ""
            return
        }
        if view.length == WGPUStringView.noLength {
            self = String(cString: data)
        } else {
            let bytes = UnsafeRawPointer(data).assumingMemoryBound(to: UInt8.self)
            self = String(decoding: UnsafeBufferPointer(start: bytes, count: Int(view.length)),
                          as: UTF8.self)
        }
    }
}

/// 디스크립터를 만드는 동안 C가 참조할 메모리를 붙잡아 두는 아레나.
///
/// Dawn은 디스크립터를 **호출 중에 복사**하므로 (webgpu.h 계약 — 반환 후에는 참조하지 않는다)
/// 아레나의 수명은 C 호출 하나를 감싸면 충분하다. 중첩 배열(바인드 그룹 엔트리, 컬러 타깃,
/// 정점 속성 …)을 `withUnsafe…` 중첩 없이 만들기 위한 장치다.
final class DawnArena {
    private var allocations: [UnsafeMutableRawPointer] = []

    deinit {
        for allocation in allocations { allocation.deallocate() }
    }

    /// String → `WGPUStringView`. nil은 "문자열 없음" 센티널 (`{nil, WGPU_STRLEN}`) —
    /// entryPoint 자동 결정처럼 nil과 빈 문자열이 다른 자리에 중요하다.
    func string(_ value: String?) -> WGPUStringView {
        guard let value else { return WGPUStringView(data: nil, length: WGPUStringView.noLength) }
        let utf8 = Array(value.utf8)
        let buffer = UnsafeMutableRawPointer.allocate(
            byteCount: max(utf8.count, 1), alignment: 1
        )
        allocations.append(buffer)
        utf8.withUnsafeBytes { source in
            if let base = source.baseAddress {
                buffer.copyMemory(from: base, byteCount: utf8.count)
            }
        }
        return WGPUStringView(
            data: buffer.assumingMemoryBound(to: CChar.self),
            length: utf8.count
        )
    }

    /// 값 배열 → C 배열 포인터. 빈 배열은 nil을 돌려준다 (count 0과 짝).
    func array<T>(_ values: [T]) -> UnsafePointer<T>? {
        guard !values.isEmpty else { return nil }
        let buffer = UnsafeMutableBufferPointer<T>.allocate(capacity: values.count)
        _ = buffer.initialize(from: values)
        guard let base = buffer.baseAddress else { return nil }
        allocations.append(UnsafeMutableRawPointer(base))
        return UnsafePointer(base)
    }

    /// 값 하나 → 포인터 (옵셔널 서브 디스크립터 자리).
    func value<T>(_ value: T) -> UnsafePointer<T> {
        let pointer = UnsafeMutablePointer<T>.allocate(capacity: 1)
        pointer.initialize(to: value)
        allocations.append(UnsafeMutableRawPointer(pointer))
        return UnsafePointer(pointer)
    }
}

/// 부트스트랩 실패 — GPU 오류가 아니라 "Dawn을 세울 수 없다".
struct DawnBootstrapError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

/// 인스턴스 → 어댑터 → 디바이스 동기 기동.
///
/// 요청 API는 전부 비동기(future)지만 콜백 모드를 `AllowProcessEvents`로 걸고
/// `wgpuInstanceProcessEvents`를 돌리면 같은 스레드에서 완료를 받을 수 있다 —
/// 런타임의 `processEvents()`가 쓰는 것과 같은 펌프다.
enum DawnBootstrap {

    static func requestAdapter(instance: WGPUInstance) throws -> WGPUAdapter {
        let box = DawnBox<WGPUAdapter>()
        var callbackInfo = WGPURequestAdapterCallbackInfo()
        callbackInfo.mode = WGPUCallbackMode_AllowProcessEvents
        callbackInfo.callback = { status, adapter, message, userdata1, _ in
            guard let userdata1 else { return }
            let box = Unmanaged<DawnBox<WGPUAdapter>>.fromOpaque(userdata1).takeRetainedValue()
            if status == WGPURequestAdapterStatus_Success {
                box.value = adapter
            } else {
                box.message = String(wgpu: message)
            }
            box.done = true
        }
        callbackInfo.userdata1 = Unmanaged.passRetained(box).toOpaque()
        _ = wgpuInstanceRequestAdapter(instance, nil, callbackInfo)
        try pump(instance: instance, until: { box.done }, what: "requestAdapter")
        guard let adapter = box.value else {
            throw DawnBootstrapError("어댑터 요청 실패 — \(box.message)")
        }
        return adapter
    }

    /// 어댑터가 지원하는 기능 중 **명세 철자로 옮길 수 있고 이 환경에서 안전한 것**.
    ///
    /// 광고(adapterInfo)와 요청(requestDevice)이 **같은 목록**을 써야 한다 — 광고만 하고
    /// 요청하지 않으면 JS가 본 기능과 디바이스 실제가 어긋난다 (한계와 같은 함정).
    /// 시뮬레이터에서는 간접 드로우가 Metal 단언으로 죽는 경로라 `indirect-first-instance`를
    /// 뺀다 (`CLAUDE.md` — Metal 런타임과 같은 판단).
    static func supportedFeatures(adapter: WGPUAdapter) -> [WGPUFeatureName] {
        var supported = WGPUSupportedFeatures()
        wgpuAdapterGetFeatures(adapter, &supported)
        defer { wgpuSupportedFeaturesFreeMembers(supported) }
        var result: [WGPUFeatureName] = []
        guard let features = supported.features else { return result }
        for index in 0..<supported.featureCount {
            let feature = features[index]
            guard DawnEnum.featureLabel(feature) != nil else { continue }
            #if targetEnvironment(simulator)
            if feature == WGPUFeatureName_IndirectFirstInstance { continue }
            #endif
            result.append(feature)
        }
        return result
    }

    /// - Parameter onUncapturedError: 스코프에 안 잡힌 디바이스 오류 (지연 보고 큐로 보낼 것).
    static func requestDevice(
        instance: WGPUInstance,
        adapter: WGPUAdapter,
        onUncapturedError: @escaping (WGPUErrorType, String) -> Void
    ) throws -> WGPUDevice {
        // 언캡처드 콜백은 디바이스 수명 내내 불릴 수 있으므로 상자를 **영구 보유**한다
        // (passRetained 후 회수하지 않는다 — 디바이스와 함께 산다).
        final class ErrorSink {
            let handler: (WGPUErrorType, String) -> Void
            init(_ handler: @escaping (WGPUErrorType, String) -> Void) { self.handler = handler }
        }
        var descriptor = WGPUDeviceDescriptor()
        var uncaptured = WGPUUncapturedErrorCallbackInfo()
        uncaptured.callback = { _, type, message, userdata1, _ in
            guard let userdata1 else { return }
            let sink = Unmanaged<ErrorSink>.fromOpaque(userdata1).takeUnretainedValue()
            sink.handler(type, String(wgpu: message))
        }
        uncaptured.userdata1 = Unmanaged.passRetained(ErrorSink(onUncapturedError)).toOpaque()
        descriptor.uncapturedErrorCallbackInfo = uncaptured

        var lost = WGPUDeviceLostCallbackInfo()
        lost.mode = WGPUCallbackMode_AllowProcessEvents
        lost.callback = { _, _, message, _, _ in
            // 테스트 픽스처에서는 기록만 한다 — 로스트 이후의 호출은 Dawn이 무해하게 무시한다.
            print("DawnCheck: 디바이스 로스트 — \(String(wgpu: message))")
        }
        descriptor.deviceLostCallbackInfo = lost

        let box = DawnBox<WGPUDevice>()
        var callbackInfo = WGPURequestDeviceCallbackInfo()
        callbackInfo.mode = WGPUCallbackMode_AllowProcessEvents
        callbackInfo.callback = { status, device, message, userdata1, _ in
            guard let userdata1 else { return }
            let box = Unmanaged<DawnBox<WGPUDevice>>.fromOpaque(userdata1).takeRetainedValue()
            if status == WGPURequestDeviceStatus_Success {
                box.value = device
            } else {
                box.message = String(wgpu: message)
            }
            box.done = true
        }
        callbackInfo.userdata1 = Unmanaged.passRetained(box).toOpaque()

        // 어댑터의 한계를 **그대로 요구**한다 — 안 넘기면 디바이스는 명세 기본값(예:
        // maxUniformBufferBindingSize 64KB)에 묶이는데, adapterInfo는 어댑터 한계를 광고하므로
        // JS가 본 한계와 실제 디바이스 한계가 어긋난다 (threelab이 실제로 밟았다: 광고 256MB,
        // 디바이스 64KB → 256KB 유니폼 바인딩이 거부됐다). 기능도 같은 이유로 함께 요구한다.
        var adapterLimits = WGPULimits()
        let haveLimits = wgpuAdapterGetLimits(adapter, &adapterLimits) == WGPUStatus_Success
        let features = supportedFeatures(adapter: adapter)
        _ = withUnsafePointer(to: adapterLimits) { limitsPointer in
            features.withUnsafeBufferPointer { featuresPointer in
                descriptor.requiredLimits = haveLimits ? limitsPointer : nil
                descriptor.requiredFeatureCount = features.count
                descriptor.requiredFeatures = featuresPointer.baseAddress
                return withUnsafePointer(to: descriptor) { pointer in
                    wgpuAdapterRequestDevice(adapter, pointer, callbackInfo)
                }
            }
        }
        try pump(instance: instance, until: { box.done }, what: "requestDevice")
        guard let device = box.value else {
            throw DawnBootstrapError("디바이스 요청 실패 — \(box.message)")
        }
        return device
    }

    /// 완료까지 이벤트를 퍼 올린다. 상한은 교착이 아니라 진단을 위해서다.
    static func pump(
        instance: WGPUInstance,
        until done: () -> Bool,
        what: String,
        timeout: TimeInterval = 10
    ) throws {
        let deadline = Date(timeIntervalSinceNow: timeout)
        while !done() {
            wgpuInstanceProcessEvents(instance)
            if done() { return }
            guard Date() < deadline else {
                throw DawnBootstrapError("\(what)이(가) \(timeout)초 안에 완료되지 않았다")
            }
            usleep(500)
        }
    }
}
