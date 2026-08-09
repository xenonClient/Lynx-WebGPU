import Foundation
import WebGPU
import LynxWebGPUCore

// webgpu.h C API ↔ Swift bridging tools.
//
// A C callback cannot capture context (every `WGPU*CallbackInfo` is a function pointer plus
// `userdata1/2`), so a box is passed across as a retained pointer and reclaimed inside the callback.

/// A result box filled by a single callback. `done` is the polling loop's exit condition.
final class DawnBox<Value> {
    var value: Value?
    var message = ""
    var done = false
}

// MARK: - Safe integer conversion
//
// An Int sent by JS may be negative or enormous. A direct conversion like `UInt32(value)` is a
// **Swift runtime trap (process death)** when out of range — squarely against this library's contract
// (`WGPUError`) that "bad arguments never kill the process". Every place that moves a value into a
// GPU argument width goes through the helpers below and **rejects with a validation error**.

func dawnU32(_ value: Int, _ field: String) throws -> UInt32 {
    guard let converted = UInt32(exactly: value) else {
        throw WGPUError.validation("\(field) value \(value) is out of u32 range")
    }
    return converted
}

func dawnU64(_ value: Int, _ field: String) throws -> UInt64 {
    guard value >= 0 else {
        throw WGPUError.validation("\(field) value \(value) is negative")
    }
    return UInt64(value)
}

func dawnI32(_ value: Int, _ field: String) throws -> Int32 {
    guard let converted = Int32(exactly: value) else {
        throw WGPUError.validation("\(field) value \(value) is out of i32 range")
    }
    return converted
}

func dawnU16(_ value: Int, _ field: String) throws -> UInt16 {
    guard let converted = UInt16(exactly: value) else {
        throw WGPUError.validation("\(field) value \(value) is out of u16 range")
    }
    return converted
}

/// Moves a CGSize (layout- or JS-derived) into a texture size — it filters out NaN, negative, zero and over-cap.
/// `UInt32(CGFloat.nan)` is a trap too.
func dawnTextureDimensions(_ size: CGSize, _ what: String) throws -> (width: UInt32, height: UInt32) {
    guard size.width.isFinite, size.height.isFinite,
          size.width >= 1, size.height >= 1,
          size.width <= 16384, size.height <= 16384 else {
        throw WGPUError.validation("\(what) size is not valid: \(Int(size.width))×\(Int(size.height))")
    }
    return (UInt32(size.width), UInt32(size.height))
}

extension WGPUStringView {
    /// `WGPU_STRLEN` (SIZE_MAX) — the "no explicit length" sentinel. C's `size_t` arrives in Swift as
    /// `Int`, so it has to be built from the **bit pattern** — passing `Int.max` makes Dawn try to build
    /// a string of length 2^63 and abort with `length_error` (experienced first hand).
    static var noLength: Int { Int(bitPattern: UInt.max) }
}

extension String {
    /// `WGPUStringView` → String. If `data` is nil the result is the empty string;
    /// if the length is `WGPU_STRLEN` (the SIZE_MAX sentinel) it is read as a nul-terminated string.
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

/// An arena holding on to the memory C will reference while a descriptor is being built.
///
/// Dawn **copies descriptors during the call** (the webgpu.h contract — it does not reference them after
/// returning), so an arena's lifetime only needs to wrap a single C call. It exists to build nested arrays
/// (bind group entries, color targets, vertex attributes …) without nesting `withUnsafe…`.
final class DawnArena {
    private var allocations: [UnsafeMutableRawPointer] = []

    deinit {
        for allocation in allocations { allocation.deallocate() }
    }

    /// String → `WGPUStringView`. nil is the "no string" sentinel (`{nil, WGPU_STRLEN}`) —
    /// nil differing from the empty string matters in places like automatic entryPoint selection.
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

    /// A value array → a C array pointer. An empty array returns nil (paired with a count of 0).
    func array<T>(_ values: [T]) -> UnsafePointer<T>? {
        guard !values.isEmpty else { return nil }
        let buffer = UnsafeMutableBufferPointer<T>.allocate(capacity: values.count)
        _ = buffer.initialize(from: values)
        guard let base = buffer.baseAddress else { return nil }
        allocations.append(UnsafeMutableRawPointer(base))
        return UnsafePointer(base)
    }

    /// A single value → a pointer (for optional sub-descriptor slots).
    func value<T>(_ value: T) -> UnsafePointer<T> {
        let pointer = UnsafeMutablePointer<T>.allocate(capacity: 1)
        pointer.initialize(to: value)
        allocations.append(UnsafeMutableRawPointer(pointer))
        return UnsafePointer(pointer)
    }
}

/// A bootstrap failure — not a GPU error but "Dawn cannot be brought up".
struct DawnBootstrapError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

/// Instance → adapter → device, brought up synchronously.
///
/// The request APIs are all asynchronous (futures), but setting the callback mode to
/// `AllowProcessEvents` and running `wgpuInstanceProcessEvents` lets completion arrive on the same
/// thread — the same pump the runtime's `processEvents()` uses.
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
            throw DawnBootstrapError("adapter request failed — \(box.message)")
        }
        return adapter
    }

    /// Of the features the adapter supports, **those translatable to the spec spelling and safe in this environment**.
    ///
    /// Advertisement (adapterInfo) and request (requestDevice) must use **the same list** — advertising
    /// without requesting makes the features JS sees disagree with the actual device (the same trap as limits).
    /// On the simulator, indirect draw is a path that dies on a Metal assertion, so `indirect-first-instance`
    /// is dropped (`CLAUDE.md` — the same judgement as the Metal runtime).
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

    /// - Parameter onUncapturedError: device errors not caught by a scope (to be sent to the deferred report queue).
    static func requestDevice(
        instance: WGPUInstance,
        adapter: WGPUAdapter,
        onUncapturedError: @escaping (WGPUErrorType, String) -> Void
    ) throws -> WGPUDevice {
        // The uncaptured callback can be called for the device's whole lifetime, so the box is **held forever**
        // (passRetained with no reclamation — it lives with the device).
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
            // In the test fixture it is only recorded — Dawn harmlessly ignores calls after a loss.
            print("DawnCheck: device lost — \(String(wgpu: message))")
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

        // **Request the adapter's limits as they are** — without passing them the device is bound to the
        // spec defaults (e.g. maxUniformBufferBindingSize 64KB), while adapterInfo advertises the adapter's
        // limits, so the limits JS sees disagree with the device's actual ones (threelab really hit this:
        // 256MB advertised, 64KB device → a 256KB uniform binding was rejected). Features are requested along for the same reason.
        var adapterLimits = WGPULimits()
        let haveLimits = wgpuAdapterGetLimits(adapter, &adapterLimits) == WGPUStatus_Success
        var requestedFeatures = supportedFeatures(adapter: adapter)
        // A Dawn-only thread safety feature — not a spec feature, so **it is not in the advertised list** (request only).
        // The runtime's lock is the first line of defence, but this makes Dawn's internal paths (callbacks,
        // queue events) serialize at the device level too. Without it, the lock alone carries it.
        if wgpuAdapterHasFeature(adapter, WGPUFeatureName_ImplicitDeviceSynchronization) != 0 {
            requestedFeatures.append(WGPUFeatureName_ImplicitDeviceSynchronization)
        }
        _ = withUnsafePointer(to: adapterLimits) { limitsPointer in
            requestedFeatures.withUnsafeBufferPointer { featuresPointer in
                descriptor.requiredLimits = haveLimits ? limitsPointer : nil
                descriptor.requiredFeatureCount = requestedFeatures.count
                descriptor.requiredFeatures = featuresPointer.baseAddress
                return withUnsafePointer(to: descriptor) { pointer in
                    wgpuAdapterRequestDevice(adapter, pointer, callbackInfo)
                }
            }
        }
        try pump(instance: instance, until: { box.done }, what: "requestDevice")
        guard let device = box.value else {
            throw DawnBootstrapError("device request failed — \(box.message)")
        }
        return device
    }

    /// Pumps events until completion. The cap is for diagnosis, not deadlock.
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
                throw DawnBootstrapError("\(what) did not complete within \(timeout)s")
            }
            usleep(500)
        }
    }
}
