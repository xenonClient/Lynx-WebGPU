import Foundation
import Metal
import QuartzCore
import LynxWebGPUCore

/// 한 프레임의 표시 대상.
public protocol WGPUDrawable: AnyObject {
    var texture: MTLTexture { get }
    /// 커맨드 버퍼가 커밋될 때 화면에 올린다. 오프스크린 표면은 아무것도 하지 않는다.
    func present(with commandBuffer: MTLCommandBuffer)
}

/// `GPUCanvasContext`가 그리는 표면 — 화면(CAMetalLayer) 또는 오프스크린 텍스처.
///
/// 호스트가 `LynxWebGPUContext.registerSurface(_:)`로 등록하고, JS는 `canvas` 문자열 id로 지목한다.
public protocol WGPUSurface: AnyObject {
    /// JS가 `context.configure({ canvas: "main" })`에서 쓰는 id.
    var identifier: String { get }
    /// 픽셀 단위 크기. JS가 뷰포트·투영행렬을 계산할 때 돌려준다.
    var pixelSize: CGSize { get }
    var configuredFormat: WGPUTextureFormat { get }

    func configure(_ configuration: WGPUCanvasConfiguration, device: MTLDevice) throws
    /// 이번 프레임에 그릴 드로어블. 표면이 아직 크기를 못 받았거나 드로어블이 고갈되면 nil.
    func nextDrawable() -> WGPUDrawable?

    // MARK: in-flight 프레임 회계

    /// 새 프레임을 받아들일 수 있는가. GPU가 in-flight 한도만큼 밀려 있으면 false —
    /// 프레임 티커가 이 값을 보고 틱을 건너뛰어, JS가 `nextDrawable()` 블로킹에 걸리지 않게 한다.
    var isReadyForNextFrame: Bool { get }
    /// 이 표면의 드로어블을 실은 커맨드 버퍼가 커밋될 때 해석기가 부른다.
    func noteFrameCommitted()
    /// 그 커맨드 버퍼가 GPU에서 완료될 때 해석기가 부른다 (임의 스레드).
    func noteFrameCompleted()
}

public extension WGPUSurface {
    // 오프스크린처럼 스왑체인이 없는 표면은 밀릴 일이 없다 — 기본은 항상 준비 상태.
    var isReadyForNextFrame: Bool { true }
    func noteFrameCommitted() {}
    func noteFrameCompleted() {}
}

// MARK: - 화면 표면 (CAMetalLayer)

final class WGPUMetalLayerDrawable: WGPUDrawable {
    private let drawable: CAMetalDrawable
    var texture: MTLTexture { drawable.texture }

    init(_ drawable: CAMetalDrawable) { self.drawable = drawable }

    func present(with commandBuffer: MTLCommandBuffer) {
        commandBuffer.present(drawable)
    }
}

/// `CAMetalLayer` 기반 표면. `<webgpu-canvas>` 엘리먼트가 만들어 등록한다.
///
/// 스레딩: 레이어 프로퍼티 설정(`drawableSize`, `pixelFormat`)은 **메인 스레드**에서,
/// `nextDrawable()`은 커맨드를 해석하는 **JS 스레드**에서 일어난다. 크기는 메인 스레드가
/// 갱신한 값을 락으로 감싼 캐시에서 읽어, 렌더 스레드가 레이어 프로퍼티를 건드리지 않게 한다.
public final class WGPUMetalLayerSurface: WGPUSurface {
    /// 동시에 GPU에 걸어 둘 프레임 수 상한. `CAMetalLayer`의 드로어블 풀 크기(기본 3)와 같다 —
    /// 이보다 밀리면 `nextDrawable()`이 최대 1초까지 JS 스레드를 세우므로, 그 전에 프레임을 거른다.
    public static let maxFramesInFlight = 3

    public let identifier: String
    public let layer: CAMetalLayer

    private var cachedSize: CGSize = .zero
    private var format: WGPUTextureFormat = .bgra8unorm
    private var framesInFlight = 0
    private let lock = NSLock()

    public init(identifier: String, layer: CAMetalLayer) {
        self.identifier = identifier
        self.layer = layer
    }

    public var pixelSize: CGSize {
        lock.lock()
        defer { lock.unlock() }
        return cachedSize
    }

    public var configuredFormat: WGPUTextureFormat {
        lock.lock()
        defer { lock.unlock() }
        return format
    }

    /// 메인 스레드에서 레이아웃이 바뀔 때 호출한다.
    public func updateDrawableSize(_ size: CGSize) {
        layer.drawableSize = size
        lock.lock()
        cachedSize = size
        lock.unlock()
    }

    public func configure(_ configuration: WGPUCanvasConfiguration, device: MTLDevice) throws {
        let pixelFormat = try WGPUMetalMapping.pixelFormat(configuration.format)
        lock.lock()
        format = configuration.format
        let size = layer.drawableSize
        cachedSize = size
        lock.unlock()

        // CAMetalLayer 설정은 메인 스레드 전용이다. JS 스레드에서 `main.sync`를 걸면
        // 메인이 Lynx 런타임을 기다리는 순간 교착이 나므로 **비동기**로 넘긴다.
        // 그래서 요청한 포맷은 다음 프레임부터 적용된다 — `getCurrentTexture`는 캐시된 설정이 아니라
        // 실제 드로어블 텍스처의 포맷을 보고한다 (기본값이 bgra8unorm이라 보통은 첫 프레임부터 일치한다).
        let apply = { [layer] in
            layer.device = device
            layer.pixelFormat = pixelFormat
            layer.isOpaque = configuration.alphaMode == .opaque
            layer.framebufferOnly = !configuration.usage.contains(.copySrc)
        }
        if Thread.isMainThread {
            apply()
        } else {
            DispatchQueue.main.async(execute: apply)
        }
    }

    public func nextDrawable() -> WGPUDrawable? {
        guard pixelSize.width > 0, pixelSize.height > 0 else { return nil }
        guard let drawable = layer.nextDrawable() else { return nil }
        return WGPUMetalLayerDrawable(drawable)
    }

    // MARK: in-flight 프레임 회계

    public var isReadyForNextFrame: Bool {
        lock.lock()
        defer { lock.unlock() }
        return framesInFlight < Self.maxFramesInFlight
    }

    public func noteFrameCommitted() {
        lock.lock()
        framesInFlight += 1
        lock.unlock()
    }

    public func noteFrameCompleted() {
        lock.lock()
        framesInFlight = max(framesInFlight - 1, 0)
        lock.unlock()
    }

    /// 테스트 관찰용.
    var currentFramesInFlight: Int {
        lock.lock()
        defer { lock.unlock() }
        return framesInFlight
    }
}

// MARK: - 오프스크린 표면 (테스트 하네스)

final class WGPUOffscreenDrawable: WGPUDrawable {
    let texture: MTLTexture
    init(texture: MTLTexture) { self.texture = texture }
    func present(with commandBuffer: MTLCommandBuffer) {}
}

/// 화면 없이 텍스처에 그리는 표면.
///
/// GPU 결과를 눈이 아니라 **픽셀 값으로 검증**하기 위한 것이다 — 시뮬레이터 스크린샷 대신
/// 렌더 결과를 읽어 단언한다 (`docs/TESTING.md` §4).
public final class WGPUOffscreenSurface: WGPUSurface {
    public let identifier: String
    public private(set) var texture: MTLTexture?
    public private(set) var size: CGSize
    private var format: WGPUTextureFormat = .rgba8unorm
    private let device: MTLDevice

    public init(identifier: String = "offscreen", size: CGSize, device: MTLDevice) {
        self.identifier = identifier
        self.size = size
        self.device = device
    }

    public var pixelSize: CGSize { size }
    public var configuredFormat: WGPUTextureFormat { format }

    public func configure(_ configuration: WGPUCanvasConfiguration, device: MTLDevice) throws {
        format = configuration.format
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: try WGPUMetalMapping.pixelFormat(configuration.format),
            width: Int(size.width),
            height: Int(size.height),
            mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .private
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw WGPUError.outOfMemory("오프스크린 표면 텍스처 생성 실패")
        }
        texture.label = "webgpu.offscreen.\(identifier)"
        self.texture = texture
    }

    public func nextDrawable() -> WGPUDrawable? {
        guard let texture else { return nil }
        return WGPUOffscreenDrawable(texture: texture)
    }

    /// 렌더 결과를 RGBA8 바이트로 읽어 온다. 호출 전에 GPU 작업이 끝나 있어야 한다.
    public func readPixels(queue: MTLCommandQueue) throws -> Data {
        guard let texture else {
            throw WGPUError.validation("표면이 아직 configure 되지 않았다")
        }
        let bytesPerRow = texture.width * 4
        let length = bytesPerRow * texture.height
        guard let staging = device.makeBuffer(length: length, options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw WGPUError.backend("픽셀 읽기용 blit 생성 실패")
        }
        blit.copy(
            from: texture,
            sourceSlice: 0,
            sourceLevel: 0,
            sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
            sourceSize: MTLSize(width: texture.width, height: texture.height, depth: 1),
            to: staging,
            destinationOffset: 0,
            destinationBytesPerRow: bytesPerRow,
            destinationBytesPerImage: length
        )
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return Data(bytes: staging.contents(), count: length)
    }
}
