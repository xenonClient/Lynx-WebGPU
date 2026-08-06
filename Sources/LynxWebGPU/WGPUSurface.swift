import Foundation
import CoreGraphics
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

    /// 드로어블 풀이 있어 **밀릴 수 있는** 표면인가.
    ///
    /// in-flight 회계 자체는 백엔드 밖(`WGPUFrameCoordinator`)에 있다 — Dawn을 쓰든
    /// Metal을 쓰든 같은 정책이 필요하기 때문이다. 표면이 답할 것은 "내가 그 대상인가"뿐이다.
    /// 오프스크린 표면은 풀이 없으므로 false.
    var pacesFrames: Bool { get }
}

public extension WGPUSurface {
    var pacesFrames: Bool { false }
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
    public let identifier: String
    public let layer: CAMetalLayer

    /// 드로어블 풀이 고갈되면 `nextDrawable()`이 JS 스레드를 세운다 — 페이싱 대상이다.
    /// 세는 일은 `WGPUFrameCoordinator`가 한다.
    public var pacesFrames: Bool { true }

    private var cachedSize: CGSize = .zero
    private var format: WGPUTextureFormat = .bgra8unorm
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

            // EDR — `extended`면 1.0을 넘는 값을 SDR 흰색 위쪽 여유 밝기로 그대로 내보낸다.
            // 색공간을 확장 **선형**으로 함께 바꿔야 한다. 둘 중 하나만 걸면 값이 잘리거나
            // 감마가 두 번 먹는다. (셰이더는 sRGB 인코딩 없이 선형 값을 그대로 써야 하고,
            // 포맷도 1.0 초과를 담는 `rgba16float`여야 실제로 밝아진다.)
            let extended = configuration.toneMappingMode == .extended
            layer.wantsExtendedDynamicRangeContent = extended
            let space: CFString
            switch (configuration.colorSpace, extended) {
            case (.srgb, false): space = CGColorSpace.sRGB
            case (.srgb, true): space = CGColorSpace.extendedLinearSRGB
            case (.displayP3, false): space = CGColorSpace.displayP3
            case (.displayP3, true): space = CGColorSpace.extendedLinearDisplayP3
            }
            layer.colorspace = CGColorSpace(name: space)
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

    /// 렌더 결과를 **표면에 설정된 포맷 그대로** 읽어 온다. 호출 전에 GPU 작업이 끝나 있어야 한다.
    ///
    /// 예전에는 `Data`만 돌려주면서 픽셀당 4바이트를 가정했다. 그러면 `rgba16float` 표면에서
    /// 길이도 해석도 틀린 바이트가 **오류 없이** 나오므로, 지금은 포맷·크기·행 간격을 함께 묶은
    /// `WGPUPixelReadback`을 돌려준다. 값 하나는 `readback.rgba(x:y:)`로 꺼낸다.
    ///
    /// - Throws: 아직 `configure`되지 않았거나 표면이 depth/stencil 포맷이면 `WGPUError.validation`.
    ///           depth/stencil은 Metal blit이 aspect 지정 없이 한 덩어리로 복사할 수 없고,
    ///           `depth32float-stencil8`처럼 두 aspect가 섞인 포맷은 픽셀당 바이트 수 자체가
    ///           연속된 한 블록으로 존재하지 않는다.
    public func readPixels(queue: MTLCommandQueue) throws -> WGPUPixelReadback {
        guard let texture else {
            throw WGPUError.validation("표면이 아직 configure 되지 않았다")
        }
        guard !format.isDepthOrStencil else {
            throw WGPUError.validation(
                "\(format.rawValue) 표면은 readPixels로 읽을 수 없다 — depth/stencil은 aspect별 복사가 필요하다"
            )
        }
        let bytesPerRow = texture.width * format.bytesPerPixel
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
        return WGPUPixelReadback(
            data: Data(bytes: staging.contents(), count: length),
            format: format,
            width: texture.width,
            height: texture.height,
            bytesPerRow: bytesPerRow
        )
    }
}
