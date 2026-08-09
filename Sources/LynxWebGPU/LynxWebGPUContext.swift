import Foundation
import CoreGraphics
import Metal
import QuartzCore
import LynxWebGPUCore
import LynxWebGPUShader

/// `WebGPURuntime`의 **Metal 직접 구현** — 이 패키지가 기본으로 주는 것.
///
/// 실체는 `WGPUBackendEngine`(Core — 오케스트레이션) 위에 `WGPUMetalBackend`(인코딩)를
/// 얹은 조합이다. 이 타입은 그 조합을 만들어 묶고, 호스트가 쓰는 Metal 고유 표면
/// (디바이스·큐·커스텀 `WGPUSurface` 등록)을 노출한다. 호스트 앱이 이 객체를 만들어 Lynx
/// 브리지에 넘기거나 (`LynxWebGPUHost(runtime:)`), Lynx 없이 Swift에서 직접 커맨드 스트림을
/// 실행할 수 있다 (테스트 하네스가 그렇게 쓴다).
///
/// 다른 구현으로 갈아끼우는 이야기는 `WebGPURuntime` 문서에, 백엔드만 갈아끼우는 이야기는
/// `WGPUBackend` 문서에 있다.
///
/// **스레딩** — 직렬화는 엔진이 책임진다 (`WGPUBackendEngine` 문서). 표면 등록/해제는
/// 메인/임의 스레드에서 와도 안전하다 (엔진의 등록부 락).
public final class LynxWebGPUContext: WebGPURuntime {
    public let device: MTLDevice
    public let queue: MTLCommandQueue

    private let backend: WGPUMetalBackend
    private let engine: WGPUBackendEngine<WGPUMetalBackend>

    /// in-flight 프레임 회계 — **백엔드와 무관한 정책**이라 Core에 있다
    /// (`WGPUFrameCoordinator` 문서 참고). 호스트가 프레임 티커를 자기 방식으로 몰 때
    /// 들여다볼 수 있도록 공개한다.
    public var frameCoordinator: WGPUFrameCoordinator { engine.frameCoordinator }

    public init(device: MTLDevice? = nil, frameCoordinator: WGPUFrameCoordinator = WGPUFrameCoordinator()) throws {
        guard let resolved = device ?? MTLCreateSystemDefaultDevice() else {
            throw WGPUError.backend("Metal 디바이스를 만들 수 없다 (시뮬레이터/기기 지원 확인)")
        }
        guard let queue = resolved.makeCommandQueue() else {
            throw WGPUError.backend("MTLCommandQueue 생성 실패")
        }
        self.device = resolved
        self.queue = queue
        self.backend = WGPUMetalBackend(device: resolved, queue: queue)
        self.engine = WGPUBackendEngine(backend: backend, frameCoordinator: frameCoordinator)
        WGPULog.device.info("WebGPU 컨텍스트 시작 — \(resolved.name, privacy: .public)")
    }

    // MARK: - 캔버스 (WebGPURuntime)

    /// `<webgpu-canvas>`의 `CAMetalLayer`를 표면으로 붙인다.
    ///
    /// 표면 타입을 **런타임이** 고르는 것이 요점이다 — 브리지는 레이어만 넘기므로, 런타임을
    /// 갈아끼워도 엘리먼트 코드가 그대로다 (`WebGPURuntime.attachCanvas` 참고).
    public func attachCanvas(identifier: String, layer: CAMetalLayer) {
        engine.attachCanvas(identifier: identifier, layer: layer)
    }

    public func attachOffscreenCanvas(identifier: String, size: CGSize) throws {
        try engine.attachOffscreenCanvas(identifier: identifier, size: size)
    }

    public func resizeCanvas(identifier: String, drawableSize: CGSize) {
        engine.resizeCanvas(identifier: identifier, drawableSize: drawableSize)
    }

    public func detachCanvas(identifier: String) {
        engine.detachCanvas(identifier: identifier)
    }

    public func canvasInfo(identifier: String) -> [String: Any] {
        engine.canvasInfo(identifier: identifier)
    }

    public func readCanvasPixels(identifier: String) throws -> WGPUPixelReadback {
        try engine.readCanvasPixels(identifier: identifier)
    }

    // MARK: - 표면 등록 (Metal 고유 표면)

    /// 커스텀 `WGPUSurface` 구현(테스트 더블 등)을 직접 등록한다.
    public func registerSurface(_ surface: WGPUSurface) {
        engine.registerSurface(surface, identifier: surface.identifier, pacesFrames: surface.pacesFrames)
    }

    public func unregisterSurface(identifier: String) {
        engine.detachCanvas(identifier: identifier)
    }

    public func surface(for identifier: String) -> WGPUSurface? {
        engine.surface(for: identifier)
    }

    public var registeredSurfaceIdentifiers: [String] {
        engine.registeredSurfaceIdentifiers
    }

    /// 등록된 모든 표면이 새 프레임을 받을 수 있는가 — 회계는 `frameCoordinator`가 한다.
    public var isReadyForNextFrame: Bool { engine.isReadyForNextFrame }

    public func processEvents() {
        engine.processEvents()
    }

    // MARK: - 커맨드 실행

    /// 커맨드 스트림 하나를 실행한다.
    ///
    /// - Parameter payload: `{"commands": [ {op: …}, … ]}`
    /// - Returns: `{"ok": Bool, "errors": [...], "canvases": {...}}` — JS로 그대로 돌려준다.
    public func execute(_ payload: [String: Any]) -> [String: Any] {
        engine.execute(payload)
    }

    /// 버퍼 내용을 읽는다 (`GPUBuffer.mapAsync` + `getMappedRange`에 해당).
    public func readBuffer(
        handle: Int,
        offset: Int = 0,
        size: Int? = nil,
        completion: @escaping ([String: Any]) -> Void
    ) {
        engine.readBuffer(handle: handle, offset: offset, size: size, completion: completion)
    }

    /// 인코딩된 이미지를 풀어 `ImageBitmap` 자리의 객체로 등록한다 (JS `createImageBitmap`).
    public func decodeImage(
        handle: Int,
        data: Data?,
        name: String?,
        options: WGPUImageDecodeOptions,
        provider: WGPUAssetProvider?,
        completion: @escaping ([String: Any]) -> Void
    ) {
        engine.decodeImage(
            handle: handle, data: data, name: name,
            options: options, provider: provider, completion: completion
        )
    }

    /// `GPUShaderModule.getCompilationInfo()` — 그 모듈의 컴파일 진단.
    public func shaderCompilationInfo(handle: Int) -> [String: Any] {
        engine.shaderCompilationInfo(handle: handle)
    }

    /// `navigator.gpu.requestAdapter()` 가 돌려줄 어댑터 정보와 한계값 (`WGPUMetalBackend`).
    public func adapterInfo() -> [String: Any] {
        engine.adapterInfo()
    }

    /// 모든 GPU 객체를 버린다 (페이지 이탈 등).
    public func reset() {
        engine.reset()
    }

    public var liveObjectCount: Int { engine.liveObjectCount }

    /// 테스트 관찰용 — 업로드 스테이징 풀.
    var stagingPool: WGPUStagingPool { backend.stagingPool }
}
