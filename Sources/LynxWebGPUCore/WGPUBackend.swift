import Foundation
import CoreGraphics
import QuartzCore

/// GPU 백엔드가 구현하는 **동사(verb) 프로토콜** — `WGPUBackendEngine`의 반대편.
///
/// ## 왜 이 모양인가
///
/// `WebGPURuntime`은 와이어(커맨드 스트림) 전체를 받는 계약이라, 구현체가 배치 루프·오류
/// 스코프·프레임 수명·매핑 게이트·직렬화까지 전부 다시 만들어야 했다. Dawn 시제품에서 실제로
/// 터진 결함(펌프 경쟁, present 순서, 스코프 드레인 누락, 크래시성 변환)은 **전부 그 중복
/// 구현에서** 나왔다 — 인코딩이 아니라 오케스트레이션이 문제였다.
///
/// 그래서 오케스트레이션을 `WGPUBackendEngine`(Core) 한 곳으로 끌어올리고, 백엔드에는
/// **이미 검증·해석이 끝난 값으로 GPU API를 부르는 일**만 남긴다:
///
/// - 핸들 → 객체 해석은 엔진이 끝낸다. 동사는 `Buffer`·`Texture` 같은 **자기 타입**을 받는다.
/// - 명세 수준 검증(범위·정렬·usage·완전성·번들 격리·occlusion 중첩)은 엔진이 끝낸다.
///   동사에 도착한 값은 이미 명세를 통과한 값이다 — 백엔드 고유의 제약(기기 능력,
///   API별 변환 한계)만 동사 안에서 던진다.
/// - 디코딩·디스패치는 엔진의 exhaustive switch 한 곳이다. **op을 더하면 이 프로토콜에
///   동사를 더하게 되고, 컴파일러가 모든 백엔드의 누락을 잡는다** — 예전에 백엔드마다
///   두던 switch와 같은 보장을 프로토콜 요구사항이 대신한다.
///
/// ## 스레딩
///
/// 모든 동사는 엔진의 실행 락 아래에서 불린다 — `execute`(JS 스레드)·`pumpEvents`(메인
/// 틱)·`readBuffer` 등록이 전부 같은 락으로 직렬화된다 (`docs/COMMAND-STREAM.md` §5-1의
/// 펌프 동시성 계약을 엔진이 대신 이행한다). 백엔드가 자체 락을 둘 필요는 없지만,
/// 완료 콜백(`submit`·`readBuffer`의 클로저)은 **임의 스레드에서** 부를 수 있다 —
/// 엔진 쪽 래퍼가 자기 락을 다시 잡는다.
public protocol WGPUBackend: AnyObject {

    // MARK: - 핸들 타입

    associatedtype Buffer
    associatedtype Texture
    associatedtype TextureView
    associatedtype Sampler
    associatedtype ShaderModule
    associatedtype BindGroupLayout
    associatedtype PipelineLayout
    associatedtype BindGroup
    associatedtype RenderPipeline
    associatedtype ComputePipeline
    associatedtype QuerySet
    associatedtype RenderBundle
    associatedtype Surface

    // MARK: - 능력

    var capabilities: WGPUBackendCapabilities { get }

    /// 이 기기가 해당 압축 계열을 지원하는가 — 엔진의 압축 텍스처 생성 검증이 묻는다.
    func supportsTextureCompression(_ format: WGPUTextureFormat) -> Bool

    /// 간접 드로우·디스패치를 못 하는 기기면 **백엔드가 자기 문맥이 담긴 오류로** 던진다
    /// (Metal: 시뮬레이터 = Apple GPU family 2 안내). 나머지 간접 인자 검증(정렬·범위·
    /// usage)은 엔진이 한다.
    func ensureIndirectSupported() throws

    /// `navigator.gpu.requestAdapter()`가 쓰는 정보·한계값·기능. 키는 명세 철자 그대로
    /// (`WebGPURuntime.adapterInfo` 문서 참고). 값 전부가 기기·API 고유라 통째로 동사다.
    func adapterInfo() -> [String: Any]

    /// 비동기 완료 펌프 (`WebGPURuntime.processEvents`). 엔진 락 아래에서 불린다.
    func pumpEvents()

    /// 디바이스를 버릴 때 (`reset`) — 프레임 중간 상태(드로어블·마지막 커맨드 버퍼 등)를 버린다.
    func reset()

    // MARK: - 배치 수명

    /// 배치 시작 — 배치 단위 진단 수집이 필요한 백엔드의 자리다 (Dawn: 디바이스 오류
    /// 스코프 push). Metal처럼 완료 핸들러로 실패가 오는 백엔드는 할 일이 없다.
    func beginBatch()

    /// 배치가 제출된 뒤, 백엔드가 이 배치에서 모은 진단을 내놓는다 (Dawn: 스코프 pop +
    /// 펌프 + 오류 회수). 엔진이 오류 스코프/배치 결과로 흘려보낸다.
    func collectBatchDiagnostics() -> [WGPUError]

    /// 이 배치에서 제출할 GPU 작업이 만들어졌는가 (Metal: 커맨드 버퍼 존재).
    /// 엔진이 present·회계·프레임 만료를 이 값으로 판단한다.
    var hasPendingWork: Bool { get }

    /// 명령 없이 present만 해야 하는 배치(틱의 마무리)를 위해 빈 제출 거리를 만든다.
    /// 실패해도 조용히 넘어간다 — 다음 배치가 다시 시도한다.
    func ensureSubmittableWork()

    /// 배치의 GPU 작업을 제출한다. `hasPendingWork`가 참일 때만 불린다.
    ///
    /// - present가 참이면 이번 프레임에 획득한 표면 텍스처를 **백엔드 규칙에 맞는 시점**에
    ///   화면으로 보낸다 (Metal: commit 전에 `present(drawable)`, Dawn: submit 후
    ///   `wgpuSurfacePresent`). 보낸 뒤에는 백엔드가 들고 있던 획득 목록을 비운다.
    /// - `onCompleted`는 GPU 실행이 끝났을 때 **임의 스레드에서** 부른다. 실패면 오류를
    ///   담는다 — 엔진이 지연 오류 큐로 흘려 다음 배치에 보고한다.
    func submit(present: Bool, onCompleted: @escaping (WGPUError?) -> Void)

    // MARK: - 리소스

    func makeBuffer(_ descriptor: WGPUBufferDescriptor) throws -> Buffer
    func writeBuffer(_ buffer: Buffer, offset: Int, data: Data) throws

    /// `unmap()` — 와이어 매핑 상태는 엔진이 관리하고, **실제 매핑을 가진 백엔드**(Dawn의
    /// `mappedAtCreation`)만 여기서 자기 매핑을 푼다. 그런 것이 없으면 no-op이다.
    func unmapBuffer(_ buffer: Buffer)

    /// 버퍼 내용을 읽는다 — 앞서 제출한 GPU 작업이 끝난 뒤의 값을 보장할 것.
    /// `deliver`는 임의 스레드에서, 이미 끝났으면 동기로도 부를 수 있다.
    func readBuffer(_ buffer: Buffer, offset: Int, length: Int,
                    deliver: @escaping (Result<Data, WGPUError>) -> Void)

    func makeTexture(_ descriptor: WGPUTextureDescriptor) throws -> Texture

    /// CPU 데이터를 텍스처로 올린다 — `writeTexture`와 `copyExternalImageToTexture`가
    /// 여기로 수렴한다 (후자는 엔진이 디코딩·잘라내기·flipY를 끝내고 픽셀만 넘긴다).
    /// `bytesPerRow`·`rowsPerImage`의 생략 기본값은 엔진이 채워서 온다.
    func writeTexture(_ texture: Texture, data: Data, origin: WGPUOrigin3D, size: WGPUExtent3D,
                      mipLevel: Int, bytesPerRow: Int, rowsPerImage: Int) throws

    /// `format`은 엔진이 확정한 뷰 포맷 (`descriptor.format ?? 원본 포맷`).
    func makeTextureView(_ texture: Texture, descriptor: WGPUTextureViewDescriptor,
                         format: WGPUTextureFormat) throws -> TextureView

    func makeSampler(_ descriptor: WGPUSamplerDescriptor) throws -> Sampler

    /// 명세에서 셰이더 모듈은 **컴파일에 실패해도 만들어진다** — 컴파일 진단은 던지지 않고
    /// 결과의 `failure`에 실어 준다 (엔진이 등록은 하되 진단을 그 자리에서 보고한다).
    /// 던지는 것은 **모듈 자체를 만들 수 없는 경우**뿐이다 (지원하지 않는 언어 등).
    func makeShaderModule(_ descriptor: WGPUShaderModuleDescriptor,
                          fieldPath: (String) -> String?) throws -> WGPUShaderModuleCreation<Self>

    /// `getCompilationInfo()` — 지금까지 쌓인 진단 (파이프라인 생성 실패 포함).
    func compilationMessages(of module: ShaderModule) -> [WGPUCompilationMessage]

    func makeBindGroupLayout(_ entries: [WGPUBindGroupLayoutEntry]) throws -> BindGroupLayout
    func makePipelineLayout(_ groups: [BindGroupLayout]) throws -> PipelineLayout

    /// 리소스 해석(핸들 → 객체, `boundSize` 기본값, 레이아웃 항목 매칭)은 엔진이 끝냈다.
    /// `layoutEntry`는 레이아웃 항목을 아는 경우에만 온다 — 네이티브 파생 레이아웃
    /// (`getBindGroupLayout`이 항목을 못 주는 백엔드)에서는 nil이다.
    func makeBindGroup(layout: BindGroupLayout,
                       entries: [WGPUResolvedBindGroupEntry<Self>]) throws -> BindGroup

    func makeQuerySet(_ descriptor: WGPUQuerySetDescriptor) throws -> QuerySet

    /// 진입점 해석(생략 시 유일 진입점)은 **백엔드 몫**이다 — Metal은 WGSL 리플렉션으로,
    /// Dawn은 네이티브 기본 규칙으로 한다. `info`에는 엔진의 드로우 전 검사가 쓸 메타데이터를
    /// 담는다 — 백엔드가 스스로 검증하는 항목은 nil로 두면 엔진 검사가 빠진다.
    /// `fieldPath`는 오류에 붙일 커맨드 스트림 경로를 만든다 (`fieldPath("vertex.entryPoint")`
    /// → `commands[3].vertex.entryPoint`) — 백엔드가 세부 필드 단위 진단을 보낼 때 쓴다.
    func makeRenderPipeline(_ descriptor: WGPURenderPipelineDescriptor,
                            vertexModule: ShaderModule, fragmentModule: ShaderModule?,
                            layout: WGPUResolvedPipelineLayout<Self>,
                            fieldPath: (String) -> String?) throws -> WGPURenderPipelineCreation<Self>
    func makeComputePipeline(_ descriptor: WGPUComputePipelineDescriptor,
                             module: ShaderModule,
                             layout: WGPUResolvedPipelineLayout<Self>,
                             fieldPath: (String) -> String?) throws -> WGPUComputePipelineCreation<Self>

    /// `pipeline.getBindGroupLayout(index)`. 그 자리에 그룹이 없으면 nil을 돌려준다 —
    /// 엔진이 명세 문구의 오류로 바꾼다. `entries`를 알면 함께 준다 (엔진의 바인드 그룹
    /// 항목 매칭이 쓴다).
    func bindGroupLayout(of pipeline: WGPUResolvedPipeline<Self>,
                         index: Int) throws -> WGPUBindGroupLayoutCreation<Self>?

    /// 네이티브 렌더 번들 생성 — `capabilities.supportsNativeRenderBundles`가 참인 백엔드만
    /// 불린다. 명령은 엔진이 디코딩을 끝낸 값이고, 안에 든 핸들은 `resolver`로 푼다.
    /// 거짓인 백엔드(Metal)에서는 엔진이 record/replay로 대신한다 — 불릴 일이 없다.
    func makeRenderBundle(_ descriptor: WGPURenderBundleDescriptor, commands: [WGPUCommand],
                          resolver: WGPUBundleResolver<Self>) throws -> RenderBundle

    // MARK: - 표면

    /// `CAMetalLayer`는 두 백엔드의 공통분모다 — Dawn도 Apple 플랫폼에서 같은 레이어를 받는다
    /// (`WebGPURuntime.attachCanvas` 문서).
    func makeLayerSurface(identifier: String, layer: CAMetalLayer) -> WGPUSurfaceCreation<Self>
    func makeOffscreenSurface(identifier: String, size: CGSize) throws -> WGPUSurfaceCreation<Self>
    func configureSurface(_ surface: Surface, configuration: WGPUCanvasConfiguration) throws
    /// 메인 스레드에서 온다 (레이아웃 변경).
    func resizeSurface(_ surface: Surface, size: CGSize)
    /// 현재 픽셀 크기·포맷 실측값 — `canvasInfo`와 배치 결과의 `canvases` 보고가 쓴다.
    func surfaceReport(_ surface: Surface) -> WGPUSurfaceReport

    /// 이번 프레임에 그릴 표면 텍스처를 얻는다. nil이면 엔진이 "크기가 0이거나 드로어블
    /// 고갈" 오류로 바꾼다. present 대상 기억은 백엔드 몫이다 (`submit(present:)`가 쓴다).
    func acquireFrameTexture(_ surface: Surface) throws -> WGPUAcquiredSurfaceTexture<Self>?

    /// 오프스크린 표면의 픽셀 읽기 — 아닌 표면이면 백엔드가 자기 문구로 던진다.
    func readPixels(_ surface: Surface, identifier: String) throws -> WGPUPixelReadback

    // MARK: - 패스

    func beginRenderPass(_ pass: WGPUResolvedRenderPass<Self>) throws
    func beginComputePass(_ pass: WGPUResolvedComputePass<Self>) throws
    /// 열려 있는 패스/내부 인코더를 닫는다. 여러 번 불려도 안전해야 한다.
    func endPass()

    /// 패스가 열려 있을 때만 불린다 (엔진이 보장). 파이프라인 교체에 따르는 백엔드 상태
    /// (컬링·와인딩·깊이 상태 등)도 여기서 함께 올린다.
    func setRenderPipeline(_ pipeline: RenderPipeline)
    func setComputePipeline(_ pipeline: ComputePipeline)

    /// 드로우 직전, 엔진의 그림자 상태에서 **더러워진 그룹만** 내려온다 — 번들 경계의
    /// 바인딩 무효화가 엔진 그림자 상태로 표현되기 때문이다 (`WGPUBackendEngine` 문서).
    func applyBindGroup(_ group: BindGroup, at index: Int, dynamicOffsets: [Int]) throws
    /// 드로우 직전, 더러워진 슬롯만 내려온다. `offset`은 엔진이 범위를 확인한 값이다.
    func applyVertexBuffer(_ buffer: Buffer, offset: Int, slot: Int)

    func setViewport(_ command: WGPUSetViewportCommand) throws
    func setScissorRect(_ command: WGPUSetScissorRectCommand) throws
    func setBlendConstant(_ color: WGPUColor) throws
    func setStencilReference(_ reference: UInt32) throws

    /// `index`는 엔진이 범위·중첩·재사용을 확인한 값이다.
    func beginOcclusionQuery(index: Int) throws
    func endOcclusionQuery(index: Int) throws

    /// 네이티브 번들 실행 — `supportsNativeRenderBundles`가 참인 백엔드만 불린다.
    /// 호환성 검증은 엔진이 먼저 끝냈다.
    func executeBundles(_ bundles: [RenderBundle]) throws

    /// `scope`는 명세의 두 층 그대로다 — 패스 안(.pass)이냐 프레임 구간(.frame)이냐.
    /// 짝 맞추기(깊이 계산)는 엔진이 한다.
    func pushDebugGroup(_ label: String, scope: WGPUDebugScope) throws
    func popDebugGroup(scope: WGPUDebugScope)
    /// 제출 직전, 프레임 구간에 열린 채 남은 그룹을 이만큼 닫는다 (엔진이 오류로 알린 뒤).
    func popFrameDebugGroups(count: Int)
    func insertDebugMarker(_ label: String, scope: WGPUDebugScope) throws

    // MARK: - 드로우 / 디스패치

    func draw(_ command: WGPUDrawCommand) throws
    func drawIndexed(_ command: WGPUDrawIndexedCommand, index: WGPUResolvedIndexBinding<Self>) throws
    func drawIndirect(buffer: Buffer, offset: Int) throws
    func drawIndexedIndirect(buffer: Buffer, offset: Int, index: WGPUResolvedIndexBinding<Self>) throws
    func dispatchWorkgroups(_ command: WGPUDispatchWorkgroupsCommand) throws
    func dispatchWorkgroupsIndirect(buffer: Buffer, offset: Int) throws

    // MARK: - 복사

    func copyBufferToBuffer(source: Buffer, sourceOffset: Int,
                            destination: Buffer, destinationOffset: Int, size: Int) throws
    /// `range`는 엔진이 정렬·범위를 확인했고 비어 있지 않다. 0으로 채운다.
    func clearBuffer(_ buffer: Buffer, range: Range<Int>) throws
    func copyTextureToBuffer(texture: Texture, slice: Int, mipLevel: Int, origin: WGPUOrigin3D,
                             size: WGPUExtent3D, buffer: Buffer, offset: Int,
                             bytesPerRow: Int, bytesPerImage: Int) throws
    func copyBufferToTexture(buffer: Buffer, offset: Int, bytesPerRow: Int, bytesPerImage: Int,
                             texture: Texture, slice: Int, mipLevel: Int, origin: WGPUOrigin3D,
                             size: WGPUExtent3D) throws
    func copyTextureToTexture(source: Texture, sourceSlice: Int, sourceMipLevel: Int,
                              sourceOrigin: WGPUOrigin3D,
                              destination: Texture, destinationSlice: Int, destinationMipLevel: Int,
                              destinationOrigin: WGPUOrigin3D, size: WGPUExtent3D) throws
    /// 종류(occlusion/timestamp)에 따른 blit 선택은 백엔드 몫이다. 범위·정렬·usage는
    /// 엔진이 확인했고 `count > 0`이다.
    func resolveQuerySet(_ querySet: QuerySet, first: Int, count: Int,
                         destination: Buffer, destinationOffset: Int) throws
}

// MARK: - 능력

public struct WGPUBackendCapabilities {
    /// 렌더 번들을 백엔드가 네이티브로 기록·실행하는가.
    /// 거짓이면 엔진이 record/replay로 대신한다 (Metal — 대응 객체가 없다).
    public let supportsNativeRenderBundles: Bool
    /// 정점 버퍼 슬롯 수 상한 — 엔진의 `setVertexBuffer` 슬롯 검사가 쓴다
    /// (Metal은 인자 테이블 배정 규칙에서, Dawn은 명세 기본값 8에서 나온다).
    public let maxVertexBufferSlots: Int

    public init(supportsNativeRenderBundles: Bool, maxVertexBufferSlots: Int) {
        self.supportsNativeRenderBundles = supportsNativeRenderBundles
        self.maxVertexBufferSlots = maxVertexBufferSlots
    }
}

// MARK: - 동사 인자 값 타입

/// 디버그 그룹·마커의 스코프 — 패스 안 구간과 프레임 구간 (`docs/COMMAND-STREAM.md`).
public enum WGPUDebugScope {
    case pass
    case frame
}

public struct WGPUShaderModuleCreation<B: WGPUBackend> {
    public let module: B.ShaderModule
    /// 모듈을 쓸 수 없게 하는 진단 (파싱 실패 등). 있어도 모듈은 등록된다.
    public let failure: WGPUError?

    public init(module: B.ShaderModule, failure: WGPUError? = nil) {
        self.module = module
        self.failure = failure
    }
}

/// `getCompilationInfo()`의 메시지 하나 — 명세 `GPUCompilationMessage` 모양.
public struct WGPUCompilationMessage {
    public let message: String
    /// `"error"` / `"warning"` / `"info"`.
    public let type: String
    public let lineNum: Int
    public let linePos: Int
    public let offset: Int
    public let length: Int

    public init(message: String, type: String = "error",
                lineNum: Int = 0, linePos: Int = 0, offset: Int = 0, length: Int = 0) {
        self.message = message
        self.type = type
        self.lineNum = lineNum
        self.linePos = linePos
        self.offset = offset
        self.length = length
    }
}

/// 엔진의 드로우/디스패치 전 검사가 쓰는 파이프라인 메타데이터.
/// **nil은 "백엔드가 스스로 검증한다"는 뜻**이다 — 그 항목의 엔진 검사가 빠진다.
public struct WGPURenderPipelineInfo {
    /// 드로우 전에 바인드되어 있어야 하는 그룹 인덱스 (빈 그룹은 제외).
    public let requiredGroups: Set<Int>?
    /// 드로우 전에 바인드되어 있어야 하는 정점 버퍼 슬롯 (`vertex.buffers` 선언).
    public let requiredVertexSlots: Set<Int>?
    /// `depthReadOnly`/`stencilReadOnly` 패스에서 거부할 때 쓴다.
    public let writesDepth: Bool?
    public let writesStencil: Bool?

    public init(requiredGroups: Set<Int>? = nil, requiredVertexSlots: Set<Int>? = nil,
                writesDepth: Bool? = nil, writesStencil: Bool? = nil) {
        self.requiredGroups = requiredGroups
        self.requiredVertexSlots = requiredVertexSlots
        self.writesDepth = writesDepth
        self.writesStencil = writesStencil
    }
}

public struct WGPUComputePipelineInfo {
    public let requiredGroups: Set<Int>?

    public init(requiredGroups: Set<Int>? = nil) {
        self.requiredGroups = requiredGroups
    }
}

public struct WGPURenderPipelineCreation<B: WGPUBackend> {
    public let pipeline: B.RenderPipeline
    public let info: WGPURenderPipelineInfo

    public init(pipeline: B.RenderPipeline, info: WGPURenderPipelineInfo) {
        self.pipeline = pipeline
        self.info = info
    }
}

public struct WGPUComputePipelineCreation<B: WGPUBackend> {
    public let pipeline: B.ComputePipeline
    public let info: WGPUComputePipelineInfo

    public init(pipeline: B.ComputePipeline, info: WGPUComputePipelineInfo) {
        self.pipeline = pipeline
        self.info = info
    }
}

public struct WGPUBindGroupLayoutCreation<B: WGPUBackend> {
    public let layout: B.BindGroupLayout
    /// 레이아웃 항목을 알면 준다 — 엔진의 바인드 그룹 항목 매칭·visibility 판정이 쓴다.
    /// 네이티브 파생 레이아웃이라 항목을 모르면 nil (그 검사는 백엔드 검증에 맡겨진다).
    public let entries: [WGPUBindGroupLayoutEntry]?

    public init(layout: B.BindGroupLayout, entries: [WGPUBindGroupLayoutEntry]?) {
        self.layout = layout
        self.entries = entries
    }
}

public enum WGPUResolvedPipelineLayout<B: WGPUBackend> {
    case auto
    case explicit(B.PipelineLayout)
}

public enum WGPUResolvedPipeline<B: WGPUBackend> {
    case render(B.RenderPipeline)
    case compute(B.ComputePipeline)
}

public struct WGPUResolvedBindGroupEntry<B: WGPUBackend> {
    public enum Resource {
        /// `boundSize`는 이 바인딩이 보는 바이트 수 — 생략 기본값(`버퍼 끝까지`)은 엔진이 채웠다.
        case buffer(B.Buffer, offset: Int, boundSize: Int)
        case sampler(B.Sampler)
        case textureView(B.TextureView)
    }

    public let binding: Int
    /// 매칭된 레이아웃 항목 — visibility·dynamic offset 여부가 여기 있다.
    /// 항목을 모르는 레이아웃(네이티브 파생)이면 nil.
    public let layoutEntry: WGPUBindGroupLayoutEntry?
    public let resource: Resource

    public init(binding: Int, layoutEntry: WGPUBindGroupLayoutEntry?, resource: Resource) {
        self.binding = binding
        self.layoutEntry = layoutEntry
        self.resource = resource
    }
}

public struct WGPUResolvedIndexBinding<B: WGPUBackend> {
    public let buffer: B.Buffer
    public let offset: Int
    public let format: WGPUIndexFormat
    /// 인덱스 하나의 바이트 수 — `drawIndexed`의 `firstIndex` 바이트 환산에 쓴다.
    public let stride: Int

    public init(buffer: B.Buffer, offset: Int, format: WGPUIndexFormat, stride: Int) {
        self.buffer = buffer
        self.offset = offset
        self.format = format
        self.stride = stride
    }
}

public struct WGPUResolvedTimestampWrites<B: WGPUBackend> {
    public let querySet: B.QuerySet
    public let beginningOfPassWriteIndex: Int?
    public let endOfPassWriteIndex: Int?

    public init(querySet: B.QuerySet, beginningOfPassWriteIndex: Int?, endOfPassWriteIndex: Int?) {
        self.querySet = querySet
        self.beginningOfPassWriteIndex = beginningOfPassWriteIndex
        self.endOfPassWriteIndex = endOfPassWriteIndex
    }
}

/// 핸들이 전부 백엔드 객체로 풀린 렌더 패스 — `beginRenderPass` 동사의 인자.
public struct WGPUResolvedRenderPass<B: WGPUBackend> {
    public struct ColorAttachment {
        public let view: B.TextureView
        public let resolveTarget: B.TextureView?
        public let loadOp: WGPULoadOp
        public let storeOp: WGPUStoreOp
        public let clearValue: WGPUColor

        public init(view: B.TextureView, resolveTarget: B.TextureView?,
                    loadOp: WGPULoadOp, storeOp: WGPUStoreOp, clearValue: WGPUColor) {
            self.view = view
            self.resolveTarget = resolveTarget
            self.loadOp = loadOp
            self.storeOp = storeOp
            self.clearValue = clearValue
        }
    }

    public struct DepthStencilAttachment {
        public let view: B.TextureView
        /// 뷰의 WebGPU 포맷 — 백엔드가 깊이/스텐실 aspect 유무를 가를 때 쓴다.
        public let format: WGPUTextureFormat
        /// readOnly면 nil이다 — 내용을 그대로 읽고 그대로 남긴다 (load/store에 해당).
        public let depthLoadOp: WGPULoadOp?
        public let depthStoreOp: WGPUStoreOp?
        public let depthClearValue: Double
        public let stencilLoadOp: WGPULoadOp?
        public let stencilStoreOp: WGPUStoreOp?
        public let stencilClearValue: Int
        public let depthReadOnly: Bool
        public let stencilReadOnly: Bool

        public init(view: B.TextureView, format: WGPUTextureFormat,
                    depthLoadOp: WGPULoadOp?, depthStoreOp: WGPUStoreOp?, depthClearValue: Double,
                    stencilLoadOp: WGPULoadOp?, stencilStoreOp: WGPUStoreOp?, stencilClearValue: Int,
                    depthReadOnly: Bool, stencilReadOnly: Bool) {
            self.view = view
            self.format = format
            self.depthLoadOp = depthLoadOp
            self.depthStoreOp = depthStoreOp
            self.depthClearValue = depthClearValue
            self.stencilLoadOp = stencilLoadOp
            self.stencilStoreOp = stencilStoreOp
            self.stencilClearValue = stencilClearValue
            self.depthReadOnly = depthReadOnly
            self.stencilReadOnly = stencilReadOnly
        }
    }

    public let label: String?
    public let colorAttachments: [ColorAttachment]
    public let depthStencil: DepthStencilAttachment?
    public let occlusionQuerySet: B.QuerySet?
    public let timestampWrites: WGPUResolvedTimestampWrites<B>?

    public init(label: String?, colorAttachments: [ColorAttachment],
                depthStencil: DepthStencilAttachment?, occlusionQuerySet: B.QuerySet?,
                timestampWrites: WGPUResolvedTimestampWrites<B>?) {
        self.label = label
        self.colorAttachments = colorAttachments
        self.depthStencil = depthStencil
        self.occlusionQuerySet = occlusionQuerySet
        self.timestampWrites = timestampWrites
    }
}

public struct WGPUResolvedComputePass<B: WGPUBackend> {
    public let label: String?
    public let timestampWrites: WGPUResolvedTimestampWrites<B>?

    public init(label: String?, timestampWrites: WGPUResolvedTimestampWrites<B>?) {
        self.label = label
        self.timestampWrites = timestampWrites
    }
}

// MARK: - 표면 값 타입

public struct WGPUSurfaceCreation<B: WGPUBackend> {
    public let surface: B.Surface
    /// 드로어블 풀이 있어 밀릴 수 있는 표면인가 — 프레임 회계(`WGPUFrameCoordinator`) 대상.
    public let pacesFrames: Bool

    public init(surface: B.Surface, pacesFrames: Bool) {
        self.surface = surface
        self.pacesFrames = pacesFrames
    }
}

public struct WGPUSurfaceReport {
    public let width: Int
    public let height: Int
    public let format: WGPUTextureFormat

    public init(width: Int, height: Int, format: WGPUTextureFormat) {
        self.width = width
        self.height = height
        self.format = format
    }
}

public struct WGPUAcquiredSurfaceTexture<B: WGPUBackend> {
    public let texture: B.Texture
    /// 실제 텍스처의 포맷 — 캔버스 설정 반영이 한 프레임 늦을 수 있어 설정값이 아니라 실측이다.
    public let format: WGPUTextureFormat
    public let width: Int
    public let height: Int
    public let sampleCount: Int

    public init(texture: B.Texture, format: WGPUTextureFormat, width: Int, height: Int,
                sampleCount: Int = 1) {
        self.texture = texture
        self.format = format
        self.width = width
        self.height = height
        self.sampleCount = sampleCount
    }
}

/// 네이티브 번들 기록 중 명령 안의 핸들을 푸는 조회 창구 — 엔진 레지스트리를 백엔드에
/// 통째로 주지 않기 위한 좁은 통로다.
public struct WGPUBundleResolver<B: WGPUBackend> {
    public let renderPipeline: (WGPUHandle, String?) throws -> B.RenderPipeline
    public let bindGroup: (WGPUHandle, String?) throws -> B.BindGroup
    public let buffer: (WGPUHandle, String?) throws -> B.Buffer

    public init(renderPipeline: @escaping (WGPUHandle, String?) throws -> B.RenderPipeline,
                bindGroup: @escaping (WGPUHandle, String?) throws -> B.BindGroup,
                buffer: @escaping (WGPUHandle, String?) throws -> B.Buffer) {
        self.renderPipeline = renderPipeline
        self.bindGroup = bindGroup
        self.buffer = buffer
    }
}

// MARK: - 압축 계열

/// 포맷이 속한 압축 계열 — 명세의 선택 기능 이름과 1:1로 대응한다.
/// (기기 지원 여부는 백엔드가 안다 — `WGPUBackend.supportsTextureCompression`.)
public enum WGPUTextureCompressionFamily {
    case none, bc, etc2, astc

    /// `adapter.features`에 싣는 이름 (명세 철자 그대로).
    public var featureName: String? {
        switch self {
        case .none: return nil
        case .bc: return "texture-compression-bc"
        case .etc2: return "texture-compression-etc2"
        case .astc: return "texture-compression-astc"
        }
    }
}

public extension WGPUTextureFormat {
    /// ETC2와 EAC는 명세에서 **같은 기능 비트**다 (`texture-compression-etc2`).
    var compressionFamily: WGPUTextureCompressionFamily {
        guard isCompressed else { return .none }
        if rawValue.hasPrefix("bc") { return .bc }
        if rawValue.hasPrefix("astc-") { return .astc }
        return .etc2
    }
}
