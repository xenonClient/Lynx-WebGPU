import Foundation

/// 엔진이 레지스트리에 저장하는 복합 객체들 — **백엔드 핸들 + 명세 검증에 필요한 메타데이터**.
///
/// 검증을 엔진이 하려면 크기·usage·포맷 같은 명세 수준 사실을 엔진이 알아야 한다.
/// 백엔드 객체에 물어보는 대신 생성 시점에 여기 새겨 둔다 — 백엔드는 조회 API가 없을 수도
/// 있고(Dawn은 버퍼 usage를 되묻는 API가 없다), 있어도 계약이 제각각이기 때문이다.
///
/// `isMapped` 같은 가변 상태는 전부 **엔진의 실행 락 아래에서만** 읽고 쓴다.

public final class WGPUEngineBuffer<B: WGPUBackend> {
    public let raw: B.Buffer
    public let size: Int
    public let usage: WGPUBufferUsage
    /// 지금 CPU에 매핑되어 있는가 (`mapAsync` ~ `unmap` 사이) — 명세의 "unavailable".
    /// 이 상태의 버퍼를 큐 작업에 쓰는 것은 엔진의 매핑 게이트가 막는다.
    public var isMapped = false

    public init(raw: B.Buffer, size: Int, usage: WGPUBufferUsage) {
        self.raw = raw
        self.size = size
        self.usage = usage
    }
}

public final class WGPUEngineTexture<B: WGPUBackend> {
    public let raw: B.Texture
    public let format: WGPUTextureFormat
    public let size: WGPUExtent3D
    public let sampleCount: Int
    /// 캔버스 드로어블에서 온 텍스처인가 — 프레임이 present되면 핸들이 만료된다.
    public let isDrawable: Bool

    public init(raw: B.Texture, format: WGPUTextureFormat, size: WGPUExtent3D,
                sampleCount: Int, isDrawable: Bool) {
        self.raw = raw
        self.format = format
        self.size = size
        self.sampleCount = sampleCount
        self.isDrawable = isDrawable
    }
}

public final class WGPUEngineTextureView<B: WGPUBackend> {
    public let raw: B.TextureView
    public let format: WGPUTextureFormat
    public let sampleCount: Int

    public init(raw: B.TextureView, format: WGPUTextureFormat, sampleCount: Int) {
        self.raw = raw
        self.format = format
        self.sampleCount = sampleCount
    }
}

public final class WGPUEngineSampler<B: WGPUBackend> {
    public let raw: B.Sampler

    public init(raw: B.Sampler) { self.raw = raw }
}

public final class WGPUEngineShaderModule<B: WGPUBackend> {
    public let raw: B.ShaderModule

    public init(raw: B.ShaderModule) { self.raw = raw }
}

public final class WGPUEngineBindGroupLayout<B: WGPUBackend> {
    public let raw: B.BindGroupLayout
    /// 항목을 아는 레이아웃이면 바인딩 오름차순으로 정렬해 둔다. 네이티브 파생 레이아웃
    /// (`getBindGroupLayout`이 항목을 못 주는 백엔드)은 nil — 항목 매칭 검사가 빠지고
    /// 백엔드 검증에 맡겨진다.
    public let entries: [WGPUBindGroupLayoutEntry]?

    public init(raw: B.BindGroupLayout, entries: [WGPUBindGroupLayoutEntry]?) {
        self.raw = raw
        self.entries = entries?.sorted { $0.binding < $1.binding }
    }

    public func entry(binding: Int) -> WGPUBindGroupLayoutEntry? {
        entries?.first { $0.binding == binding }
    }
}

public final class WGPUEnginePipelineLayout<B: WGPUBackend> {
    public let raw: B.PipelineLayout

    public init(raw: B.PipelineLayout) { self.raw = raw }
}

public final class WGPUEngineBindGroup<B: WGPUBackend> {
    public let raw: B.BindGroup
    /// 이 그룹이 물고 있는 버퍼 — 드로우 직전 "매핑 중인가"를 보려면 필요하다
    /// (그룹은 만들 때 버퍼를 고정하므로, 만든 뒤에 매핑된 경우가 거기서 걸린다).
    public let buffers: [WGPUEngineBuffer<B>]

    public init(raw: B.BindGroup, buffers: [WGPUEngineBuffer<B>]) {
        self.raw = raw
        self.buffers = buffers
    }
}

public final class WGPUEngineRenderPipeline<B: WGPUBackend> {
    public let raw: B.RenderPipeline
    public let info: WGPURenderPipelineInfo

    public init(raw: B.RenderPipeline, info: WGPURenderPipelineInfo) {
        self.raw = raw
        self.info = info
    }
}

public final class WGPUEngineComputePipeline<B: WGPUBackend> {
    public let raw: B.ComputePipeline
    public let info: WGPUComputePipelineInfo

    public init(raw: B.ComputePipeline, info: WGPUComputePipelineInfo) {
        self.raw = raw
        self.info = info
    }
}

public final class WGPUEngineQuerySet<B: WGPUBackend> {
    /// 결과 하나의 크기 — 두 종류 다 `u64`다 (명세와 백엔드들이 일치한다).
    public static var resultStride: Int { 8 }

    public let raw: B.QuerySet
    public let type: WGPUQueryType
    public let count: Int

    public init(raw: B.QuerySet, type: WGPUQueryType, count: Int) {
        self.raw = raw
        self.type = type
        self.count = count
    }

    /// 쿼리 구간이 이 셋 안에 들어오는지. 넘기면 백엔드가 단언으로 죽을 수 있어 여기서 막는다.
    public func checkRange(first: Int, count queryCount: Int, path: String?) throws {
        guard first >= 0, queryCount >= 0, first + queryCount <= count else {
            throw WGPUError.validation(
                "쿼리 범위를 벗어났다 — \(first)부터 \(queryCount)개, 쿼리셋 크기 \(count)",
                path: path
            )
        }
    }
}

/// `GPURenderBundle` — 명령 목록(record/replay 백엔드용)이나 네이티브 번들, 또는 둘 다.
///
/// 번들의 계약은 "직접 인코딩과 같은 결과"다. 네이티브 번들이 없는 백엔드(Metal)에서는
/// 명령을 저장했다가 현재 패스에 다시 흘리는 것으로 계약을 그대로 만족시킨다 — 재사용해도
/// 안전한 이유도 같다: 저장된 것은 값 타입인 리더뿐이라 실행이 원본을 바꾸지 않는다.
public final class WGPUEngineRenderBundle<B: WGPUBackend> {
    /// 번들 안에서 쓸 수 있는 명령. 명세가 정한 목록 그대로다 — 뷰포트·시저·블렌드 상수·
    /// 스텐실 참조·복사·중첩 번들은 번들에 담을 수 없다.
    ///
    /// 디버그 마커는 **담을 수 있다** — 명세의 `GPURenderBundleEncoder`가
    /// `GPUDebugCommandsMixin`을 포함한다. 빠뜨리면 마커를 하나 넣은 것만으로 번들 전체가
    /// 거부되고, 사용자는 마커가 원인이라고 생각하기 어렵다.
    public static var allowedOps: Set<String> {
        [
            "setPipeline", "setBindGroup", "setVertexBuffer", "setIndexBuffer",
            "draw", "drawIndexed", "drawIndirect", "drawIndexedIndirect",
            "pushDebugGroup", "popDebugGroup", "insertDebugMarker",
        ]
    }

    /// replay 백엔드용 — 실행할 때마다 다시 디코딩한다 (디코딩 오류는 실행 시점에 드러난다).
    public let commands: [WGPUValueReader]
    /// 네이티브 백엔드용 — 만들 때 이미 기록이 끝난 번들.
    public let native: B.RenderBundle?
    public let descriptor: WGPURenderBundleDescriptor

    /// `validateOps(_:)`를 먼저 통과한 명령만 담을 것 — 네이티브 기록은 디코딩보다도 먼저
    /// 목록 검증이 끝나야 해서, 검증이 init에 있으면 순서를 강제할 수 없다.
    public init(commands: [WGPUValueReader], native: B.RenderBundle?,
                descriptor: WGPURenderBundleDescriptor) {
        self.commands = commands
        self.native = native
        self.descriptor = descriptor
    }

    /// 번들에 담을 수 있는 명령인지 목록을 확인한다 (명세 `GPURenderBundleEncoder`의 op 집합).
    public static func validateOps(_ commands: [WGPUValueReader]) throws {
        for command in commands {
            let op = try command.requiredString("op")
            guard allowedOps.contains(op) else {
                throw WGPUError.validation(
                    "렌더 번들에는 '\(op)'을(를) 담을 수 없다 "
                        + "(가능: \(allowedOps.sorted().joined(separator: ", ")))",
                    path: command.fieldPath("op")
                )
            }
        }
    }

    /// 이 번들이 지금 패스에서 실행될 수 있는가.
    ///
    /// 번들은 "어떤 모양의 패스에서 쓸 것"이라고 선언하고 만들어진다. 그 선언과 실제 패스가
    /// 어긋나면 브라우저는 오류를 내지만, replay 구현은 명령을 되풀이할 뿐이라 백엔드가 못
    /// 잡는다 (파이프라인이 패스와 맞기만 하면 그냥 그려진다). 여기서 막지 않으면
    /// 브라우저에서만 깨지는 코드가 나간다.
    public func checkCompatibility(
        color: [WGPUTextureFormat],
        depthStencil: WGPUTextureFormat?,
        sampleCount: Int,
        depthReadOnly: Bool,
        stencilReadOnly: Bool
    ) throws {
        // 명세의 "render pass layout equals"는 **후행 null을 무시하고** colorFormats를 비교한다.
        // 자르지 않으면 `['bgra8unorm', null]` 번들이 컬러 1개짜리 패스에서 오탐으로 거부된다.
        let bundleFormats = Self.trimmingTrailingNulls(descriptor.colorFormats)
        guard bundleFormats.count == color.count else {
            throw WGPUError.validation(
                "번들의 컬러 어태치먼트 수(\(bundleFormats.count))가 "
                    + "패스(\(color.count))와 다르다"
            )
        }
        for (index, expected) in bundleFormats.enumerated() where expected != color[index] {
            throw WGPUError.validation(
                "번들의 colorFormats[\(index)]가 패스와 다르다 — "
                    + "번들 \(expected?.rawValue ?? "null"), 패스 \(color[index].rawValue)"
            )
        }
        guard descriptor.depthStencilFormat == depthStencil else {
            throw WGPUError.validation(
                "번들의 depthStencilFormat이 패스와 다르다 — "
                    + "번들 \(descriptor.depthStencilFormat?.rawValue ?? "없음"), "
                    + "패스 \(depthStencil?.rawValue ?? "없음")"
            )
        }
        guard descriptor.sampleCount == sampleCount else {
            throw WGPUError.validation(
                "번들의 sampleCount(\(descriptor.sampleCount))가 패스(\(sampleCount))와 다르다"
            )
        }
        // 한 방향만 요구한다 — read-only 패스에는 read-only 번들만 넣을 수 있지만,
        // 쓰기가 가능한 패스에 read-only 번들을 넣는 것은 문제가 없다.
        guard !depthReadOnly || descriptor.depthReadOnly else {
            throw WGPUError.validation(
                "depthReadOnly 패스에서는 depthReadOnly: true로 만든 번들만 실행할 수 있다"
            )
        }
        guard !stencilReadOnly || descriptor.stencilReadOnly else {
            throw WGPUError.validation(
                "stencilReadOnly 패스에서는 stencilReadOnly: true로 만든 번들만 실행할 수 있다"
            )
        }
    }

    /// 후행 `null` 슬롯을 잘라낸다 — 명세의 레이아웃 동치 비교가 이것들을 무시한다.
    private static func trimmingTrailingNulls(_ formats: [WGPUTextureFormat?]) -> [WGPUTextureFormat?] {
        var trimmed = formats
        while let last = trimmed.last, last == nil { trimmed.removeLast() }
        return trimmed
    }
}
