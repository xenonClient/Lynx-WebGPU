import Foundation

/// 커맨드 스트림 op 하나의 **디코딩 결과**.
///
/// `WGPUDescriptors.swift`가 명세의 `GPU*Descriptor` 딕셔너리를 옮긴다면, 이 파일은
/// **op 자체의 인자**를 옮긴다 (`draw`의 `vertexCount`, `copyBufferToBuffer`의 오프셋 …).
///
/// 왜 나눠 두는가 — 해석기가 리더에서 직접 필드를 읽으면 **디코딩과 백엔드 인코딩이 한 함수에
/// 붙어 버린다.** 그러면 백엔드를 갈아끼울 때(Dawn 등) 디코딩까지 다시 써야 하고, JS와 Swift가
/// 쓰는 필드 이름이 코드 곳곳에 흩어져 한쪽만 고쳐도 양쪽 다 컴파일되는 드리프트가 생긴다
/// (`CLAUDE.md` — "커맨드 스트림의 필드 이름은 타입 검사가 잡아 주지 않는다").
/// 여기 모아 두면 이름의 출처가 하나이고, 백엔드는 **값만** 받는다.
///
/// 규칙 두 가지:
/// - **객체를 봐야 정해지는 기본값은 여기서 채우지 않는다.** `copyBufferToBuffer`의 `size`
///   (원본의 남은 전부)처럼 레지스트리 조회가 필요한 것은 `nil`로 남기고 해석기가 정한다.
/// - **명세가 정한 값 변환은 여기서 한다.** WebIDL의 `u32` modulo 변환, 워크그룹 수의 하한처럼
///   백엔드와 무관한 규칙은 어느 백엔드를 써도 같아야 한다.
public protocol WGPUCommandFields {
    /// 커맨드 스트림에서의 위치 (`commands[3]`). 뒤늦은 검증이 오류에 붙일 경로의 뿌리다.
    var path: String { get }
}

public extension WGPUCommandFields {
    /// 이 커맨드 아래 필드 하나의 경로 (`commands[3].size`).
    func fieldPath(_ key: String) -> String { path.isEmpty ? key : "\(path).\(key)" }
}

// MARK: - 객체 생성

/// 커맨드 스트림에서 리더 하나로 만들어지는 디스크립터.
///
/// `WGPUCreateCommand`가 이 요구만 보고 디스크립터를 만든다 — 그래서 `create*` op을 하나
/// 더할 때 해석기에 손댈 것이 없다.
public protocol WGPUDecodableDescriptor {
    init(from reader: WGPUValueReader) throws
}

/// `create*` op 하나 — **JS가 발급한 핸들**과 명세 디스크립터의 짝.
///
/// 핸들을 클라이언트가 내는 것이 이 설계의 전제다 (`WGPUHandle` 참고). 그래서 모든 생성 op은
/// 명세에 없는 `id` 필드를 하나 더 싣는다 — 그 이름이 **여기 한 곳**에만 있게 한다.
public struct WGPUCreateCommand<Descriptor: WGPUDecodableDescriptor>: WGPUCommandFields {
    public let id: WGPUHandle
    public let descriptor: Descriptor
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        id = try reader.requiredHandle("id")
        descriptor = try Descriptor(from: reader)
        path = reader.path
    }
}

extension WGPUBufferDescriptor: WGPUDecodableDescriptor {}
extension WGPUTextureDescriptor: WGPUDecodableDescriptor {}
extension WGPUSamplerDescriptor: WGPUDecodableDescriptor {}
extension WGPUShaderModuleDescriptor: WGPUDecodableDescriptor {}
extension WGPUBindGroupLayoutDescriptor: WGPUDecodableDescriptor {}
extension WGPUPipelineLayoutDescriptor: WGPUDecodableDescriptor {}
extension WGPUBindGroupDescriptor: WGPUDecodableDescriptor {}
extension WGPUQuerySetDescriptor: WGPUDecodableDescriptor {}
extension WGPURenderPipelineDescriptor: WGPUDecodableDescriptor {}
extension WGPUComputePipelineDescriptor: WGPUDecodableDescriptor {}

/// `device.createTextureView()` — 원본 텍스처 핸들이 디스크립터 밖에 따로 붙는다.
public struct WGPUCreateTextureViewCommand: WGPUCommandFields {
    public let id: WGPUHandle
    public let texture: WGPUHandle
    public let descriptor: WGPUTextureViewDescriptor
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        id = try reader.requiredHandle("id")
        texture = try reader.requiredHandle("texture")
        descriptor = try WGPUTextureViewDescriptor(from: reader)
        path = reader.path
    }
}

/// `bundleEncoder.finish()` — 명령 목록을 **값으로 저장**했다가 렌더 패스에 되풀이한다.
///
/// 여기만 리더를 그대로 들고 있다. 번들의 계약이 "직접 인코딩과 같은 결과"라 저장 대상이
/// 디코딩된 값이 아니라 **명령 그 자체**이기 때문이다. 리더는 값 타입이라 되풀이해도
/// 원본이 바뀌지 않는다.
public struct WGPUCreateRenderBundleCommand: WGPUCommandFields {
    public let id: WGPUHandle
    public let commands: [WGPUValueReader]
    public let descriptor: WGPURenderBundleDescriptor
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        id = try reader.requiredHandle("id")
        commands = try reader.requiredObjects("commands")
        descriptor = try WGPURenderBundleDescriptor(from: reader)
        path = reader.path
    }
}

// MARK: - 공통 복사 인자

/// 명세 `GPUTexelCopyTextureInfo` — 복사의 텍스처 쪽 끝.
public struct WGPUTexelCopyTextureInfo: WGPUCommandFields {
    public let texture: WGPUHandle
    public let mipLevel: Int
    public let origin: WGPUOrigin3D
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        texture = try reader.requiredHandle("texture")
        mipLevel = reader.int("mipLevel", default: 0)
        origin = try reader.origin("origin")
        path = reader.path
    }
}

/// 명세 `GPUTexelCopyBufferInfo` — 복사의 버퍼 쪽 끝.
///
/// `bytesPerRow`·`rowsPerImage`는 생략할 수 있고, 그때의 기본값은 **텍스처 포맷을 알아야**
/// 나온다 (블록 포맷은 행이 픽셀이 아니라 블록 단위다). 그래서 여기서는 nil로 남긴다.
public struct WGPUTexelCopyBufferInfo: WGPUCommandFields {
    public let buffer: WGPUHandle
    public let offset: Int
    public let bytesPerRow: Int?
    public let rowsPerImage: Int?
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        buffer = try reader.requiredHandle("buffer")
        offset = reader.int("offset", default: 0)
        bytesPerRow = reader.optionalInt("bytesPerRow")
        rowsPerImage = reader.optionalInt("rowsPerImage")
        path = reader.path
    }
}

// MARK: - 리소스 · 큐

/// `queue.writeBuffer()`.
public struct WGPUWriteBufferCommand: WGPUCommandFields {
    public let buffer: WGPUHandle
    public let data: Data
    public let bufferOffset: Int
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        buffer = try reader.requiredHandle("buffer")
        data = try reader.requiredData("data")
        bufferOffset = reader.int("bufferOffset", default: 0)
        path = reader.path
    }
}

/// `queue.writeTexture()`.
public struct WGPUWriteTextureCommand: WGPUCommandFields {
    public let texture: WGPUHandle
    public let data: Data
    public let mipLevel: Int
    public let origin: WGPUOrigin3D
    public let size: WGPUExtent3D
    public let bytesPerRow: Int?
    public let rowsPerImage: Int?
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        texture = try reader.requiredHandle("texture")
        data = try reader.requiredData("data")
        mipLevel = reader.int("mipLevel", default: 0)
        origin = try reader.origin("origin")
        size = try reader.requiredExtent("size")
        bytesPerRow = reader.optionalInt("bytesPerRow")
        rowsPerImage = reader.optionalInt("rowsPerImage")
        path = reader.path
    }
}

/// `queue.copyExternalImageToTexture()`의 소스 (명세 `GPUCopyExternalImageSourceInfo`).
///
/// `flipY`는 **복사 시점**의 뒤집기다 — `createImageBitmap`의 `flipY`(디코딩 시점)와 별개다.
public struct WGPUExternalImageSource: WGPUCommandFields {
    public let image: WGPUHandle
    public let origin: WGPUOrigin3D
    public let flipY: Bool
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        image = try reader.requiredHandle("source")
        origin = try reader.origin("origin")
        flipY = reader.bool("flipY", default: false)
        path = reader.path
    }
}

/// `queue.copyExternalImageToTexture()`.
public struct WGPUCopyExternalImageCommand: WGPUCommandFields {
    public let source: WGPUExternalImageSource
    public let destination: WGPUTexelCopyTextureInfo
    /// 생략하면 **이미지의 남은 부분 전부**다 — 이미지 크기를 알아야 정해지므로 nil로 남긴다.
    public let copySize: WGPUExtent3D?
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        source = try WGPUExternalImageSource(from: try reader.requiredObject("source"))
        destination = try WGPUTexelCopyTextureInfo(from: try reader.requiredObject("destination"))
        copySize = reader.extent("copySize")
        path = reader.path
    }
}

/// 핸들 하나만 받는 op (`destroy`).
public struct WGPUDestroyCommand: WGPUCommandFields {
    public let id: WGPUHandle
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        id = try reader.requiredHandle("id")
        path = reader.path
    }
}

/// `buffer.unmap()`.
public struct WGPUUnmapBufferCommand: WGPUCommandFields {
    public let buffer: WGPUHandle
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        buffer = try reader.requiredHandle("buffer")
        path = reader.path
    }
}

/// `pipeline.getBindGroupLayout(index)`.
public struct WGPUGetBindGroupLayoutCommand: WGPUCommandFields {
    public let id: WGPUHandle
    public let pipeline: WGPUHandle
    public let index: Int
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        id = try reader.requiredHandle("id")
        pipeline = try reader.requiredHandle("pipeline")
        index = try reader.requiredInt("index")
        path = reader.path
    }
}

// MARK: - 오류 스코프

/// `device.pushErrorScope(filter)`.
public struct WGPUPushErrorScopeCommand: WGPUCommandFields {
    public let filter: WGPUErrorFilter
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        path = reader.path
        filter = try reader.requiredEnum("filter", WGPUErrorFilter.self)
    }
}

// MARK: - 캔버스

/// `context.getCurrentTexture()`.
public struct WGPUGetCurrentTextureCommand: WGPUCommandFields {
    public let id: WGPUHandle
    public let canvas: String
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        id = try reader.requiredHandle("id")
        canvas = try reader.requiredString("canvas")
        path = reader.path
    }
}

// MARK: - 패스 상태

/// `pass.setPipeline()` — 렌더/컴퓨트 공용이다 (어느 쪽인지는 열려 있는 패스가 정한다).
public struct WGPUSetPipelineCommand: WGPUCommandFields {
    public let pipeline: WGPUHandle
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        pipeline = try reader.requiredHandle("pipeline")
        path = reader.path
    }
}

/// `pass.setBindGroup()`.
public struct WGPUSetBindGroupCommand: WGPUCommandFields {
    public let index: Int
    public let bindGroup: WGPUHandle
    public let dynamicOffsets: [Int]
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        index = try reader.requiredInt("index")
        bindGroup = try reader.requiredHandle("bindGroup")
        // 못 읽으면 **빈 배열로 본다.** 동적 오프셋이 없는 레이아웃이 압도적으로 많고,
        // 실제로 필요한데 빠졌다면 바인드 그룹 적용에서 "개수가 부족하다"로 잡힌다.
        dynamicOffsets = (try? reader.integers("dynamicOffsets")) ?? []
        path = reader.path
    }
}

/// `pass.setVertexBuffer()`.
public struct WGPUSetVertexBufferCommand: WGPUCommandFields {
    public let slot: Int
    public let buffer: WGPUHandle
    public let offset: Int
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        slot = try reader.requiredInt("slot")
        buffer = try reader.requiredHandle("buffer")
        offset = reader.int("offset", default: 0)
        path = reader.path
    }
}

/// `pass.setIndexBuffer()`.
public struct WGPUSetIndexBufferCommand: WGPUCommandFields {
    public let buffer: WGPUHandle
    public let format: WGPUIndexFormat
    public let offset: Int
    public let path: String

    /// 인덱스 하나의 바이트 수 — `firstIndex`를 바이트 오프셋으로 옮길 때 쓴다.
    public var indexStride: Int { format == .uint16 ? 2 : 4 }

    public init(from reader: WGPUValueReader) throws {
        buffer = try reader.requiredHandle("buffer")
        format = try reader.requiredEnum("format", WGPUIndexFormat.self)
        offset = reader.int("offset", default: 0)
        path = reader.path
    }
}

/// `pass.setViewport()`.
public struct WGPUSetViewportCommand: WGPUCommandFields {
    public let x: Double
    public let y: Double
    public let width: Double
    public let height: Double
    public let minDepth: Double
    public let maxDepth: Double
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        x = reader.double("x", default: 0)
        y = reader.double("y", default: 0)
        width = try reader.requiredDouble("width")
        height = try reader.requiredDouble("height")
        minDepth = reader.double("minDepth", default: 0)
        maxDepth = reader.double("maxDepth", default: 1)
        path = reader.path
    }
}

/// `pass.setScissorRect()`.
public struct WGPUSetScissorRectCommand: WGPUCommandFields {
    public let x: Int
    public let y: Int
    public let width: Int
    public let height: Int
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        x = reader.int("x", default: 0)
        y = reader.int("y", default: 0)
        width = try reader.requiredInt("width")
        height = try reader.requiredInt("height")
        path = reader.path
    }
}

/// `pass.setBlendConstant()`.
public struct WGPUSetBlendConstantCommand: WGPUCommandFields {
    public let color: WGPUColor
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        color = try reader.color("color", default: .transparent)
        path = reader.path
    }
}

/// `pass.setStencilReference()`.
public struct WGPUSetStencilReferenceCommand: WGPUCommandFields {
    /// **WebIDL의 `u32` 변환(modulo)을 여기서 끝낸다.** 비-truncating 이니셜라이저를 쓰면
    /// `setStencilReference(-1)` 한 줄로 Swift 런타임이 트랩한다 — "잘못된 인자로 프로세스를
    /// 죽이지 않는다"는 이 라이브러리의 계약(`WGPUError.swift`)에 어긋난다.
    public let reference: UInt32
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        reference = UInt32(truncatingIfNeeded: reader.int("reference", default: 0))
        path = reader.path
    }
}

// MARK: - 드로우 · 디스패치

/// `pass.draw()`.
public struct WGPUDrawCommand: WGPUCommandFields {
    public let vertexCount: Int
    public let instanceCount: Int
    public let firstVertex: Int
    public let firstInstance: Int
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        vertexCount = try reader.requiredInt("vertexCount")
        instanceCount = reader.int("instanceCount", default: 1)
        firstVertex = reader.int("firstVertex", default: 0)
        firstInstance = reader.int("firstInstance", default: 0)
        path = reader.path
    }
}

/// `pass.drawIndexed()`.
public struct WGPUDrawIndexedCommand: WGPUCommandFields {
    public let indexCount: Int
    public let instanceCount: Int
    public let firstIndex: Int
    public let baseVertex: Int
    public let firstInstance: Int
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        indexCount = try reader.requiredInt("indexCount")
        instanceCount = reader.int("instanceCount", default: 1)
        firstIndex = reader.int("firstIndex", default: 0)
        baseVertex = reader.int("baseVertex", default: 0)
        firstInstance = reader.int("firstInstance", default: 0)
        path = reader.path
    }
}

/// 간접 드로우·디스패치 세 op의 공통 인자 (`drawIndirect` · `drawIndexedIndirect` ·
/// `dispatchWorkgroupsIndirect`). 실제 인자 값은 GPU 버퍼 안에 있어 여기서는 볼 수 없다.
public struct WGPUIndirectCommand: WGPUCommandFields {
    public let indirectBuffer: WGPUHandle
    public let indirectOffset: Int
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        indirectBuffer = try reader.requiredHandle("indirectBuffer")
        indirectOffset = reader.int("indirectOffset", default: 0)
        path = reader.path
    }
}

/// `pass.dispatchWorkgroups()`.
public struct WGPUDispatchWorkgroupsCommand: WGPUCommandFields {
    public let x: Int
    public let y: Int
    public let z: Int
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        // 0을 넘기면 Metal이 빈 그리드로 단언한다. 명세도 0을 허용하지 않으므로 1로 올린다.
        x = max(reader.int("x", default: 1), 1)
        y = max(reader.int("y", default: 1), 1)
        z = max(reader.int("z", default: 1), 1)
        path = reader.path
    }
}

/// `pass.executeBundles()`.
public struct WGPUExecuteBundlesCommand: WGPUCommandFields {
    public let bundles: [WGPUHandle]
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        bundles = try reader.handles("bundles")
        path = reader.path
    }
}

// MARK: - 쿼리

/// `pass.beginOcclusionQuery()`.
public struct WGPUBeginOcclusionQueryCommand: WGPUCommandFields {
    public let queryIndex: Int
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        queryIndex = try reader.requiredInt("queryIndex")
        path = reader.path
    }
}

/// `encoder.resolveQuerySet()`.
public struct WGPUResolveQuerySetCommand: WGPUCommandFields {
    public let querySet: WGPUHandle
    public let firstQuery: Int
    /// 생략하면 **쿼리셋의 남은 전부**다 — 쿼리셋 크기를 알아야 하므로 nil로 남긴다.
    public let queryCount: Int?
    public let destination: WGPUHandle
    public let destinationOffset: Int
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        querySet = try reader.requiredHandle("querySet")
        firstQuery = reader.int("firstQuery", default: 0)
        queryCount = reader.optionalInt("queryCount")
        destination = try reader.requiredHandle("destination")
        destinationOffset = reader.int("destinationOffset", default: 0)
        path = reader.path
    }
}

// MARK: - 복사

/// `encoder.copyBufferToBuffer()`.
public struct WGPUCopyBufferToBufferCommand: WGPUCommandFields {
    public let source: WGPUHandle
    public let sourceOffset: Int
    public let destination: WGPUHandle
    public let destinationOffset: Int
    /// 생략하면 **원본의 남은 전부**다 (명세의 짧은 오버로드). 버퍼 크기를 알아야 하므로 nil.
    public let size: Int?
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        source = try reader.requiredHandle("source")
        sourceOffset = reader.int("sourceOffset", default: 0)
        destination = try reader.requiredHandle("destination")
        destinationOffset = reader.int("destinationOffset", default: 0)
        size = reader.optionalInt("size")
        path = reader.path
    }
}

/// `encoder.clearBuffer()`.
public struct WGPUClearBufferCommand: WGPUCommandFields {
    public let buffer: WGPUHandle
    public let offset: Int
    /// 생략하면 **버퍼 끝까지**다.
    public let size: Int?
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        buffer = try reader.requiredHandle("buffer")
        offset = reader.int("offset", default: 0)
        size = reader.optionalInt("size")
        path = reader.path
    }
}

/// `encoder.copyTextureToBuffer()`.
public struct WGPUCopyTextureToBufferCommand: WGPUCommandFields {
    public let source: WGPUTexelCopyTextureInfo
    public let destination: WGPUTexelCopyBufferInfo
    public let copySize: WGPUExtent3D
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        source = try WGPUTexelCopyTextureInfo(from: try reader.requiredObject("source"))
        destination = try WGPUTexelCopyBufferInfo(from: try reader.requiredObject("destination"))
        copySize = try reader.requiredExtent("copySize")
        path = reader.path
    }
}

/// `encoder.copyBufferToTexture()`.
public struct WGPUCopyBufferToTextureCommand: WGPUCommandFields {
    public let source: WGPUTexelCopyBufferInfo
    public let destination: WGPUTexelCopyTextureInfo
    public let copySize: WGPUExtent3D
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        source = try WGPUTexelCopyBufferInfo(from: try reader.requiredObject("source"))
        destination = try WGPUTexelCopyTextureInfo(from: try reader.requiredObject("destination"))
        copySize = try reader.requiredExtent("copySize")
        path = reader.path
    }
}

/// `encoder.copyTextureToTexture()`.
public struct WGPUCopyTextureToTextureCommand: WGPUCommandFields {
    public let source: WGPUTexelCopyTextureInfo
    public let destination: WGPUTexelCopyTextureInfo
    public let copySize: WGPUExtent3D
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        source = try WGPUTexelCopyTextureInfo(from: try reader.requiredObject("source"))
        destination = try WGPUTexelCopyTextureInfo(from: try reader.requiredObject("destination"))
        copySize = try reader.requiredExtent("copySize")
        path = reader.path
    }
}

// MARK: - 디버그 마커

/// `pushDebugGroup()`.
public struct WGPUPushDebugGroupCommand: WGPUCommandFields {
    public let groupLabel: String
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        groupLabel = try reader.requiredString("groupLabel")
        path = reader.path
    }
}

/// `insertDebugMarker()`.
public struct WGPUInsertDebugMarkerCommand: WGPUCommandFields {
    public let markerLabel: String
    public let path: String

    public init(from reader: WGPUValueReader) throws {
        markerLabel = try reader.requiredString("markerLabel")
        path = reader.path
    }
}
