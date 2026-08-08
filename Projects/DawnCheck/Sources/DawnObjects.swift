import Foundation
import CoreGraphics
import WebGPU
import LynxWebGPUCore

// Dawn C 핸들의 Swift 래퍼.
//
// `WGPUObjectRegistry`(Core)는 `AnyObject`를 저장하므로, C의 불투명 포인터를 클래스로 감싼다.
// 래퍼가 참조 +1을 소유하고 `deinit`에서 릴리스한다 — 레지스트리에서 빠지면(destroy·reset)
// Dawn 객체도 따라 죽는 수명 모델이다. 핸들 타입 불일치가 경로 붙은 validation 오류가 되는
// 것도 레지스트리 계약 그대로다 (`handle-type-mismatch` 검사).

final class DawnBufferObject {
    let buffer: WGPUBuffer
    let size: Int
    let usage: LynxWebGPUCore.WGPUBufferUsage
    /// 와이어 계약의 매핑 상태 — `mapAsync`(readBuffer)부터 `unmapBuffer` op까지 true.
    var isMapped = false
    /// Dawn 수준에서 실제로 매핑돼 있는가 (mappedAtCreation·직접 map 경로).
    var dawnMapped = false

    init(buffer: WGPUBuffer, size: Int, usage: LynxWebGPUCore.WGPUBufferUsage) {
        self.buffer = buffer
        self.size = size
        self.usage = usage
    }

    deinit { wgpuBufferRelease(buffer) }
}

final class DawnTextureObject {
    let texture: WGPUTexture
    let format: LynxWebGPUCore.WGPUTextureFormat
    let width: Int
    let height: Int

    /// - Parameter retain: 이미 소유된 참조를 감싸면 false(생성 반환값),
    ///   남의 참조를 나눠 가지면 true(캔버스 텍스처 — 캔버스도 계속 쓴다).
    init(
        texture: WGPUTexture,
        format: LynxWebGPUCore.WGPUTextureFormat,
        width: Int,
        height: Int,
        retain: Bool = false
    ) {
        if retain { wgpuTextureAddRef(texture) }
        self.texture = texture
        self.format = format
        self.width = width
        self.height = height
    }

    deinit { wgpuTextureRelease(texture) }
}

final class DawnTextureViewObject {
    let view: WGPUTextureView
    init(view: WGPUTextureView) { self.view = view }
    deinit { wgpuTextureViewRelease(view) }
}

final class DawnSamplerObject {
    let sampler: WGPUSampler
    init(sampler: WGPUSampler) { self.sampler = sampler }
    deinit { wgpuSamplerRelease(sampler) }
}

final class DawnShaderModuleObject {
    let module: WGPUShaderModule
    init(module: WGPUShaderModule) { self.module = module }
    deinit { wgpuShaderModuleRelease(module) }
}

final class DawnBindGroupLayoutObject {
    let layout: WGPUBindGroupLayout
    init(layout: WGPUBindGroupLayout) { self.layout = layout }
    deinit { wgpuBindGroupLayoutRelease(layout) }
}

final class DawnPipelineLayoutObject {
    let layout: WGPUPipelineLayout
    init(layout: WGPUPipelineLayout) { self.layout = layout }
    deinit { wgpuPipelineLayoutRelease(layout) }
}

final class DawnBindGroupObject {
    let group: WGPUBindGroup
    init(group: WGPUBindGroup) { self.group = group }
    deinit { wgpuBindGroupRelease(group) }
}

final class DawnRenderPipelineObject {
    let pipeline: WGPURenderPipeline
    init(pipeline: WGPURenderPipeline) { self.pipeline = pipeline }
    deinit { wgpuRenderPipelineRelease(pipeline) }
}

final class DawnComputePipelineObject {
    let pipeline: WGPUComputePipeline
    init(pipeline: WGPUComputePipeline) { self.pipeline = pipeline }
    deinit { wgpuComputePipelineRelease(pipeline) }
}

final class DawnQuerySetObject {
    let querySet: WGPUQuerySet
    let type: LynxWebGPUCore.WGPUQueryType
    let count: Int

    init(querySet: WGPUQuerySet, type: LynxWebGPUCore.WGPUQueryType, count: Int) {
        self.querySet = querySet
        self.type = type
        self.count = count
    }

    deinit { wgpuQuerySetRelease(querySet) }
}

final class DawnRenderBundleObject {
    let bundle: WGPURenderBundle
    init(bundle: WGPURenderBundle) { self.bundle = bundle }
    deinit { wgpuRenderBundleRelease(bundle) }
}

/// `createImageBitmap` 결과 — 디코딩된 RGBA8 픽셀 (GPU 객체가 아니라 CPU 바이트).
final class DawnImageBitmapObject {
    let data: Data
    let width: Int
    let height: Int

    init(data: Data, width: Int, height: Int) {
        self.data = data
        self.width = width
        self.height = height
    }
}

/// 오프스크린 캔버스 — 적합성 스위트의 픽셀 통로 (`attachOffscreenCanvas`/`readCanvasPixels`).
///
/// Metal 런타임의 `WGPUOffscreenSurface`와 같은 역할이다: `configure`가 백킹 텍스처를 만들고,
/// `getCurrentTexture`가 그 텍스처를 프레임 스코프 핸들로 내준다.
final class DawnOffscreenCanvas {
    let identifier: String
    private(set) var size: CGSize
    private(set) var format: LynxWebGPUCore.WGPUTextureFormat = .rgba8unorm
    private(set) var texture: WGPUTexture?

    init(identifier: String, size: CGSize) {
        self.identifier = identifier
        self.size = size
    }

    deinit { releaseTexture() }

    func configure(device: WGPUDevice, format: LynxWebGPUCore.WGPUTextureFormat) throws {
        self.format = format
        try remakeTexture(device: device)
    }

    /// 크기 변경 — 이미 configure됐다면 백킹 텍스처를 새 크기로 다시 만든다
    /// (`resize-canvas` 검사의 계약, 오프스크린 Metal 표면과 같은 규칙).
    func updateSize(_ newSize: CGSize, device: WGPUDevice) {
        guard newSize != size else { return }
        size = newSize
        guard texture != nil else { return }
        try? remakeTexture(device: device)
    }

    private func remakeTexture(device: WGPUDevice) throws {
        releaseTexture()
        var descriptor = WGPUTextureDescriptor()
        descriptor.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc
            | WGPUTextureUsage_TextureBinding
        descriptor.dimension = WGPUTextureDimension_2D
        descriptor.size = WebGPU.WGPUExtent3D(
            width: UInt32(size.width), height: UInt32(size.height), depthOrArrayLayers: 1
        )
        descriptor.format = try DawnEnum.textureFormat(format)
        descriptor.mipLevelCount = 1
        descriptor.sampleCount = 1
        guard let texture = wgpuDeviceCreateTexture(device, &descriptor) else {
            throw WGPUError.outOfMemory("오프스크린 캔버스 텍스처 생성 실패")
        }
        self.texture = texture
    }

    private func releaseTexture() {
        if let texture { wgpuTextureRelease(texture) }
        texture = nil
    }
}
