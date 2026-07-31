import XCTest
@testable import LynxWebGPUCore

/// 디스크립터 디코딩 — 명세 기본값과 검증 규칙.
final class WGPUDescriptorTests: XCTestCase {
    func test_버퍼_디스크립터는_초기데이터_길이로_크기를_유추한다() throws {
        let data = Data([1, 2, 3, 4, 5, 6, 7, 8])
        let descriptor = try WGPUBufferDescriptor(from: WGPUValueReader([
            "usage": 0x20, "data": data.base64EncodedString(),
        ]))

        XCTAssertEqual(descriptor.size, 8)
        XCTAssertEqual(descriptor.initialData, data)
        XCTAssertTrue(descriptor.usage.contains(.vertex))
    }

    func test_초기데이터가_버퍼보다_크면_오류다() {
        XCTAssertThrowsError(try WGPUBufferDescriptor(from: WGPUValueReader([
            "size": 4, "usage": 0x20, "data": Data([1, 2, 3, 4, 5, 6]).base64EncodedString(),
        ])))
    }

    func test_크기0_버퍼는_거부한다() {
        XCTAssertThrowsError(try WGPUBufferDescriptor(from: WGPUValueReader(["size": 0, "usage": 0x20])))
    }

    func test_샘플러_기본값은_명세를_따른다() throws {
        let sampler = try WGPUSamplerDescriptor(from: WGPUValueReader([:]))

        XCTAssertEqual(sampler.addressModeU, .clampToEdge)
        XCTAssertEqual(sampler.magFilter, .nearest)
        XCTAssertEqual(sampler.mipmapFilter, .nearest)
        XCTAssertEqual(sampler.lodMinClamp, 0)
        XCTAssertNil(sampler.compare)
    }

    func test_바인드그룹_레이아웃은_리소스_종류를_정확히_하나_요구한다() throws {
        let buffer = try WGPUBindGroupLayoutEntry(from: WGPUValueReader([
            "binding": 0, "visibility": 0x1, "buffer": ["type": "read-only-storage"],
        ]))
        guard case .buffer(let layout) = buffer.layout else { return XCTFail("버퍼 바인딩이 아니다") }
        XCTAssertEqual(layout.type, .readOnlyStorage)
        XCTAssertEqual(buffer.layout.metalSlotKind, .buffer)

        let texture = try WGPUBindGroupLayoutEntry(from: WGPUValueReader([
            "binding": 1, "visibility": 0x2, "texture": ["sampleType": "float"],
        ]))
        XCTAssertEqual(texture.layout.metalSlotKind, .texture)

        XCTAssertThrowsError(try WGPUBindGroupLayoutEntry(from: WGPUValueReader([
            "binding": 2, "visibility": 0x2,
        ])))
    }

    func test_파이프라인_레이아웃_참조는_auto와_핸들을_모두_받는다() throws {
        if case .auto = try WGPUPipelineLayoutRef(from: WGPUValueReader(["layout": "auto"])) {} else {
            XCTFail("auto여야 한다")
        }
        if case .explicit(let handle) = try WGPUPipelineLayoutRef(from: WGPUValueReader(["layout": 7])) {
            XCTAssertEqual(handle, WGPUHandle(7))
        } else {
            XCTFail("명시적 핸들이어야 한다")
        }
        // 생략하면 auto다.
        if case .auto = try WGPUPipelineLayoutRef(from: WGPUValueReader([:])) {} else {
            XCTFail("생략 시 auto여야 한다")
        }
        XCTAssertThrowsError(try WGPUPipelineLayoutRef(from: WGPUValueReader(["layout": "nope"])))
    }

    func test_깊이스텐실은_깊이포맷만_받는다() {
        XCTAssertNoThrow(try WGPUDepthStencilState(from: WGPUValueReader(["format": "depth32float"])))
        XCTAssertThrowsError(try WGPUDepthStencilState(from: WGPUValueReader(["format": "rgba8unorm"])))
    }

    func test_블렌드_기본값은_교체다() throws {
        let target = try WGPUColorTargetState(from: WGPUValueReader(["format": "bgra8unorm"]))
        XCTAssertNil(target.blend)
        XCTAssertEqual(target.writeMask, .all)

        let blended = try WGPUColorTargetState(from: WGPUValueReader([
            "format": "bgra8unorm",
            "blend": ["color": ["srcFactor": "src-alpha", "dstFactor": "one-minus-src-alpha"]],
        ]))
        XCTAssertEqual(blended.blend?.color.srcFactor, .srcAlpha)
        XCTAssertEqual(blended.blend?.color.operation, .add)
        // alpha를 생략하면 명세 기본값(one/zero)이다.
        XCTAssertEqual(blended.blend?.alpha.srcFactor, .one)
        XCTAssertEqual(blended.blend?.alpha.dstFactor, .zero)
    }

    func test_텍스처포맷의_깊이_스텐실_판별과_바이트수() {
        XCTAssertTrue(WGPUTextureFormat.depth24plusStencil8.isDepthOrStencil)
        XCTAssertTrue(WGPUTextureFormat.depth24plusStencil8.hasDepth)
        XCTAssertTrue(WGPUTextureFormat.depth24plusStencil8.hasStencil)
        XCTAssertFalse(WGPUTextureFormat.rgba8unorm.isDepthOrStencil)

        XCTAssertEqual(WGPUTextureFormat.rgba8unorm.bytesPerPixel, 4)
        XCTAssertEqual(WGPUTextureFormat.rgba32float.bytesPerPixel, 16)
        XCTAssertEqual(WGPUTextureFormat.r8unorm.bytesPerPixel, 1)
    }

    func test_정점포맷_바이트수() {
        XCTAssertEqual(WGPUVertexFormat.float32x3.byteSize, 12)
        XCTAssertEqual(WGPUVertexFormat.unorm8x4.byteSize, 4)
        XCTAssertEqual(WGPUVertexFormat.sint32x4.byteSize, 16)
    }
}

/// 핸들 레지스트리 — 타입 안전성과 수명.
final class WGPUObjectRegistryTests: XCTestCase {
    private final class Dummy {}
    private final class Other {}

    func test_등록과_조회() throws {
        let registry = WGPUObjectRegistry()
        let object = Dummy()
        registry.insert(object, at: WGPUHandle(1))

        XCTAssertTrue(registry.contains(WGPUHandle(1)))
        XCTAssertTrue(try registry.lookup(WGPUHandle(1), as: Dummy.self, kind: "Dummy") === object)
        XCTAssertEqual(registry.count, 1)
    }

    func test_없는_핸들은_설명이_붙은_오류다() {
        let registry = WGPUObjectRegistry()
        XCTAssertThrowsError(try registry.lookup(WGPUHandle(9), as: Dummy.self, kind: "GPUBuffer")) { error in
            let message = (error as? WGPUError)?.message ?? ""
            XCTAssertTrue(message.contains("GPUBuffer"))
            XCTAssertTrue(message.contains("destroy"))
        }
    }

    func test_타입이_다르면_실제_타입을_알려준다() {
        let registry = WGPUObjectRegistry()
        registry.insert(Other(), at: WGPUHandle(2))

        XCTAssertThrowsError(try registry.lookup(WGPUHandle(2), as: Dummy.self, kind: "Dummy")) { error in
            XCTAssertTrue(((error as? WGPUError)?.message ?? "").contains("Other"))
        }
    }

    func test_해제하면_조회가_실패한다() {
        let registry = WGPUObjectRegistry()
        registry.insert(Dummy(), at: WGPUHandle(3))
        registry.remove(WGPUHandle(3))

        XCTAssertFalse(registry.contains(WGPUHandle(3)))
        XCTAssertEqual(registry.count, 0)
    }

    func test_객체가_임계값을_넘으면_경고하고_임계는_두배씩_올라간다() {
        let registry = WGPUObjectRegistry()
        let floor = WGPUObjectRegistry.growthWarningFloor
        XCTAssertEqual(registry.lastWarnedThreshold, 0, "임계 아래에서는 경고하지 않는다")

        for index in 0..<floor {
            registry.insert(Dummy(), at: WGPUHandle(index))
        }
        XCTAssertEqual(registry.lastWarnedThreshold, floor)

        for index in floor..<(floor * 2) {
            registry.insert(Dummy(), at: WGPUHandle(index))
        }
        XCTAssertEqual(registry.lastWarnedThreshold, floor * 2, "같은 임계로 반복 경고하지 않는다")
    }

    func test_removeAll은_경고_임계도_리셋한다() {
        let registry = WGPUObjectRegistry()
        for index in 0..<WGPUObjectRegistry.growthWarningFloor {
            registry.insert(Dummy(), at: WGPUHandle(index))
        }
        XCTAssertGreaterThan(registry.lastWarnedThreshold, 0)

        registry.removeAll()
        XCTAssertEqual(registry.lastWarnedThreshold, 0, "새 페이지는 깨끗한 상태에서 시작한다")
    }
}
