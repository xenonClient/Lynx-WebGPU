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

    func test_스텐실_상태_기본값은_아무것도_하지_않는다() throws {
        let state = try WGPUDepthStencilState(from: WGPUValueReader(["format": "depth24plus-stencil8"]))

        // 명세 기본값 — 비교는 항상 통과, 세 연산은 모두 keep. 그래서 결과에 영향이 없다.
        XCTAssertEqual(state.stencilFront, WGPUStencilFaceState())
        XCTAssertEqual(state.stencilFront.compare, .always)
        XCTAssertEqual(state.stencilFront.failOp, .keep)
        XCTAssertEqual(state.stencilFront.depthFailOp, .keep)
        XCTAssertEqual(state.stencilFront.passOp, .keep)
        XCTAssertEqual(state.stencilBack, WGPUStencilFaceState())
        XCTAssertEqual(state.stencilReadMask, 0xFFFF_FFFF)
        XCTAssertEqual(state.stencilWriteMask, 0xFFFF_FFFF)
        XCTAssertFalse(state.usesStencil, "기본값만 있으면 스텐실 상태를 만들 이유가 없다")
    }

    func test_스텐실_앞뒤면을_따로_읽는다() throws {
        let state = try WGPUDepthStencilState(from: WGPUValueReader([
            "format": "stencil8",
            "stencilFront": ["compare": "equal", "passOp": "replace"],
            "stencilBack": ["compare": "never", "failOp": "increment-wrap"],
            "stencilReadMask": 0x0F,
            "stencilWriteMask": 0,
        ]))

        XCTAssertEqual(state.stencilFront.compare, .equal)
        XCTAssertEqual(state.stencilFront.passOp, .replace)
        XCTAssertEqual(state.stencilFront.failOp, .keep, "주지 않은 필드는 명세 기본값")
        XCTAssertEqual(state.stencilBack.compare, .never)
        XCTAssertEqual(state.stencilBack.failOp, .incrementWrap)
        XCTAssertEqual(state.stencilReadMask, 0x0F)
        XCTAssertEqual(state.stencilWriteMask, 0)
        XCTAssertTrue(state.usesStencil)
    }

    func test_스텐실_연산_철자는_명세_그대로다() {
        // JS가 문자열로 보내므로 철자가 곧 API다 — 바꾸면 조용히 "알 수 없는 값"이 된다.
        XCTAssertEqual(
            WGPUStencilOperation.allCases.map(\.rawValue),
            ["keep", "zero", "replace", "invert",
             "increment-clamp", "decrement-clamp", "increment-wrap", "decrement-wrap"]
        )
    }

    func test_알수없는_스텐실_연산은_후보를_알려준다() {
        XCTAssertThrowsError(try WGPUDepthStencilState(from: WGPUValueReader([
            "format": "stencil8", "stencilFront": ["passOp": "incrementClamp"],
        ]))) { error in
            XCTAssertTrue(
                "\(error)".contains("increment-clamp"),
                "명세 철자를 후보로 보여 줘야 한다: \(error)"
            )
        }
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

    /// 살아 있는 핸들을 덮어쓰는 것은 **언제나 핸들 발급 버그**다 (JS가 번호를 재사용했다).
    ///
    /// 오류로 드러나지 않는 것이 문제다 — 같은 타입끼리 겹치면 조회도 통과하고, "내 버퍼에
    /// 남이 그린다"는 증상만 남는다. 그래서 세어 두고 경고한다.
    func test_살아있는_핸들을_덮어쓰면_센다() {
        let registry = WGPUObjectRegistry()
        XCTAssertEqual(registry.displacedHandleCount, 0)

        registry.insert(Dummy(), at: WGPUHandle(1))
        registry.insert(Dummy(), at: WGPUHandle(2))
        XCTAssertEqual(registry.displacedHandleCount, 0, "다른 핸들은 겹친 것이 아니다")

        registry.insert(Dummy(), at: WGPUHandle(1))
        XCTAssertEqual(registry.displacedHandleCount, 1)
        XCTAssertEqual(registry.count, 2, "덮어쓴 것이므로 개수는 그대로다")
    }

    /// 해제한 뒤 같은 번호를 다시 쓰는 것은 겹친 것이 아니다 — 자리가 비어 있었다.
    func test_해제한_핸들을_다시_쓰는_것은_겹침이_아니다() {
        let registry = WGPUObjectRegistry()
        registry.insert(Dummy(), at: WGPUHandle(1))
        registry.remove(WGPUHandle(1))
        registry.insert(Dummy(), at: WGPUHandle(1))

        XCTAssertEqual(registry.displacedHandleCount, 0)
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

        registry.insert(Dummy(), at: WGPUHandle(0))   // 겹침 카운터도 올려 둔다
        XCTAssertGreaterThan(registry.displacedHandleCount, 0)

        registry.removeAll()
        XCTAssertEqual(registry.lastWarnedThreshold, 0, "새 페이지는 깨끗한 상태에서 시작한다")
        XCTAssertEqual(registry.displacedHandleCount, 0)
    }
}
