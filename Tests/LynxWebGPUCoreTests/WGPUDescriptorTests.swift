import XCTest
@testable import LynxWebGPUCore

/// Descriptor decoding — spec defaults and validation rules.
final class WGPUDescriptorTests: XCTestCase {
    func test_bufferDescriptorInfersSizeFromInitialDataLength() throws {
        let data = Data([1, 2, 3, 4, 5, 6, 7, 8])
        let descriptor = try WGPUBufferDescriptor(from: WGPUValueReader([
            "usage": 0x20, "data": data.base64EncodedString(),
        ]))

        XCTAssertEqual(descriptor.size, 8)
        XCTAssertEqual(descriptor.initialData, data)
        XCTAssertTrue(descriptor.usage.contains(.vertex))
    }

    func test_initialDataLargerThanTheBufferIsAnError() {
        XCTAssertThrowsError(try WGPUBufferDescriptor(from: WGPUValueReader([
            "size": 4, "usage": 0x20, "data": Data([1, 2, 3, 4, 5, 6]).base64EncodedString(),
        ])))
    }

    func test_aZeroSizeBufferIsRejected() {
        XCTAssertThrowsError(try WGPUBufferDescriptor(from: WGPUValueReader(["size": 0, "usage": 0x20])))
    }

    func test_samplerDefaultsFollowTheSpec() throws {
        let sampler = try WGPUSamplerDescriptor(from: WGPUValueReader([:]))

        XCTAssertEqual(sampler.addressModeU, .clampToEdge)
        XCTAssertEqual(sampler.magFilter, .nearest)
        XCTAssertEqual(sampler.mipmapFilter, .nearest)
        XCTAssertEqual(sampler.lodMinClamp, 0)
        XCTAssertNil(sampler.compare)
    }

    func test_theStencilStateDefaultDoesNothing() throws {
        let state = try WGPUDepthStencilState(from: WGPUValueReader(["format": "depth24plus-stencil8"]))

        // The spec defaults — compare always passes and all three ops are keep, so it has no effect.
        XCTAssertEqual(state.stencilFront, WGPUStencilFaceState())
        XCTAssertEqual(state.stencilFront.compare, .always)
        XCTAssertEqual(state.stencilFront.failOp, .keep)
        XCTAssertEqual(state.stencilFront.depthFailOp, .keep)
        XCTAssertEqual(state.stencilFront.passOp, .keep)
        XCTAssertEqual(state.stencilBack, WGPUStencilFaceState())
        XCTAssertEqual(state.stencilReadMask, 0xFFFF_FFFF)
        XCTAssertEqual(state.stencilWriteMask, 0xFFFF_FFFF)
        XCTAssertFalse(state.usesStencil, "with only defaults there is no reason to build stencil state")
    }

    func test_readsStencilFrontAndBackSeparately() throws {
        let state = try WGPUDepthStencilState(from: WGPUValueReader([
            "format": "stencil8",
            "stencilFront": ["compare": "equal", "passOp": "replace"],
            "stencilBack": ["compare": "never", "failOp": "increment-wrap"],
            "stencilReadMask": 0x0F,
            "stencilWriteMask": 0,
        ]))

        XCTAssertEqual(state.stencilFront.compare, .equal)
        XCTAssertEqual(state.stencilFront.passOp, .replace)
        XCTAssertEqual(state.stencilFront.failOp, .keep, "fields not supplied take the spec default")
        XCTAssertEqual(state.stencilBack.compare, .never)
        XCTAssertEqual(state.stencilBack.failOp, .incrementWrap)
        XCTAssertEqual(state.stencilReadMask, 0x0F)
        XCTAssertEqual(state.stencilWriteMask, 0)
        XCTAssertTrue(state.usesStencil)
    }

    func test_stencilOpSpellingsMatchTheSpec() {
        // JS sends strings, so the spelling is the API — change it and it silently becomes "unknown value".
        XCTAssertEqual(
            WGPUStencilOperation.allCases.map(\.rawValue),
            ["keep", "zero", "replace", "invert",
             "increment-clamp", "decrement-clamp", "increment-wrap", "decrement-wrap"]
        )
    }

    func test_anUnknownStencilOpListsTheCandidates() {
        XCTAssertThrowsError(try WGPUDepthStencilState(from: WGPUValueReader([
            "format": "stencil8", "stencilFront": ["passOp": "incrementClamp"],
        ]))) { error in
            XCTAssertTrue(
                "\(error)".contains("increment-clamp"),
                "it must offer the spec spellings as candidates: \(error)"
            )
        }
    }

    func test_bindGroupLayoutRequiresExactlyOneResourceKind() throws {
        let buffer = try WGPUBindGroupLayoutEntry(from: WGPUValueReader([
            "binding": 0, "visibility": 0x1, "buffer": ["type": "read-only-storage"],
        ]))
        guard case .buffer(let layout) = buffer.layout else { return XCTFail("not a buffer binding") }
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

    func test_pipelineLayoutRefAcceptsBothAutoAndAHandle() throws {
        if case .auto = try WGPUPipelineLayoutRef(from: WGPUValueReader(["layout": "auto"])) {} else {
            XCTFail("should be auto")
        }
        if case .explicit(let handle) = try WGPUPipelineLayoutRef(from: WGPUValueReader(["layout": 7])) {
            XCTAssertEqual(handle, WGPUHandle(7))
        } else {
            XCTFail("should be an explicit handle")
        }
        // Omitted means auto.
        if case .auto = try WGPUPipelineLayoutRef(from: WGPUValueReader([:])) {} else {
            XCTFail("omitted should mean auto")
        }
        XCTAssertThrowsError(try WGPUPipelineLayoutRef(from: WGPUValueReader(["layout": "nope"])))
    }

    func test_depthStencilAcceptsOnlyDepthFormats() {
        XCTAssertNoThrow(try WGPUDepthStencilState(from: WGPUValueReader(["format": "depth32float"])))
        XCTAssertThrowsError(try WGPUDepthStencilState(from: WGPUValueReader(["format": "rgba8unorm"])))
    }

    func test_theBlendDefaultIsReplace() throws {
        let target = try WGPUColorTargetState(from: WGPUValueReader(["format": "bgra8unorm"]))
        XCTAssertNil(target.blend)
        XCTAssertEqual(target.writeMask, .all)

        let blended = try WGPUColorTargetState(from: WGPUValueReader([
            "format": "bgra8unorm",
            "blend": ["color": ["srcFactor": "src-alpha", "dstFactor": "one-minus-src-alpha"]],
        ]))
        XCTAssertEqual(blended.blend?.color.srcFactor, .srcAlpha)
        XCTAssertEqual(blended.blend?.color.operation, .add)
        // Omitting alpha takes the spec default (one/zero).
        XCTAssertEqual(blended.blend?.alpha.srcFactor, .one)
        XCTAssertEqual(blended.blend?.alpha.dstFactor, .zero)
    }

    func test_textureFormatDepthStencilDetectionAndByteSizes() {
        XCTAssertTrue(WGPUTextureFormat.depth24plusStencil8.isDepthOrStencil)
        XCTAssertTrue(WGPUTextureFormat.depth24plusStencil8.hasDepth)
        XCTAssertTrue(WGPUTextureFormat.depth24plusStencil8.hasStencil)
        XCTAssertFalse(WGPUTextureFormat.rgba8unorm.isDepthOrStencil)

        XCTAssertEqual(WGPUTextureFormat.rgba8unorm.bytesPerPixel, 4)
        XCTAssertEqual(WGPUTextureFormat.rgba32float.bytesPerPixel, 16)
        XCTAssertEqual(WGPUTextureFormat.r8unorm.bytesPerPixel, 1)
    }

    func test_vertexFormatByteSizes() {
        XCTAssertEqual(WGPUVertexFormat.float32x3.byteSize, 12)
        XCTAssertEqual(WGPUVertexFormat.unorm8x4.byteSize, 4)
        XCTAssertEqual(WGPUVertexFormat.sint32x4.byteSize, 16)
    }
}

/// The handle registry — type safety and lifetime.
final class WGPUObjectRegistryTests: XCTestCase {
    private final class Dummy {}
    private final class Other {}

    func test_insertAndLookup() throws {
        let registry = WGPUObjectRegistry()
        let object = Dummy()
        registry.insert(object, at: WGPUHandle(1))

        XCTAssertTrue(registry.contains(WGPUHandle(1)))
        XCTAssertTrue(try registry.lookup(WGPUHandle(1), as: Dummy.self, kind: "Dummy") === object)
        XCTAssertEqual(registry.count, 1)
    }

    func test_aMissingHandleIsAnErrorWithAnExplanation() {
        let registry = WGPUObjectRegistry()
        XCTAssertThrowsError(try registry.lookup(WGPUHandle(9), as: Dummy.self, kind: "GPUBuffer")) { error in
            let message = (error as? WGPUError)?.message ?? ""
            XCTAssertTrue(message.contains("GPUBuffer"))
            XCTAssertTrue(message.contains("destroy"))
        }
    }

    func test_aTypeMismatchReportsTheActualType() {
        let registry = WGPUObjectRegistry()
        registry.insert(Other(), at: WGPUHandle(2))

        XCTAssertThrowsError(try registry.lookup(WGPUHandle(2), as: Dummy.self, kind: "Dummy")) { error in
            XCTAssertTrue(((error as? WGPUError)?.message ?? "").contains("Other"))
        }
    }

    func test_lookupFailsAfterRemoval() {
        let registry = WGPUObjectRegistry()
        registry.insert(Dummy(), at: WGPUHandle(3))
        registry.remove(WGPUHandle(3))

        XCTAssertFalse(registry.contains(WGPUHandle(3)))
        XCTAssertEqual(registry.count, 0)
    }

    /// Overwriting a live handle is **always a handle issuance bug** (JS reused a number).
    ///
    /// The problem is that it never surfaces as an error — same-typed collisions pass lookup too, leaving
    /// only the symptom "someone else draws into my buffer". So we count it and warn.
    func test_overwritingALiveHandleIsCounted() {
        let registry = WGPUObjectRegistry()
        XCTAssertEqual(registry.displacedHandleCount, 0)

        registry.insert(Dummy(), at: WGPUHandle(1))
        registry.insert(Dummy(), at: WGPUHandle(2))
        XCTAssertEqual(registry.displacedHandleCount, 0, "a different handle is not a collision")

        registry.insert(Dummy(), at: WGPUHandle(1))
        XCTAssertEqual(registry.displacedHandleCount, 1)
        XCTAssertEqual(registry.count, 2, "it was an overwrite, so the count is unchanged")
    }

    /// Reusing a number after removal is not a collision — the slot was empty.
    func test_reusingAReleasedHandleIsNotADisplacement() {
        let registry = WGPUObjectRegistry()
        registry.insert(Dummy(), at: WGPUHandle(1))
        registry.remove(WGPUHandle(1))
        registry.insert(Dummy(), at: WGPUHandle(1))

        XCTAssertEqual(registry.displacedHandleCount, 0)
    }

    func test_crossingTheObjectThresholdWarnsAndDoublesIt() {
        let registry = WGPUObjectRegistry()
        let floor = WGPUObjectRegistry.growthWarningFloor
        XCTAssertEqual(registry.lastWarnedThreshold, 0, "below the threshold it does not warn")

        for index in 0..<floor {
            registry.insert(Dummy(), at: WGPUHandle(index))
        }
        XCTAssertEqual(registry.lastWarnedThreshold, floor)

        for index in floor..<(floor * 2) {
            registry.insert(Dummy(), at: WGPUHandle(index))
        }
        XCTAssertEqual(registry.lastWarnedThreshold, floor * 2, "it does not warn repeatedly at the same threshold")
    }

    func test_removeAllResetsTheWarningThresholdToo() {
        let registry = WGPUObjectRegistry()
        for index in 0..<WGPUObjectRegistry.growthWarningFloor {
            registry.insert(Dummy(), at: WGPUHandle(index))
        }
        XCTAssertGreaterThan(registry.lastWarnedThreshold, 0)

        registry.insert(Dummy(), at: WGPUHandle(0))   // bump the displacement counter too
        XCTAssertGreaterThan(registry.displacedHandleCount, 0)

        registry.removeAll()
        XCTAssertEqual(registry.lastWarnedThreshold, 0, "a new page starts from a clean state")
        XCTAssertEqual(registry.displacedHandleCount, 0)
    }
}
