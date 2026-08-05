import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// WebGPU 열거형 → Metal 열거형 매핑. **GPU가 필요 없다.**
///
/// 조합 폭발을 GPU 테스트로 감당하면 느리기만 하고 얻는 것이 없다. 값 대응은 여기서 전수로 보고,
/// GPU 테스트는 "그 값이 실제로 렌더 결과를 바꾸는가"만 대표 조합으로 확인한다.
final class MetalMappingTests: XCTestCase {
    /// **모든** 텍스처 포맷이 Metal 대응을 갖는지 — 케이스를 늘리고 매핑을 빠뜨리면 여기서 걸린다.
    ///
    /// 빠뜨리면 그 포맷을 쓰는 순간 `unsupported`가 나는데, 열거형에 있으니 지원한다고 믿고
    /// 쓰게 된다. 전수로 도는 것이 요점이다.
    func test_모든_텍스처_포맷이_Metal_대응을_갖는다() throws {
        for format in WGPUTextureFormat.allCases {
            let metal = try WGPUMetalMapping.pixelFormat(format)
            XCTAssertNotEqual(metal, .invalid, "'\(format.rawValue)'의 Metal 대응이 없다")
        }
    }

    /// 역방향 매핑이 **모든** 포맷을 되돌리는지 — 표를 손으로 적던 시절에는 캔버스에 쓰이는
    /// 몇 개만 있어서 `stencil8`·`rgba8snorm` 같은 것이 조용히 nil이었다.
    func test_모든_텍스처_포맷이_역방향으로도_돌아온다() throws {
        for format in WGPUTextureFormat.allCases {
            let metal = try WGPUMetalMapping.pixelFormat(format)
            XCTAssertNotNil(
                WGPUMetalMapping.textureFormat(from: metal),
                "'\(format.rawValue)'을(를) 되돌리지 못한다"
            )
        }
    }

    /// 접히는 자리(`depth24plus`도 `depth32float`도 `.depth32Float`)에서는 **정밀도를
    /// 그대로 말해 주는 쪽**이 나와야 한다. 약하게 들리는 이름이 나오면 진단이 사람을 속인다.
    func test_같은_Metal_포맷으로_접히는_깊이_포맷은_정밀도가_높은_이름으로_돌아온다() {
        XCTAssertEqual(WGPUMetalMapping.textureFormat(from: .depth32Float), .depth32float)
        XCTAssertEqual(WGPUMetalMapping.textureFormat(from: .depth32Float_stencil8), .depth32floatStencil8)
    }

    /// 픽셀당 바이트 수가 Metal이 실제로 쓰는 크기와 맞는지 — 어긋나면 `writeTexture`의
    /// 기본 `bytesPerRow`가 틀려 **오류 없이** 어긋난 행을 올린다.
    func test_팩된_32비트_포맷의_픽셀_크기가_4바이트다() {
        for format: WGPUTextureFormat in [.rgb10a2unorm, .rgb10a2uint, .rg11b10ufloat, .rgb9e5ufloat] {
            XCTAssertEqual(format.bytesPerPixel, 4, "\(format.rawValue)")
        }
    }

    func test_스텐실_연산이_Metal로_전수_매핑된다() {
        let expected: [WGPUStencilOperation: MTLStencilOperation] = [
            .keep: .keep,
            .zero: .zero,
            .replace: .replace,
            .invert: .invert,
            .incrementClamp: .incrementClamp,
            .decrementClamp: .decrementClamp,
            .incrementWrap: .incrementWrap,
            .decrementWrap: .decrementWrap,
        ]
        // CaseIterable로 도는 것이 요점이다 — 케이스를 늘리면 이 표를 채우지 않는 한 실패한다.
        for operation in WGPUStencilOperation.allCases {
            guard let want = expected[operation] else {
                XCTFail("새 스텐실 연산 '\(operation.rawValue)'의 Metal 대응이 이 표에 없다")
                continue
            }
            XCTAssertEqual(WGPUMetalMapping.stencilOperation(operation), want, operation.rawValue)
        }
    }

    func test_비교함수가_Metal로_전수_매핑된다() {
        let expected: [WGPUCompareFunction: MTLCompareFunction] = [
            .never: .never,
            .less: .less,
            .equal: .equal,
            .lessEqual: .lessEqual,
            .greater: .greater,
            .notEqual: .notEqual,
            .greaterEqual: .greaterEqual,
            .always: .always,
        ]
        for function in WGPUCompareFunction.allCases {
            guard let want = expected[function] else {
                XCTFail("새 비교 함수 '\(function.rawValue)'의 Metal 대응이 이 표에 없다")
                continue
            }
            XCTAssertEqual(WGPUMetalMapping.compareFunction(function), want, function.rawValue)
        }
    }

    /// 네 연산이 각각 제 슬롯에 들어가는지 — `failOp`와 `depthFailOp`가 바뀌어도
    /// 같은 값을 쓰면 아무도 모른다. 그래서 넷을 모두 다른 값으로 준다.
    func test_스텐실_디스크립터가_네_연산을_제_슬롯에_넣는다() {
        let descriptor = WGPUMetalMapping.stencilDescriptor(
            WGPUStencilFaceState(
                compare: .greater, failOp: .zero, depthFailOp: .invert, passOp: .replace
            ),
            readMask: 0x0F,
            writeMask: 0xF0
        )

        XCTAssertEqual(descriptor.stencilCompareFunction, .greater)
        XCTAssertEqual(descriptor.stencilFailureOperation, .zero)
        XCTAssertEqual(descriptor.depthFailureOperation, .invert)
        XCTAssertEqual(descriptor.depthStencilPassOperation, .replace)
        XCTAssertEqual(descriptor.readMask, 0x0F)
        XCTAssertEqual(descriptor.writeMask, 0xF0)
    }

    func test_스텐실_마스크_기본값이_32비트_전체를_통과시킨다() {
        let descriptor = WGPUMetalMapping.stencilDescriptor(
            WGPUStencilFaceState(), readMask: 0xFFFF_FFFF, writeMask: 0xFFFF_FFFF
        )
        XCTAssertEqual(descriptor.readMask, UInt32.max)
        XCTAssertEqual(descriptor.writeMask, UInt32.max)
    }
}
