import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// WebGPU 열거형 → Metal 열거형 매핑. **GPU가 필요 없다.**
///
/// 조합 폭발을 GPU 테스트로 감당하면 느리기만 하고 얻는 것이 없다. 값 대응은 여기서 전수로 보고,
/// GPU 테스트는 "그 값이 실제로 렌더 결과를 바꾸는가"만 대표 조합으로 확인한다.
final class MetalMappingTests: XCTestCase {
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
