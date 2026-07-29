import Foundation
import LynxWebGPUCore

/// WGSL `@group(g) @binding(b)` 좌표.
public struct WGSLBindingKey: Hashable, Sendable {
    public let group: Int
    public let binding: Int

    public init(group: Int, binding: Int) {
        self.group = group
        self.binding = binding
    }
}

/// `@group/@binding` → Metal 인자 테이블 인덱스 배정.
///
/// WebGPU는 (그룹, 바인딩) 2차원 좌표를 쓰지만 Metal은 스테이지별 1차원 인덱스를 쓴다.
/// 파이프라인 레이아웃이 정해질 때 이 표를 만들고, **셰이더 방출과 인코딩이 같은 표를 본다**.
/// 그래서 MSL 생성은 셰이더 모듈 생성 시점이 아니라 **파이프라인 생성 시점**에 일어난다
/// (Dawn이 셰이더를 파이프라인 레이아웃마다 다시 컴파일하는 것과 같은 이유).
public struct WGSLBindingAssignment: Equatable {
    public private(set) var indices: [WGSLBindingKey: Int] = [:]

    public init(indices: [WGSLBindingKey: Int] = [:]) {
        self.indices = indices
    }

    public mutating func set(group: Int, binding: Int, index: Int) {
        indices[WGSLBindingKey(group: group, binding: binding)] = index
    }

    public func index(group: Int, binding: Int) -> Int? {
        indices[WGSLBindingKey(group: group, binding: binding)]
    }

    /// 캐시 키 — 같은 배정이면 같은 MSL이 나오므로 컴파일 결과를 재사용할 수 있다.
    public var signature: String {
        indices
            .sorted { ($0.key.group, $0.key.binding) < ($1.key.group, $1.key.binding) }
            .map { "\($0.key.group):\($0.key.binding)=\($0.value)" }
            .joined(separator: ",")
    }
}

/// Metal 인자 테이블의 물리적 한계와 이 구현의 배정 규칙.
public enum WGSLMetalLimits {
    /// Metal 버퍼 인자 테이블 크기 (0...30).
    public static let bufferSlotCount = 31
    public static let textureSlotCount = 31
    public static let samplerSlotCount = 16

    /// 정점 버퍼가 쓰는 최대 슬롯 수.
    public static let maxVertexBufferSlots = 8

    /// 정점 버퍼는 테이블 **위쪽**부터 역순으로 배정한다 (Dawn과 같은 규칙).
    /// 바인드 그룹 버퍼는 0부터 올라오므로 둘이 만나기 전까지 충돌하지 않는다.
    public static func vertexBufferIndex(slot: Int) -> Int {
        bufferSlotCount - 1 - slot
    }

    /// 바인드 그룹이 쓸 수 있는 버퍼 인덱스 상한 (정점 버퍼 영역 제외).
    public static let maxBindGroupBuffers = bufferSlotCount - maxVertexBufferSlots
}

public enum WGSLBindingAssigner {
    /// 바인드 그룹 레이아웃(그룹 인덱스 순서)에서 Metal 인덱스를 배정한다.
    ///
    /// 그룹 → 바인딩 오름차순으로 훑으며 종류별 카운터를 하나씩 올린다.
    /// **결정적**이므로 같은 레이아웃이면 항상 같은 MSL이 나온다.
    public static func assign(groups: [[WGPUBindGroupLayoutEntry]]) throws -> WGSLBindingAssignment {
        var assignment = WGSLBindingAssignment()
        var next: [WGPUMetalSlotKind: Int] = [.buffer: 0, .texture: 0, .sampler: 0]

        for (groupIndex, entries) in groups.enumerated() {
            for entry in entries.sorted(by: { $0.binding < $1.binding }) {
                let kind = entry.layout.metalSlotKind
                let index = next[kind, default: 0]
                next[kind] = index + 1

                switch kind {
                case .buffer where index >= WGSLMetalLimits.maxBindGroupBuffers:
                    throw WGPUError.validation(
                        "바인드 그룹 버퍼가 \(WGSLMetalLimits.maxBindGroupBuffers)개를 넘었다 "
                            + "(Metal 버퍼 인자 테이블 상한 — 정점 버퍼 \(WGSLMetalLimits.maxVertexBufferSlots)슬롯 예약)"
                    )
                case .texture where index >= WGSLMetalLimits.textureSlotCount:
                    throw WGPUError.validation("바인드 그룹 텍스처가 \(WGSLMetalLimits.textureSlotCount)개를 넘었다")
                case .sampler where index >= WGSLMetalLimits.samplerSlotCount:
                    throw WGPUError.validation("바인드 그룹 샘플러가 \(WGSLMetalLimits.samplerSlotCount)개를 넘었다")
                default:
                    break
                }
                assignment.set(group: groupIndex, binding: entry.binding, index: index)
            }
        }
        return assignment
    }
}
