import Foundation
import LynxWebGPUCore

/// A WGSL `@group(g) @binding(b)` coordinate.
public struct WGSLBindingKey: Hashable, Sendable {
    public let group: Int
    public let binding: Int

    public init(group: Int, binding: Int) {
        self.group = group
        self.binding = binding
    }
}

/// `@group/@binding` → Metal argument table index assignment.
///
/// WebGPU uses a 2D (group, binding) coordinate while Metal uses a per-stage 1D index.
/// This table is built when the pipeline layout is settled, and **shader emission and encoding read
/// the same table**. That is why MSL is generated at **pipeline creation**, not shader module
/// creation (the same reason Dawn recompiles a shader per pipeline layout).
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

    /// Cache key — the same assignment yields the same MSL, so the compile result can be reused.
    public var signature: String {
        indices
            .sorted { ($0.key.group, $0.key.binding) < ($1.key.group, $1.key.binding) }
            .map { "\($0.key.group):\($0.key.binding)=\($0.value)" }
            .joined(separator: ",")
    }
}

/// The physical limits of Metal's argument table and this implementation's assignment rules.
public enum WGSLMetalLimits {
    /// Metal buffer argument table size (0...30).
    public static let bufferSlotCount = 31
    public static let textureSlotCount = 31
    public static let samplerSlotCount = 16

    /// Maximum number of slots vertex buffers use.
    public static let maxVertexBufferSlots = 8

    /// Vertex buffers are assigned from the **top** of the table downwards (the same rule as Dawn).
    /// Bind group buffers climb from 0, so the two do not collide until they meet.
    public static func vertexBufferIndex(slot: Int) -> Int {
        bufferSlotCount - 1 - slot
    }

    /// The reserved index used by the buffer size table for `arrayLength()`.
    ///
    /// A Metal shader cannot know a buffer's size. So a small table holding the byte size of each
    /// bound buffer is plugged in at this index, and `arrayLength(&x)` is translated into a lookup in
    /// that table (Dawn does the same).
    public static let bufferSizesIndex = bufferSlotCount - maxVertexBufferSlots - 1

    /// Cap on buffer indices available to bind groups (excluding the vertex buffer region and the size table).
    public static let maxBindGroupBuffers = bufferSizesIndex
}

public enum WGSLBindingAssigner {
    /// Assigns Metal indices from the bind group layouts (in group index order).
    ///
    /// Walks group then binding in ascending order, bumping a per-kind counter each time.
    /// **Deterministic**, so the same layout always yields the same MSL.
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
                        "bind group buffers exceeded \(WGSLMetalLimits.maxBindGroupBuffers) "
                            + "(the Metal buffer argument table limit — \(WGSLMetalLimits.maxVertexBufferSlots) slots reserved for vertex buffers)"
                    )
                case .texture where index >= WGSLMetalLimits.textureSlotCount:
                    throw WGPUError.validation("bind group textures exceeded \(WGSLMetalLimits.textureSlotCount)")
                case .sampler where index >= WGSLMetalLimits.samplerSlotCount:
                    throw WGPUError.validation("bind group samplers exceeded \(WGSLMetalLimits.samplerSlotCount)")
                default:
                    break
                }
                assignment.set(group: groupIndex, binding: entry.binding, index: index)
            }
        }
        return assignment
    }
}
