import Foundation
import LynxWebGPUCore

/// Memory layout calculator for WGSL host-shared (uniform/storage) types.
///
/// **Why it is needed** — WGSL and MSL layout rules are nearly identical but diverge on `vec3`.
/// WGSL `vec3<f32>` has alignment 16 and size 12, so `struct { pos: vec3f, r: f32 }` is 16 bytes;
/// MSL `float3` has size 16, making the same struct 32 bytes. For JS to fill a uniform buffer by
/// WGSL rules and have it read back correctly, **the MSL struct must be padded to WGSL offsets**.
/// The emitter (`MSLEmitter`) uses this calculation to choose `packed_float3` and insert padding.
///
/// Reference: the WGSL spec's "Memory Layout" (AlignOf/SizeOf/RoundUp).
enum WGSLLayout {
    struct MemberPlacement {
        var name: String
        var type: WGSLType
        var offset: Int
        var size: Int
        var align: Int
        /// A vec3 whose next member starts within offset+16, so `float3` (16 bytes) cannot express it.
        var needsPackedVector: Bool
    }

    struct StructPlacement {
        var members: [MemberPlacement]
        var size: Int
        var align: Int
    }

    /// Every size and alignment the calculator returns is clamped to this (1 TiB) — far past any
    /// buffer a device can bind, and small enough that summing a struct's members cannot overflow
    /// `Int` however large the declared array lengths or `@size`/`@align` values are.
    static let maximumByteSize = 1 << 40

    static func roundUp(_ alignment: Int, _ value: Int) -> Int {
        guard alignment > 0 else { return value }
        return ((value + alignment - 1) / alignment) * alignment
    }

    // MARK: - Alignment / size
    //
    // Every query goes through one `Walk`, which measures each type **once** (alignment and size
    // together) and remembers each struct's placement. Measuring alignment and size separately, as
    // naive recursion does, re-lays out every nested struct twice per level — `S1 { a: S0 }`,
    // `S2 { a: S1 }`, … is then exponential in the depth, and so is a struct holding two copies
    // of the one below it. The parser bounds the depth (`WGSLParser.maximumCompositeNesting`) and
    // rejects recursive types; the walk also refuses to re-enter a struct it is inside, so it stays
    // bounded whatever AST it is handed.

    static func align(of type: WGSLType, module: WGSLModule, uniform: Bool) -> Int {
        Walk(module: module, uniform: uniform).measure(type).align
    }

    static func size(of type: WGSLType, module: WGSLModule, uniform: Bool) -> Int {
        Walk(module: module, uniform: uniform).measure(type).size
    }

    static func arrayStride(element: WGSLType, module: WGSLModule, uniform: Bool) -> Int {
        let walk = Walk(module: module, uniform: uniform)
        return walk.stride(of: walk.measure(element))
    }

    // MARK: - Struct layout

    static func layout(of structure: WGSLStruct, module: WGSLModule, uniform: Bool) -> StructPlacement {
        Walk(module: module, uniform: uniform).layout(of: structure)
    }

    private final class Walk {
        let module: WGSLModule
        let uniform: Bool
        /// Structs currently being laid out — meeting one again would be a cycle.
        private var enclosing: [String] = []
        private var placements: [String: StructPlacement] = [:]

        init(module: WGSLModule, uniform: Bool) {
            self.module = module
            self.uniform = uniform
        }

        func measure(_ type: WGSLType) -> (align: Int, size: Int) {
            switch type {
            case .scalar(let name):
                let size = name == "f16" ? 2 : 4
                return (size, size)
            case .vector(let count, let element):
                let component = measure(element)
                return ((count == 2 ? 2 : 4) * component.align, count * component.size)
            case .matrix(let columns, let rows, let element):
                let column = measure(.vector(size: rows, element: element))
                return (column.align, columns * roundUp(column.align, column.size))
            case .array(let element, let count):
                let measured = measure(element)
                let align = uniform ? roundUp(16, measured.align) : measured.align
                let elementStride = stride(of: measured)
                guard let count, let literal = constantValue(count, module: module) else { return (align, elementStride) }
                let (total, overflow) = literal.multipliedReportingOverflow(by: elementStride)
                return (align, overflow ? maximumByteSize : min(total, maximumByteSize))
            case .atomic:
                return (4, 4)
            case .named(let name):
                if let structure = resolveStruct(name, module: module) {
                    let placement = layout(of: structure)
                    return (placement.align, placement.size)
                }
                // An alias of a non-struct type (`alias M = mat4x4f;`) measures as that type — reading it
                // as 4 bytes put every member after it at the wrong offset.
                guard let target = resolveAlias(name, module: module),
                      !enclosing.contains(name), enclosing.count < WGSLParser.maximumCompositeNesting else {
                    return (4, 4)
                }
                enclosing.append(name)
                defer { enclosing.removeLast() }
                return measure(target)
            case .pointer, .texture, .sampler, .void:
                return (4, 4)
            }
        }

        func stride(of element: (align: Int, size: Int)) -> Int {
            let stride = roundUp(element.align, element.size)
            return uniform ? roundUp(16, stride) : stride
        }

        func layout(of structure: WGSLStruct) -> StructPlacement {
            if let known = placements[structure.name] { return known }
            // A struct inside itself (or nested past the limit) has no layout — stop instead of recursing.
            guard !enclosing.contains(structure.name), enclosing.count < WGSLParser.maximumCompositeNesting else {
                return StructPlacement(members: [], size: 0, align: 1)
            }
            enclosing.append(structure.name)
            defer { enclosing.removeLast() }

            var members: [MemberPlacement] = []
            var offset = 0
            var maximumAlign = 1

            for member in structure.members {
                // A member spelled with an alias is placed (and emitted) as the aliased type, so the
                // vec3 packing below sees `alias V = vec3f;` as the vec3 it is.
                let memberType = resolvingAlias(member.type, module: module)
                var (memberAlign, memberSize) = measure(memberType)
                if let explicit = member.attributes.first(named: "align")?.integerValue { memberAlign = explicit }
                if let explicit = member.attributes.first(named: "size")?.integerValue { memberSize = max(memberSize, explicit) }
                // Attribute values come straight from the source — keep the running offset from overflowing.
                memberAlign = min(memberAlign, maximumByteSize)
                memberSize = min(memberSize, maximumByteSize)

                offset = roundUp(memberAlign, offset)
                members.append(MemberPlacement(
                    name: member.name, type: memberType,
                    offset: offset, size: memberSize, align: memberAlign,
                    needsPackedVector: false
                ))
                offset = min(offset + memberSize, maximumByteSize)
                maximumAlign = max(maximumAlign, memberAlign)
            }

            let structAlign = uniform ? roundUp(16, maximumAlign) : maximumAlign
            // If a member after a vec3 uses bytes 12...15, MSL `float3` (16 bytes) cannot line it up.
            for index in members.indices {
                guard case .vector(3, _) = members[index].type else { continue }
                let end = members[index].offset + 16
                let nextOffset = index + 1 < members.count
                    ? members[index + 1].offset
                    : roundUp(structAlign, offset)
                members[index].needsPackedVector = nextOffset < end
            }

            let placement = StructPlacement(members: members, size: roundUp(structAlign, offset), align: structAlign)
            placements[structure.name] = placement
            return placement
        }
    }

    // MARK: - Helpers

    /// The struct a name denotes — directly or through any chain of aliases (`alias I2 = I1; alias I1 = S;`).
    static func resolveStruct(_ name: String, module: WGSLModule) -> WGSLStruct? {
        if let structure = module.structNamed(name) { return structure }
        guard case .named(let target)? = resolveAlias(name, module: module) else { return nil }
        return module.structNamed(target)
    }

    /// The type an alias finally names, following alias-to-alias hops: a struct comes back as
    /// `.named(structName)`, anything else as the type itself. Nil when `name` is not an alias.
    /// The hops are bounded — the parser rejects alias cycles and chains past the nesting limit,
    /// and this gives up rather than loop if handed one anyway.
    static func resolveAlias(_ name: String, module: WGSLModule) -> WGSLType? {
        guard var type = module.aliases.first(where: { $0.name == name })?.type else { return nil }
        for _ in 0..<WGSLParser.maximumCompositeNesting {
            guard case .named(let next) = type, module.structNamed(next) == nil,
                  let target = module.aliases.first(where: { $0.name == next })?.type else { return type }
            type = target
        }
        return nil
    }

    /// `type` with a top-level alias replaced by what it names (other types unchanged).
    static func resolvingAlias(_ type: WGSLType, module: WGSLModule) -> WGSLType {
        guard case .named(let name) = type, module.structNamed(name) == nil else { return type }
        return resolveAlias(name, module: module) ?? type
    }

    /// Computes a value that must be known at compile time, such as an array length.
    ///
    /// An array length is an override-expression and may sit inside a function, so overrides and
    /// function-local `const`s resolve too. nil when it cannot be computed (`WGSLConstantEvaluator`
    /// says why).
    static func constantValue(_ expression: WGSLExpression, module: WGSLModule) -> Int? {
        try? arrayLength(expression, module: module)
    }

    /// `constantValue` that keeps the reason it failed.
    static func arrayLength(_ expression: WGSLExpression, module: WGSLModule) throws -> Int {
        var evaluator = WGSLConstantEvaluator(module: module, allowsOverrides: true, allowsFunctionConstants: true)
        return try evaluator.evaluate(expression)
    }

    /// Names of structs used in the uniform address space — their layout rules are stricter than storage.
    static func uniformStructNames(_ module: WGSLModule) -> Set<String> {
        structNames(inAddressSpace: "uniform", module: module)
    }

    /// Names of structs used in the storage address space.
    ///
    /// A struct reachable from **both** a uniform and a storage global must be laid out once, by the
    /// address-space-independent rules (the uniform "mode" only rounds strides and struct sizes up
    /// to 16, which a valid uniform struct already satisfies). Applying the uniform rounding to a
    /// struct that a storage buffer also holds made `array<Pair>` stride 16 in MSL while JS filled it
    /// at the spec's 8 — every element after the first read from the wrong place, with no error.
    static func storageStructNames(_ module: WGSLModule) -> Set<String> {
        structNames(inAddressSpace: "storage", module: module)
    }

    private static func structNames(inAddressSpace addressSpace: String, module: WGSLModule) -> Set<String> {
        var names = Set<String>()
        for global in module.globals where global.addressSpace == addressSpace {
            markStructs(in: global.type, module: module, into: &names)
        }
        return names
    }

    private static func markStructs(in type: WGSLType, module: WGSLModule, into names: inout Set<String>) {
        switch type {
        case .named(let name):
            guard let structure = resolveStruct(name, module: module) else {
                // An alias of a non-struct type (`alias Lights = array<Light, 4>;`) — mark what it holds.
                if let target = resolveAlias(name, module: module), target != type {
                    markStructs(in: target, module: module, into: &names)
                }
                return
            }
            // The struct's own name (not an alias of it) is what emission looks up.
            guard names.insert(structure.name).inserted else { return }
            for member in structure.members {
                markStructs(in: member.type, module: module, into: &names)
            }
        case .array(let element, _), .atomic(let element):
            markStructs(in: element, module: module, into: &names)
        case .pointer(_, let element, _):
            markStructs(in: element, module: module, into: &names)
        default:
            break
        }
    }
}

/// The value of a `…ReportingOverflow` result, or nil when it overflowed.
private func checked(_ result: (partialValue: Int, overflow: Bool)) -> Int? {
    result.overflow ? nil : result.partialValue
}
