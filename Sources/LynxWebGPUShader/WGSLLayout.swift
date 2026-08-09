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

    static func roundUp(_ alignment: Int, _ value: Int) -> Int {
        guard alignment > 0 else { return value }
        return ((value + alignment - 1) / alignment) * alignment
    }

    // MARK: - Alignment / size

    static func align(of type: WGSLType, module: WGSLModule, uniform: Bool) -> Int {
        switch type {
        case .scalar(let name):
            return name == "f16" ? 2 : 4
        case .vector(let size, let element):
            let elementAlign = align(of: element, module: module, uniform: uniform)
            return (size == 2 ? 2 : 4) * elementAlign
        case .matrix(_, let rows, let element):
            return align(of: .vector(size: rows, element: element), module: module, uniform: uniform)
        case .array(let element, _):
            let elementAlign = align(of: element, module: module, uniform: uniform)
            return uniform ? roundUp(16, elementAlign) : elementAlign
        case .atomic:
            return 4
        case .named(let name):
            guard let structure = resolveStruct(name, module: module) else { return 4 }
            let placement = layout(of: structure, module: module, uniform: uniform)
            return placement.align
        case .pointer, .texture, .sampler, .void:
            return 4
        }
    }

    static func size(of type: WGSLType, module: WGSLModule, uniform: Bool) -> Int {
        switch type {
        case .scalar(let name):
            return name == "f16" ? 2 : 4
        case .vector(let count, let element):
            return count * size(of: element, module: module, uniform: uniform)
        case .matrix(let columns, let rows, let element):
            let column = WGSLType.vector(size: rows, element: element)
            let stride = roundUp(
                align(of: column, module: module, uniform: uniform),
                size(of: column, module: module, uniform: uniform)
            )
            return columns * stride
        case .array(let element, let count):
            let stride = arrayStride(element: element, module: module, uniform: uniform)
            guard let count, let literal = constantValue(count, module: module) else { return stride }
            return literal * stride
        case .atomic:
            return 4
        case .named(let name):
            guard let structure = resolveStruct(name, module: module) else { return 4 }
            return layout(of: structure, module: module, uniform: uniform).size
        case .pointer, .texture, .sampler, .void:
            return 4
        }
    }

    static func arrayStride(element: WGSLType, module: WGSLModule, uniform: Bool) -> Int {
        let elementAlign = align(of: element, module: module, uniform: uniform)
        let stride = roundUp(elementAlign, size(of: element, module: module, uniform: uniform))
        return uniform ? roundUp(16, stride) : stride
    }

    // MARK: - Struct layout

    static func layout(of structure: WGSLStruct, module: WGSLModule, uniform: Bool) -> StructPlacement {
        var members: [MemberPlacement] = []
        var offset = 0
        var maximumAlign = 1

        for member in structure.members {
            var memberAlign = align(of: member.type, module: module, uniform: uniform)
            var memberSize = size(of: member.type, module: module, uniform: uniform)
            if let explicit = member.attributes.first(named: "align")?.firstInt { memberAlign = explicit }
            if let explicit = member.attributes.first(named: "size")?.firstInt { memberSize = max(memberSize, explicit) }

            offset = roundUp(memberAlign, offset)
            members.append(MemberPlacement(
                name: member.name, type: member.type,
                offset: offset, size: memberSize, align: memberAlign,
                needsPackedVector: false
            ))
            offset += memberSize
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

        return StructPlacement(members: members, size: roundUp(structAlign, offset), align: structAlign)
    }

    // MARK: - Helpers

    static func resolveStruct(_ name: String, module: WGSLModule) -> WGSLStruct? {
        if let structure = module.structNamed(name) { return structure }
        // Follow an alias one step.
        if let alias = module.aliases.first(where: { $0.name == name }), case .named(let target) = alias.type {
            return module.structNamed(target)
        }
        return nil
    }

    /// Computes a value that must be known at compile time, such as an array length. Only integer literals and module constants are supported.
    static func constantValue(_ expression: WGSLExpression, module: WGSLModule) -> Int? {
        switch expression {
        case .intLiteral(let text):
            return Int(text.filter(\.isNumber))
        case .paren(let inner):
            return constantValue(inner, module: module)
        case .identifier(let name):
            if let constant = module.constants.first(where: { $0.name == name }),
               let value = constant.value {
                return constantValue(value, module: module)
            }
            // A `const` declared inside a function is a compile-time constant too (`var a: array<T, maxLayers>;`).
            return uniqueLocalConstant(named: name, module: module)
        case .binary(let op, let left, let right):
            guard let l = constantValue(left, module: module), let r = constantValue(right, module: module) else {
                return nil
            }
            switch op {
            case "+": return l + r
            case "-": return l - r
            case "*": return l * r
            case "/": return r == 0 ? nil : l / r
            default: return nil
            }
        default:
            return nil
        }
    }

    /// The value of a function-local `const` declared under this name anywhere in the module.
    ///
    /// Being local, it is visible only inside its function in principle, but there is no function
    /// context when settling an array size. So it is used **only when the module settles on one
    /// value** — if the same name holds different values per function we cannot tell which, so we
    /// give up and raise the error as before.
    private static func uniqueLocalConstant(named name: String, module: WGSLModule) -> Int? {
        var found: Int?
        for function in module.functions {
            for expression in localConstants(named: name, in: function.body) {
                guard let value = constantValue(expression, module: module) else { return nil }
                if let found, found != value { return nil }
                found = value
            }
        }
        return found
    }

    private static func localConstants(named name: String, in statements: [WGSLStatement]) -> [WGSLExpression] {
        var values: [WGSLExpression] = []
        for statement in statements {
            if case .constDeclaration(let declared, _, let value) = statement, declared == name {
                values.append(value)
            }
            for block in nestedBlocks(of: statement) {
                values.append(contentsOf: localConstants(named: name, in: block))
            }
        }
        return values
    }

    /// Walks nested blocks only (so a local `const` is found wherever it sits).
    private static func nestedBlocks(of statement: WGSLStatement) -> [[WGSLStatement]] {
        switch statement {
        case .ifStatement(_, let then, let elseBranch):
            switch elseBranch {
            case .block(let statements)?: return [then, statements]
            case .chained(let nested)?: return [then, [nested]]
            case nil: return [then]
            }
        case .forStatement(let initializer, _, _, let body):
            return initializer.map { [[$0], body] } ?? [body]
        case .whileStatement(_, let body):
            return [body]
        case .loopStatement(let body, let continuing):
            return continuing.map { [body, $0] } ?? [body]
        case .switchStatement(_, let cases):
            return cases.map(\.body)
        case .block(let statements):
            return [statements]
        default:
            return []
        }
    }

    /// Names of structs used in the uniform address space — their layout rules are stricter than storage.
    static func uniformStructNames(_ module: WGSLModule) -> Set<String> {
        var names = Set<String>()
        for global in module.globals where global.addressSpace == "uniform" {
            markStructs(in: global.type, module: module, into: &names)
        }
        return names
    }

    private static func markStructs(in type: WGSLType, module: WGSLModule, into names: inout Set<String>) {
        switch type {
        case .named(let name):
            guard !names.contains(name), let structure = resolveStruct(name, module: module) else { return }
            names.insert(name)
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
