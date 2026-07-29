import Foundation
import LynxWebGPUCore

/// WGSL 호스트 공유(uniform/storage) 타입의 메모리 배치 계산기.
///
/// **왜 필요한가** — WGSL과 MSL의 배치 규칙은 거의 같지만 `vec3`에서 갈린다.
/// WGSL `vec3<f32>`는 정렬 16 / 크기 12라서 `struct { pos: vec3f, r: f32 }`가 16바이트지만,
/// MSL `float3`는 크기가 16이라 같은 구조체가 32바이트가 된다. JS가 WGSL 규칙으로 채운
/// 유니폼 버퍼를 그대로 읽으려면 **MSL 구조체를 WGSL 오프셋에 맞춰 패딩**해야 한다.
/// 방출기(`MSLEmitter`)가 이 계산 결과로 `packed_float3` 선택과 패딩 삽입을 결정한다.
///
/// 참고: WGSL 명세 "Memory Layout" (AlignOf/SizeOf/RoundUp).
enum WGSLLayout {
    struct MemberPlacement {
        var name: String
        var type: WGSLType
        var offset: Int
        var size: Int
        var align: Int
        /// 다음 멤버가 offset+16 안쪽에서 시작해 `float3`(16바이트)로는 표현할 수 없는 vec3.
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

    // MARK: - 정렬 / 크기

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

    // MARK: - 구조체 배치

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
        // vec3 멤버 뒤에 12~15 바이트 구간을 쓰는 멤버가 오면 MSL `float3`(16바이트)로는 자리를 못 맞춘다.
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

    // MARK: - 보조

    static func resolveStruct(_ name: String, module: WGSLModule) -> WGSLStruct? {
        if let structure = module.structNamed(name) { return structure }
        // alias를 한 단계 따라간다.
        if let alias = module.aliases.first(where: { $0.name == name }), case .named(let target) = alias.type {
            return module.structNamed(target)
        }
        return nil
    }

    /// 배열 길이처럼 컴파일 타임에 알아야 하는 값을 계산한다. 정수 리터럴과 모듈 상수만 지원한다.
    static func constantValue(_ expression: WGSLExpression, module: WGSLModule) -> Int? {
        switch expression {
        case .intLiteral(let text):
            return Int(text.filter(\.isNumber))
        case .paren(let inner):
            return constantValue(inner, module: module)
        case .identifier(let name):
            guard let constant = module.constants.first(where: { $0.name == name }),
                  let value = constant.value else { return nil }
            return constantValue(value, module: module)
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

    /// uniform 주소 공간에서 쓰이는 구조체 이름들 — 배치 규칙이 storage보다 엄격하다.
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
