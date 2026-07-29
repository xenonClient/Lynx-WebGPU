import Foundation
import LynxWebGPUCore

/// 셰이더가 선언한 바인딩 슬롯 하나.
public struct WGSLResourceInfo: Equatable {
    public let name: String
    public let group: Int
    public let binding: Int
    public let slotKind: WGPUMetalSlotKind
    /// `layout: "auto"` 파이프라인이 쓸 바인드 그룹 레이아웃 항목.
    public let bindingLayout: WGPUBindingLayout
}

/// 진입점 하나.
public struct WGSLEntryPointInfo: Equatable {
    public let name: String
    public let stage: WGSLStage
    /// 컴퓨트 진입점의 `@workgroup_size`.
    public let workgroupSize: (x: Int, y: Int, z: Int)?
    /// 이 진입점이 (호출 그래프를 따라) 실제로 쓰는 모듈 스코프 변수 이름들.
    public let usedGlobals: Set<String>

    public static func == (lhs: WGSLEntryPointInfo, rhs: WGSLEntryPointInfo) -> Bool {
        lhs.name == rhs.name && lhs.stage == rhs.stage && lhs.usedGlobals == rhs.usedGlobals
            && lhs.workgroupSize?.x == rhs.workgroupSize?.x
            && lhs.workgroupSize?.y == rhs.workgroupSize?.y
            && lhs.workgroupSize?.z == rhs.workgroupSize?.z
    }
}

/// 파싱된 WGSL 모듈에서 뽑아낸 파이프라인 생성용 정보.
public struct WGSLShaderReflection {
    public let entryPoints: [WGSLEntryPointInfo]
    public let resources: [WGSLResourceInfo]

    public func entryPoint(named name: String) -> WGSLEntryPointInfo? {
        entryPoints.first { $0.name == name }
    }

    public func resource(named name: String) -> WGSLResourceInfo? {
        resources.first { $0.name == name }
    }

    /// 주어진 진입점들이 실제로 쓰는 리소스만 그룹/바인딩 순으로 돌려준다.
    /// `layout: "auto"`의 바인드 그룹 레이아웃은 여기서 유도된다.
    public func resources(usedBy entryPointNames: [String]) -> [WGSLResourceInfo] {
        var used = Set<String>()
        for name in entryPointNames {
            guard let entryPoint = entryPoint(named: name) else { continue }
            used.formUnion(entryPoint.usedGlobals)
        }
        return resources
            .filter { used.contains($0.name) }
            .sorted { ($0.group, $0.binding) < ($1.group, $1.binding) }
    }

    /// 리소스를 쓰는 진입점의 스테이지를 합쳐 visibility를 만든다.
    public func visibility(of resourceName: String) -> WGPUShaderStage {
        var visibility: WGPUShaderStage = []
        for entryPoint in entryPoints where entryPoint.usedGlobals.contains(resourceName) {
            switch entryPoint.stage {
            case .vertex: visibility.insert(.vertex)
            case .fragment: visibility.insert(.fragment)
            case .compute: visibility.insert(.compute)
            }
        }
        return visibility
    }
}

enum WGSLReflectionBuilder {
    static func build(_ module: WGSLModule) -> WGSLShaderReflection {
        let usage = transitiveGlobalUsage(module)
        let globalNames = Set(module.globals.map(\.name))

        let entryPoints: [WGSLEntryPointInfo] = module.functions.compactMap { function in
            guard let stage = function.stage else { return nil }
            let workgroupSize = function.workgroupSize.map { (x: $0.0, y: $0.1, z: $0.2) }
            return WGSLEntryPointInfo(
                name: function.name,
                stage: stage,
                workgroupSize: workgroupSize,
                usedGlobals: (usage[function.name] ?? []).intersection(globalNames)
            )
        }

        let resources: [WGSLResourceInfo] = module.globals.compactMap { global in
            guard let group = global.group, let binding = global.binding,
                  let layout = bindingLayout(for: global) else { return nil }
            return WGSLResourceInfo(
                name: global.name,
                group: group,
                binding: binding,
                slotKind: layout.metalSlotKind,
                bindingLayout: layout
            )
        }.sorted { ($0.group, $0.binding) < ($1.group, $1.binding) }

        return WGSLShaderReflection(entryPoints: entryPoints, resources: resources)
    }

    /// 함수별로 "직접 참조하는 전역 + 호출한 함수" 를 모은 뒤 고정점까지 전파한다.
    /// 헬퍼 함수가 유니폼을 읽으면 진입점이 그 유니폼을 인자로 넘겨야 하므로 (MSL에는
    /// 가변 전역이 없다) 이 집합이 MSL 파라미터 스레딩의 입력이 된다.
    static func transitiveGlobalUsage(_ module: WGSLModule) -> [String: Set<String>] {
        var direct: [String: Set<String>] = [:]
        var callees: [String: Set<String>] = [:]
        let functionNames = Set(module.functions.map(\.name))

        for function in module.functions {
            var identifiers = Set<String>()
            var calls = Set<String>()
            for statement in function.body {
                collect(statement, identifiers: &identifiers, calls: &calls)
            }
            direct[function.name] = identifiers
            callees[function.name] = calls.intersection(functionNames)
        }

        var result = direct
        var changed = true
        while changed {
            changed = false
            for function in module.functions {
                let calleeSet = callees[function.name] ?? []
                var merged = result[function.name] ?? []
                for callee in calleeSet {
                    merged.formUnion(result[callee] ?? [])
                }
                if merged != result[function.name] {
                    result[function.name] = merged
                    changed = true
                }
            }
        }
        return result
    }

    // MARK: - 식별자 수집

    private static func collect(_ statement: WGSLStatement, identifiers: inout Set<String>, calls: inout Set<String>) {
        switch statement {
        case .letDeclaration(_, _, let value), .constDeclaration(_, _, let value):
            collect(value, identifiers: &identifiers, calls: &calls)
        case .varDeclaration(_, _, let value):
            if let value { collect(value, identifiers: &identifiers, calls: &calls) }
        case .assignment(let target, _, let value):
            collect(target, identifiers: &identifiers, calls: &calls)
            collect(value, identifiers: &identifiers, calls: &calls)
        case .increment(let expression), .decrement(let expression), .expressionStatement(let expression):
            collect(expression, identifiers: &identifiers, calls: &calls)
        case .ifStatement(let condition, let then, let elseBranch):
            collect(condition, identifiers: &identifiers, calls: &calls)
            then.forEach { collect($0, identifiers: &identifiers, calls: &calls) }
            switch elseBranch {
            case .block(let statements)?:
                statements.forEach { collect($0, identifiers: &identifiers, calls: &calls) }
            case .chained(let statement)?:
                collect(statement, identifiers: &identifiers, calls: &calls)
            case nil:
                break
            }
        case .forStatement(let initializer, let condition, let update, let body):
            if let initializer { collect(initializer, identifiers: &identifiers, calls: &calls) }
            if let condition { collect(condition, identifiers: &identifiers, calls: &calls) }
            if let update { collect(update, identifiers: &identifiers, calls: &calls) }
            body.forEach { collect($0, identifiers: &identifiers, calls: &calls) }
        case .whileStatement(let condition, let body):
            collect(condition, identifiers: &identifiers, calls: &calls)
            body.forEach { collect($0, identifiers: &identifiers, calls: &calls) }
        case .loopStatement(let body, let continuing):
            body.forEach { collect($0, identifiers: &identifiers, calls: &calls) }
            continuing?.forEach { collect($0, identifiers: &identifiers, calls: &calls) }
        case .switchStatement(let subject, let cases):
            collect(subject, identifiers: &identifiers, calls: &calls)
            for switchCase in cases {
                switchCase.selectors.forEach { collect($0, identifiers: &identifiers, calls: &calls) }
                switchCase.body.forEach { collect($0, identifiers: &identifiers, calls: &calls) }
            }
        case .returnStatement(let value):
            if let value { collect(value, identifiers: &identifiers, calls: &calls) }
        case .block(let statements):
            statements.forEach { collect($0, identifiers: &identifiers, calls: &calls) }
        case .breakStatement, .continueStatement, .discardStatement:
            break
        }
    }

    private static func collect(_ expression: WGSLExpression, identifiers: inout Set<String>, calls: inout Set<String>) {
        switch expression {
        case .identifier(let name):
            identifiers.insert(name)
        case .unary(_, let operand), .paren(let operand), .addressOf(let operand), .dereference(let operand):
            collect(operand, identifiers: &identifiers, calls: &calls)
        case .binary(_, let left, let right), .index(let left, let right):
            collect(left, identifiers: &identifiers, calls: &calls)
            collect(right, identifiers: &identifiers, calls: &calls)
        case .call(let callee, _, let arguments):
            calls.insert(callee)
            arguments.forEach { collect($0, identifiers: &identifiers, calls: &calls) }
        case .member(let base, _):
            collect(base, identifiers: &identifiers, calls: &calls)
        case .intLiteral, .floatLiteral, .boolLiteral:
            break
        }
    }

    // MARK: - 바인딩 레이아웃 유도

    static func bindingLayout(for global: WGSLGlobalVariable) -> WGPUBindingLayout? {
        switch global.type {
        case .texture(let texture):
            return textureLayout(texture)
        case .sampler(let comparison):
            return .sampler(WGPUSamplerBindingLayout(type: comparison ? .comparison : .filtering))
        default:
            break
        }
        switch global.addressSpace {
        case "uniform":
            return .buffer(WGPUBufferBindingLayout(type: .uniform))
        case "storage":
            let isReadWrite = global.access == "read_write" || global.access == "write"
            return .buffer(WGPUBufferBindingLayout(type: isReadWrite ? .storage : .readOnlyStorage))
        default:
            return nil
        }
    }

    private static func textureLayout(_ texture: WGSLTextureType) -> WGPUBindingLayout {
        let dimension = viewDimension(texture.dimension)
        switch texture.kind {
        case .storage:
            let access: WGPUStorageTextureAccess
            switch texture.access {
            case "read": access = .readOnly
            case "read_write": access = .readWrite
            default: access = .writeOnly
            }
            let format = texture.format.flatMap(WGPUTextureFormat.init(rawValue:)) ?? .rgba8unorm
            return .storageTexture(WGPUStorageTextureBindingLayout(
                access: access, format: format, viewDimension: dimension
            ))
        case .depth, .depthMultisampled:
            return .texture(WGPUTextureBindingLayout(
                sampleType: .depth, viewDimension: dimension, multisampled: texture.kind == .depthMultisampled
            ))
        case .multisampled:
            return .texture(WGPUTextureBindingLayout(
                sampleType: sampleType(texture.sampleType), viewDimension: dimension, multisampled: true
            ))
        case .sampled, .external:
            return .texture(WGPUTextureBindingLayout(
                sampleType: sampleType(texture.sampleType), viewDimension: dimension, multisampled: false
            ))
        }
    }

    private static func sampleType(_ type: WGSLType?) -> WGPUTextureSampleType {
        guard case .scalar(let name)? = type else { return .float }
        switch name {
        case "i32": return .sint
        case "u32": return .uint
        default: return .float
        }
    }

    private static func viewDimension(_ dimension: String) -> WGPUTextureViewDimension {
        switch dimension {
        case "1d": return .oneD
        case "2d_array": return .twoDArray
        case "3d": return .threeD
        case "cube": return .cube
        case "cube_array": return .cubeArray
        default: return .twoD
        }
    }
}
