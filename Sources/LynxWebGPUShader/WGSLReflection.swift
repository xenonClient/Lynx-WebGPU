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
    ///
    /// 수집은 **스코프를 따진다** — 매개변수나 지역 선언(`var`/`let`/`const`)에 가려진
    /// 이름은 전역 사용이 아니다. 기계 생성 셰이더(Three.js 노드 시스템 등)는 같은 이름을
    /// 모듈 스코프와 함수 지역에 일상적으로 함께 쓰므로, 이를 무시하면 쓰지도 않는 전역이
    /// 인자로 주입되어 지역 선언과 재정의 충돌을 일으킨다.
    static func transitiveGlobalUsage(_ module: WGSLModule) -> [String: Set<String>] {
        var direct: [String: Set<String>] = [:]
        var callees: [String: Set<String>] = [:]
        let functionNames = Set(module.functions.map(\.name))

        for function in module.functions {
            var identifiers = Set<String>()
            var calls = Set<String>()
            collect(
                function.body, locals: Set(function.parameters.map(\.name)),
                identifiers: &identifiers, calls: &calls
            )
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

    /// 주어진 진입점들에서 **호출로 닿는** 함수 이름들 (진입점 자신 포함).
    ///
    /// 방출기가 이 집합만 내보낸다. 모듈의 함수를 전부 내보내면, 그 진입점이 부르지도 않는
    /// 함수가 참조하는 리소스까지 필요해진다 — `layout: "auto"`의 바인드 그룹은 **쓰는 것만**
    /// 담으므로 그런 함수는 없는 바인딩을 찾다 실패한다. 실제로 `common.wgsl`을 여럿이
    /// 나눠 쓰는 셰이더(webgpu-samples의 cornell)가 이 경로로 깨졌다.
    ///
    /// 죽은 코드를 빼면 Metal 컴파일도 그만큼 빨라진다.
    static func functionsReachable(from entryPoints: [String], in module: WGSLModule) -> Set<String> {
        var callees: [String: Set<String>] = [:]
        let functionNames = Set(module.functions.map(\.name))

        for function in module.functions {
            var identifiers = Set<String>()
            var calls = Set<String>()
            collect(
                function.body, locals: Set(function.parameters.map(\.name)),
                identifiers: &identifiers, calls: &calls
            )
            callees[function.name] = calls.intersection(functionNames)
        }

        var reachable = Set<String>()
        var pending = entryPoints.filter(functionNames.contains)
        while let name = pending.popLast() {
            guard reachable.insert(name).inserted else { continue }
            pending.append(contentsOf: callees[name] ?? [])
        }
        return reachable
    }

    /// 버퍼 크기 표를 인자로 받아야 하는 함수 이름들 (전이적).
    ///
    /// 두 가지 이유로 필요하다:
    /// 1. `arrayLength()` — 원소 수를 표에서 읽는다.
    /// 2. **런타임 크기 배열 인덱싱** — 상한을 알아야 범위를 자를 수 있다 (robustness).
    ///
    /// **방출기와 파이프라인이 반드시 같은 답을 봐야 한다.** 방출기가 표를 요구했는데
    /// 파이프라인이 안 묶으면 셰이더가 바인딩되지 않은 버퍼를 읽는다 — 그래서 계산이 여기 하나뿐이다.
    static func functionsNeedingBufferSizes(in module: WGSLModule) -> Set<String> {
        let runtimeArrays = runtimeArrayGlobalNames(in: module)
        return functionsMatching(in: module) { identifiers, calls in
            calls.contains("arrayLength") || !identifiers.isDisjoint(with: runtimeArrays)
        }
    }

    /// 런타임 크기 배열(`array<T>`)을 담고 있는 전역 이름들 — 직접이든 구조체 멤버로든.
    private static func runtimeArrayGlobalNames(in module: WGSLModule) -> Set<String> {
        var names = Set<String>()
        for global in module.globals {
            switch global.type {
            case .array(_, nil):
                names.insert(global.name)
            case .named(let structName):
                guard let structure = module.structNamed(structName) else { continue }
                if structure.members.contains(where: {
                    if case .array(_, nil) = $0.type { return true } else { return false }
                }) {
                    names.insert(global.name)
                }
            default:
                break
            }
        }
        return names
    }

    /// `discard` 이후의 쓰기를 막는 플래그를 인자로 받아야 하는 함수들 (전이적).
    ///
    /// MSL의 `discard_fragment()`는 즉시 종료가 **아니다** — "이 프래그먼트는 버린다"고 표시할 뿐
    /// 뒤의 코드가 계속 돌아서 버려진 프래그먼트가 스토리지 버퍼·텍스처를 오염시킨다.
    /// 명세는 이를 금지하므로 `discard`를 플래그로 바꾸고 쓰기를 그 플래그로 가린다.
    ///
    /// 플래그는 진입점에서 선언해 호출 그래프를 따라 내려가므로, **discard를 놓는 함수**와
    /// **가려야 할 쓰기가 있는 함수**, 그리고 그것들을 부르는 함수가 전부 필요하다.
    /// 모듈에 `discard`가 없으면 빈 집합이다 — 비용이 0이어야 한다.
    static func functionsNeedingDiscardFlag(in module: WGSLModule) -> Set<String> {
        let discardsAnywhere = module.functions.contains { bodyDiscards($0.body) }
        guard discardsAnywhere else { return [] }

        let storageGlobals = Set(module.globals.filter { $0.addressSpace == "storage" }.map(\.name))
        let seed = Set(module.functions
            .filter { bodyDiscards($0.body) || bodyWritesResources($0.body, storage: storageGlobals) }
            .map(\.name))
        return closure(seed: seed, in: module)
    }

    /// 시드 함수들과, 그들을 (전이적으로) 부르는 함수들.
    private static func closure(seed: Set<String>, in module: WGSLModule) -> Set<String> {
        let functionNames = Set(module.functions.map(\.name))
        var callees: [String: Set<String>] = [:]
        for function in module.functions {
            var identifiers = Set<String>()
            var calls = Set<String>()
            collect(
                function.body, locals: Set(function.parameters.map(\.name)),
                identifiers: &identifiers, calls: &calls
            )
            callees[function.name] = calls.intersection(functionNames)
        }
        var result = seed
        var changed = true
        while changed {
            changed = false
            for function in module.functions where !result.contains(function.name) {
                if (callees[function.name] ?? []).contains(where: result.contains) {
                    result.insert(function.name)
                    changed = true
                }
            }
        }
        return result
    }

    private static func bodyDiscards(_ statements: [WGSLStatement]) -> Bool {
        statements.contains { statement in
            if case .discardStatement = statement { return true }
            return nestedBlocks(of: statement).contains(where: bodyDiscards)
        }
    }

    /// 이 본문이 스토리지 버퍼·텍스처·원자 연산에 **쓰는가** — `discard` 뒤에 가려야 할 것들.
    private static func bodyWritesResources(_ statements: [WGSLStatement], storage: Set<String>) -> Bool {
        statements.contains { statement in
            switch statement {
            case .assignment(let target, _, _):
                if let root = rootIdentifier(of: target), storage.contains(root) { return true }
            case .increment(let target), .decrement(let target):
                if let root = rootIdentifier(of: target), storage.contains(root) { return true }
            case .expressionStatement(let expression):
                if case .call(let callee, _, _) = expression {
                    if callee == "textureStore" { return true }
                    if callee.hasPrefix("atomic"), callee != "atomicLoad" { return true }
                }
            default:
                break
            }
            return nestedBlocks(of: statement).contains { bodyWritesResources($0, storage: storage) }
        }
    }

    /// 문장 안에 들어 있는 하위 블록들 (제어 구조를 따라 내려가기 위한 것).
    private static func nestedBlocks(of statement: WGSLStatement) -> [[WGSLStatement]] {
        switch statement {
        case .block(let inner):
            return [inner]
        case .ifStatement(_, let then, let elseBranch):
            var blocks = [then]
            switch elseBranch {
            case .block(let inner)?: blocks.append(inner)
            case .chained(let nested)?: blocks.append([nested])
            case nil: break
            }
            return blocks
        case .forStatement(_, _, _, let body), .whileStatement(_, let body):
            return [body]
        case .loopStatement(let body, let continuing):
            return continuing.map { [body, $0] } ?? [body]
        case .switchStatement(_, let cases):
            return cases.map(\.body)
        default:
            return []
        }
    }

    /// 대입 대상의 뿌리 식별자 (`out[i].x` → `out`).
    private static func rootIdentifier(of expression: WGSLExpression) -> String? {
        switch expression {
        case .identifier(let name): return name
        case .index(let base, _), .member(let base, _), .dereference(let base), .paren(let base):
            return rootIdentifier(of: base)
        default: return nil
        }
    }

    /// 특정 내장 함수를 (전이적으로) 호출하는 함수 이름들.
    ///
    /// `arrayLength`처럼 **추가 인자가 필요한** 내장 함수를 위해 쓴다 —
    /// 그 인자를 진입점에서 받아 호출 그래프를 따라 내려보내야 하기 때문이다.
    static func functionsCalling(_ builtin: String, in module: WGSLModule) -> Set<String> {
        functionsMatching(in: module) { _, calls in calls.contains(builtin) }
    }

    /// 조건을 만족하는 함수와, 그 함수를 (전이적으로) 부르는 함수들.
    private static func functionsMatching(
        in module: WGSLModule,
        seed: (_ identifiers: Set<String>, _ calls: Set<String>) -> Bool
    ) -> Set<String> {
        var callees: [String: Set<String>] = [:]
        var directly = Set<String>()
        let functionNames = Set(module.functions.map(\.name))

        for function in module.functions {
            var identifiers = Set<String>()
            var calls = Set<String>()
            collect(
                function.body, locals: Set(function.parameters.map(\.name)),
                identifiers: &identifiers, calls: &calls
            )
            if seed(identifiers, calls) { directly.insert(function.name) }
            callees[function.name] = calls.intersection(functionNames)
        }

        var result = directly
        var changed = true
        while changed {
            changed = false
            for function in module.functions where !result.contains(function.name) {
                if (callees[function.name] ?? []).contains(where: result.contains) {
                    result.insert(function.name)
                    changed = true
                }
            }
        }
        return result
    }

    // MARK: - 식별자 수집 (스코프 인지)

    /// 블록 하나를 순서대로 걷는다. 지역 선언은 **그 지점부터 블록 끝까지** 이름을 가리고
    /// (WGSL 명세의 point-of-declaration 규칙 — 초기값은 바깥 이름을 본다), 중첩 블록은
    /// 현재 지역 집합의 사본을 물려받는다.
    private static func collect(
        _ statements: [WGSLStatement],
        locals: Set<String>,
        identifiers: inout Set<String>,
        calls: inout Set<String>
    ) {
        var locals = locals
        for statement in statements {
            collect(statement, locals: &locals, identifiers: &identifiers, calls: &calls)
        }
    }

    private static func collect(
        _ statement: WGSLStatement,
        locals: inout Set<String>,
        identifiers: inout Set<String>,
        calls: inout Set<String>
    ) {
        switch statement {
        case .letDeclaration(let name, _, let value), .constDeclaration(let name, _, let value):
            collect(value, locals: locals, identifiers: &identifiers, calls: &calls)
            locals.insert(name)
        case .varDeclaration(let name, _, let value):
            if let value { collect(value, locals: locals, identifiers: &identifiers, calls: &calls) }
            locals.insert(name)
        case .assignment(let target, _, let value):
            collect(target, locals: locals, identifiers: &identifiers, calls: &calls)
            collect(value, locals: locals, identifiers: &identifiers, calls: &calls)
        case .increment(let expression), .decrement(let expression), .expressionStatement(let expression):
            collect(expression, locals: locals, identifiers: &identifiers, calls: &calls)
        case .ifStatement(let condition, let then, let elseBranch):
            collect(condition, locals: locals, identifiers: &identifiers, calls: &calls)
            collect(then, locals: locals, identifiers: &identifiers, calls: &calls)
            switch elseBranch {
            case .block(let statements)?:
                collect(statements, locals: locals, identifiers: &identifiers, calls: &calls)
            case .chained(let statement)?:
                var branchLocals = locals
                collect(statement, locals: &branchLocals, identifiers: &identifiers, calls: &calls)
            case nil:
                break
            }
        case .forStatement(let initializer, let condition, let update, let body):
            // for 헤더의 선언은 조건·증감·본문까지가 스코프다.
            var headerLocals = locals
            if let initializer {
                collect(initializer, locals: &headerLocals, identifiers: &identifiers, calls: &calls)
            }
            if let condition { collect(condition, locals: headerLocals, identifiers: &identifiers, calls: &calls) }
            if let update {
                var updateLocals = headerLocals
                collect(update, locals: &updateLocals, identifiers: &identifiers, calls: &calls)
            }
            collect(body, locals: headerLocals, identifiers: &identifiers, calls: &calls)
        case .whileStatement(let condition, let body):
            collect(condition, locals: locals, identifiers: &identifiers, calls: &calls)
            collect(body, locals: locals, identifiers: &identifiers, calls: &calls)
        case .loopStatement(let body, let continuing):
            // continuing은 본문 선언을 볼 수 있으므로 같은 블록으로 걷는다 (방출도 그렇게 한다).
            collect(body + (continuing ?? []), locals: locals, identifiers: &identifiers, calls: &calls)
        case .switchStatement(let subject, let cases):
            collect(subject, locals: locals, identifiers: &identifiers, calls: &calls)
            for switchCase in cases {
                for selector in switchCase.selectors {
                    collect(selector, locals: locals, identifiers: &identifiers, calls: &calls)
                }
                collect(switchCase.body, locals: locals, identifiers: &identifiers, calls: &calls)
            }
        case .returnStatement(let value):
            if let value { collect(value, locals: locals, identifiers: &identifiers, calls: &calls) }
        case .block(let statements):
            collect(statements, locals: locals, identifiers: &identifiers, calls: &calls)
        case .breakStatement, .continueStatement, .discardStatement:
            break
        }
    }

    private static func collect(
        _ expression: WGSLExpression,
        locals: Set<String>,
        identifiers: inout Set<String>,
        calls: inout Set<String>
    ) {
        switch expression {
        case .identifier(let name):
            if !locals.contains(name) { identifiers.insert(name) }
        case .unary(_, let operand), .paren(let operand), .addressOf(let operand), .dereference(let operand):
            collect(operand, locals: locals, identifiers: &identifiers, calls: &calls)
        case .binary(_, let left, let right), .index(let left, let right):
            collect(left, locals: locals, identifiers: &identifiers, calls: &calls)
            collect(right, locals: locals, identifiers: &identifiers, calls: &calls)
        case .call(let callee, _, let arguments):
            calls.insert(callee)
            arguments.forEach { collect($0, locals: locals, identifiers: &identifiers, calls: &calls) }
        case .member(let base, _):
            collect(base, locals: locals, identifiers: &identifiers, calls: &calls)
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
