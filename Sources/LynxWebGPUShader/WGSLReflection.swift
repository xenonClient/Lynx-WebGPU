import Foundation
import LynxWebGPUCore

/// One binding slot the shader declares.
public struct WGSLResourceInfo: Equatable {
    public let name: String
    public let group: Int
    public let binding: Int
    public let slotKind: WGPUMetalSlotKind
    /// The bind group layout entry a `layout: "auto"` pipeline uses.
    public let bindingLayout: WGPUBindingLayout
}

/// One entry point.
public struct WGSLEntryPointInfo: Equatable {
    public let name: String
    public let stage: WGSLStage
    /// A compute entry point's `@workgroup_size`.
    public let workgroupSize: (x: Int, y: Int, z: Int)?
    /// Names of module-scope variables this entry point actually uses (following the call graph).
    public let usedGlobals: Set<String>

    public static func == (lhs: WGSLEntryPointInfo, rhs: WGSLEntryPointInfo) -> Bool {
        lhs.name == rhs.name && lhs.stage == rhs.stage && lhs.usedGlobals == rhs.usedGlobals
            && lhs.workgroupSize?.x == rhs.workgroupSize?.x
            && lhs.workgroupSize?.y == rhs.workgroupSize?.y
            && lhs.workgroupSize?.z == rhs.workgroupSize?.z
    }
}

/// Pipeline-creation information extracted from a parsed WGSL module.
public struct WGSLShaderReflection {
    public let entryPoints: [WGSLEntryPointInfo]
    public let resources: [WGSLResourceInfo]

    public func entryPoint(named name: String) -> WGSLEntryPointInfo? {
        entryPoints.first { $0.name == name }
    }

    public func resource(named name: String) -> WGSLResourceInfo? {
        resources.first { $0.name == name }
    }

    /// Returns only the resources the given entry points actually use, in group/binding order.
    /// The bind group layout for `layout: "auto"` is derived here.
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

    /// Builds visibility by unioning the stages of the entry points that use the resource.
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

    /// Gathers "globals referenced directly + functions called" per function, then propagates to a fixed point.
    /// If a helper reads a uniform, the entry point has to pass that uniform as an argument (MSL has
    /// no mutable globals), so this set is the input to MSL parameter threading.
    ///
    /// Collection is **scope-aware** — a name shadowed by a parameter or a local declaration
    /// (`var`/`let`/`const`) is not a global use. Machine-generated shaders (Three.js's node system and
    /// the like) routinely reuse one name at module scope and function scope, and ignoring that would
    /// inject globals that are never used as arguments, colliding with the local declarations.
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

    /// Function names **reachable by calls** from the given entry points (including the entry points themselves).
    ///
    /// The emitter emits only this set. Emitting every function in the module would also require the
    /// resources referenced by functions that entry point never calls — and a `layout: "auto"` bind
    /// group contains **only what is used**, so such a function fails looking for a binding that does
    /// not exist. A shader sharing one `common.wgsl` across several entry points (webgpu-samples'
    /// cornell) really did break this way.
    ///
    /// Dropping dead code speeds up the Metal compile by the same amount.
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

    /// Names of functions that must take the buffer size table as an argument (transitively).
    ///
    /// Needed for two reasons:
    /// 1. `arrayLength()` — the element count is read from the table.
    /// 2. **Indexing a runtime-sized array** — the bound is needed to clamp the range (robustness).
    ///
    /// **The emitter and the pipeline must reach the same answer.** If the emitter asks for the table
    /// and the pipeline does not bind it, the shader reads an unbound buffer — hence one calculation, here.
    static func functionsNeedingBufferSizes(in module: WGSLModule) -> Set<String> {
        let runtimeArrays = runtimeArrayGlobalNames(in: module)
        return functionsMatching(in: module) { identifiers, calls in
            calls.contains("arrayLength") || !identifiers.isDisjoint(with: runtimeArrays)
        }
    }

    /// Names of globals holding a runtime-sized array (`array<T>`), directly or as a struct member.
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

    /// Functions that must take the flag suppressing writes after `discard` (transitively).
    ///
    /// MSL's `discard_fragment()` is **not** an immediate return — it only marks "this fragment is
    /// discarded" while the code after it keeps running, so a discarded fragment corrupts storage
    /// buffers and textures. The spec forbids that, so `discard` becomes a flag and writes are masked by it.
    ///
    /// The flag is declared at the entry point and threaded down the call graph, so we need **the
    /// functions that raise discard**, **the functions with writes to mask**, and everything that
    /// calls them. With no `discard` in the module the set is empty — it must cost nothing.
    static func functionsNeedingDiscardFlag(in module: WGSLModule) -> Set<String> {
        let discardsAnywhere = module.functions.contains { bodyDiscards($0.body) }
        guard discardsAnywhere else { return [] }

        let storageGlobals = Set(module.globals.filter { $0.addressSpace == "storage" }.map(\.name))
        let seed = Set(module.functions
            .filter { bodyDiscards($0.body) || bodyWritesResources($0.body, storage: storageGlobals) }
            .map(\.name))
        return closure(seed: seed, in: module)
    }

    /// The seed functions plus everything that calls them (transitively).
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

    /// Whether this body **writes** to a storage buffer, texture or atomic — the things to mask after `discard`.
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

    /// Sub-blocks contained in a statement (for descending through control structures).
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

    /// The root identifier of an assignment target (`out[i].x` → `out`).
    private static func rootIdentifier(of expression: WGSLExpression) -> String? {
        switch expression {
        case .identifier(let name): return name
        case .index(let base, _), .member(let base, _), .dereference(let base), .paren(let base):
            return rootIdentifier(of: base)
        default: return nil
        }
    }

    /// Names of functions that call a particular builtin (transitively).
    ///
    /// Used for builtins that **need an extra argument**, such as `arrayLength` — that argument has to
    /// be taken at the entry point and threaded down the call graph.
    static func functionsCalling(_ builtin: String, in module: WGSLModule) -> Set<String> {
        functionsMatching(in: module) { _, calls in calls.contains(builtin) }
    }

    /// Functions satisfying the condition, plus everything that calls them (transitively).
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

    // MARK: - Identifier collection (scope-aware)

    /// Walks one block in order. A local declaration shadows the name **from that point to the end of
    /// the block** (the WGSL spec's point-of-declaration rule — the initializer still sees the outer
    /// name), and a nested block inherits a copy of the current local set.
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
            // A declaration in a for header scopes over the condition, the increment and the body.
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
            // continuing can see the body's declarations, so we walk it as the same block (emission does too).
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

    // MARK: - Deriving binding layouts

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
