import Foundation
import LynxWebGPUCore

/// Emits Metal Shading Language source from the WGSL syntax tree.
///
/// Two structural transformations are the heart of it:
///
/// 1. **Resource threading** — MSL has no mutable globals. WGSL's module-scope variables (uniform,
///    storage, texture, sampler, workgroup, private) are taken as entry point parameters and **passed
///    down as function arguments** along the call graph.
///    (`WGSLReflectionBuilder.transitiveGlobalUsage` computes which function uses what.)
/// 2. **Entry point wrapping** — the WGSL entry point keeps its signature (`…_inner`) and a separate
///    wrapper carries the stage I/O attributes. This keeps attributes such as `[[attribute]]` and
///    `[[user]]` from polluting the buffer layout in shaders that use one struct for both a uniform
///    buffer and vertex I/O.
struct MSLEmitter {
    private let module: WGSLModule
    private let reflection: WGSLShaderReflection
    private let bindings: WGSLBindingAssignment
    private let uniformStructs: Set<String>
    private let usage: [String: Set<String>]
    private let globalsByName: [String: WGSLGlobalVariable]
    private let structNames: Set<String>
    /// Functions that must take the buffer size table as an argument — those using `arrayLength()` or
    /// **indexing a runtime-sized array** (the latter needs the bound to clamp the range).
    private let needsBufferSizes: Set<String>

    /// Name of the buffer size table argument.
    private static let bufferSizesName = "wgpu_buffer_sizes"

    /// The thread number within the threadgroup — `var<workgroup>` zero-initialization splits the work by it.
    /// If the shader already takes `@builtin(local_invocation_index)`, the declarations merge.
    private static let localInvocationIndexName = "wgpu_bi_local_invocation_index"

    /// Binary operations WGSL defines but C++ leaves **undefined** → a prelude helper.
    ///
    /// - `/` and `%` — division by zero and `INT_MIN / -1`
    /// - `<<` and `>>` — shifts at or beyond the width
    ///
    /// Left alone, any of the three lets the driver do whatever it likes, and optimizers sometimes
    /// treat them as "cannot happen" and delete the surrounding code.
    private static let guardedBinaryOperators = [
        "%": "wgpu_mod", "/": "wgpu_div", "<<": "wgpu_shl", ">>": "wgpu_shr",
    ]

    /// Name → texture type in the current function scope (needed to turn texture builtins into method calls).
    private var textureScope: [String: WGSLTextureType] = [:]
    /// Global names injected as parameters into the current function.
    ///
    /// WGSL allows a local declaration to shadow a global, but once injection turns that global into a
    /// **parameter**, C++ sees a redefinition in the function's top-level block, which is illegal. So
    /// colliding local declarations are renamed.
    private var injectedGlobalNames: Set<String> = []
    /// Active shadowing renames (original name → emitted name). Saved and restored at block boundaries.
    private var localRenames: [String: String] = [:]
    /// What each name declared in the current function (parameters and local declarations) refers to.
    /// Saved and restored together with `localRenames`.
    ///
    /// Two jobs: a local that shadows a global is **not** that global, and a `let` holding `&place`
    /// lets `(*p)[i]` find the bound of a runtime-sized array behind the pointer (`storagePlace(of:)`).
    private var localBindings: [String: LocalBinding] = [:]

    /// A place rooted at a module-scope variable — the variable itself, or one member of it.
    /// The only shapes a runtime-sized array can live in (it may only be the store type or its last member).
    private struct StoragePlace {
        var global: String
        var member: String?
    }

    private enum LocalBinding {
        /// A value, or a pointer to something that is not a module-scope place (a local).
        case value
        /// A `let` pointer known to point at this place.
        case pointer(StoragePlace)
        /// A pointer **parameter** (or a `let` copied from one) — it points at whatever the caller chose,
        /// so no bound can be found for it. Carries the pointee type to recognise a runtime-sized array behind it.
        case parameterPointer(WGSLType)
    }
    /// Name of the synthetic global masking writes after `discard`. Not created when the module has no `discard`.
    ///
    /// It lives in the syntax tree next to the user's globals, so it carries the synthetic sigil — no
    /// WGSL identifier can spell it, so a user global of the same MSL name cannot replace it in
    /// `globalsByName` or the usage sets (`MSLTypeMapping.syntheticNameSigil`).
    private static let discardFlagName = "\(MSLTypeMapping.syntheticNameSigil)discarded"
    /// The flag's MSL name — the same mapping its declaration and threaded arguments go through.
    private static let discardFlagMSLName = MSLTypeMapping.identifier(discardFlagName)
    /// Functions that must take that flag as an argument (empty turns the transformation off entirely).
    private let discardFlagUsers: Set<String>
    /// Names of storage address space globals — used to pick what to mask after `discard`.
    private let storageGlobalNames: Set<String>

    /// Counter keeping infinite-loop guard variable names distinct (nested loops).
    private var loopGuardCounter = 0
    /// While emitting the body of a `loop` that has a `continuing` block and uses `continue`, the
    /// name of the flag a `break` must set (the body is wrapped in a `do { } while (false)`, where
    /// a plain `break` would leave only the wrapper). Nil everywhere else — including inside any
    /// nested loop or switch, whose `break` targets that construct.
    private var breakFlag: String?
    private var output = ""
    private var indentLevel = 0

    init(module: WGSLModule, reflection: WGSLShaderReflection, bindings: WGSLBindingAssignment) {
        var module = module
        var usage = WGSLReflectionBuilder.transitiveGlobalUsage(module)

        // **Inject the flag masking writes after `discard` as a synthetic global.**
        //
        // Rather than writing new code to thread it as an argument, we put it on the resource threading
        // that already exists — a `private` global is declared as a local inside the entry point and
        // passed down the call graph by reference, so all this needs is "one global and the list of
        // functions that use it".
        let discardUsers = WGSLReflectionBuilder.functionsNeedingDiscardFlag(in: module)
        if !discardUsers.isEmpty {
            module.globals.append(WGSLGlobalVariable(
                attributes: [], name: Self.discardFlagName,
                addressSpace: "private", access: nil, type: .scalar("bool"), initializer: nil
            ))
            for name in discardUsers { usage[name, default: []].insert(Self.discardFlagName) }
        }
        self.discardFlagUsers = discardUsers

        self.module = module
        self.reflection = reflection
        self.bindings = bindings
        // A struct that a storage buffer also holds keeps the spec's address-space-independent layout
        // (see `WGSLLayout.storageStructNames`).
        self.uniformStructs = WGSLLayout.uniformStructNames(module)
            .subtracting(WGSLLayout.storageStructNames(module))
        self.usage = usage
        // The parser rejects duplicate module-scope names, but this table must never be what traps on
        // one — shader source comes from the bundle (JS), so a trap here would take the host down.
        self.globalsByName = Dictionary(module.globals.map { ($0.name, $0) }, uniquingKeysWith: { first, _ in first })
        self.structNames = Set(module.structs.map(\.name))
        self.needsBufferSizes = WGSLReflectionBuilder.functionsNeedingBufferSizes(in: module)
        self.storageGlobalNames = Set(module.globals.filter { $0.addressSpace == "storage" }.map(\.name))
    }

    // MARK: - Entry

    mutating func emit(entryPoints requested: [String]) throws -> String {
        output = ""
        line("#include <metal_stdlib>")
        line("#include <simd/simd.h>")
        line("using namespace metal;")
        line("")
        output += MSLPrelude.source + "\n\n"

        // Aliases come before the struct definitions (a struct member may be spelled with an alias),
        // yet an alias may name a struct (`alias B = Buf;`). Forward declarations let `using B = Buf;`
        // name a declared type; the definitions below complete it.
        if !module.aliases.isEmpty {
            for structure in module.structs {
                line("struct \(MSLTypeMapping.identifier(structure.name));")
            }
        }
        for alias in module.aliases {
            line("using \(MSLTypeMapping.identifier(alias.name)) = \(try MSLTypeMapping.type(alias.type, module: module));")
        }
        if !module.aliases.isEmpty { line("") }

        for constant in module.constants {
            try emitModuleConstant(constant)
        }
        if !module.constants.isEmpty { line("") }

        for structure in module.structs {
            try emitStruct(structure)
        }

        // Emit only the functions **reachable by calls** from the requested entry points. Emitting every
        // function in the module would require the binding table to hold resources referenced by
        // functions this entry point never calls — and `layout: "auto"` contains only what is used, so
        // such a function fails looking for a binding that is not there
        // (see `WGSLReflectionBuilder.functionsReachable`).
        let reachable = WGSLReflectionBuilder.functionsReachable(from: requested, in: module)
        for function in module.functions where function.stage == nil && reachable.contains(function.name) {
            try emitFunction(function)
        }

        for name in requested {
            guard let function = module.functionNamed(name), function.stage != nil else {
                throw WGPUError.validation(
                    "the shader has no entry point '\(name)' (available: "
                        + "\(reflection.entryPoints.map(\.name).joined(separator: ", ")))"
                )
            }
            try emitEntryPoint(function)
        }
        return output
    }

    /// Module-scope constants / pipeline constants.
    ///
    /// MSL does not allow `constant auto x = …`. Without a type annotation we expand to a macro —
    /// a WGSL module constant is a compile-time value, so the meaning is the same.
    private mutating func emitModuleConstant(_ constant: WGSLModuleConstant) throws {
        guard let value = constant.value else {
            throw WGPUError.validation(
                "override '\(constant.name)' has no value — it must be supplied at pipeline creation "
                    + "as `constants: { \(constant.name): … }`"
            )
        }
        if let type = constant.type ?? inferredType(of: value) {
            line("constant \(try MSLTypeMapping.type(type, module: module)) \(MSLTypeMapping.identifier(constant.name)) = \(try expression(value));")
        } else {
            line("#define \(MSLTypeMapping.identifier(constant.name)) (\(try expression(value)))")
        }
    }

    /// Whether every argument is a **suffix-free integer constant expression** (WGSL's AbstractInt).
    ///
    /// The case where the arguments give no basis for a component type, as in `vec2(4, 1)` or
    /// `vec4(1741651 * 1009, …)`. A suffix such as `4u`, or a mixed-in identifier, fixes the type, so
    /// those are left to template deduction.
    private static func isAbstractIntegerArguments(_ arguments: [WGSLExpression]) -> Bool {
        guard !arguments.isEmpty else { return false }
        return arguments.allSatisfy(isAbstractInteger)
    }

    private static func isAbstractInteger(_ expression: WGSLExpression) -> Bool {
        switch expression {
        case .intLiteral(let text):
            return !text.hasSuffix("u")
        case .paren(let inner):
            return isAbstractInteger(inner)
        case .unary(let op, let inner):
            return (op == "-" || op == "+") && isAbstractInteger(inner)
        case .binary(let op, let left, let right):
            return ["+", "-", "*", "/", "%"].contains(op)
                && isAbstractInteger(left) && isAbstractInteger(right)
        default:
            return false
        }
    }

    /// `arrayLength(&buffer)` / `arrayLength(&buffer.member)` → a buffer size table lookup.
    /// A `let` pointer to either works too (`let p = &buffer.member; arrayLength(p)`).
    ///
    /// A Metal shader cannot know a buffer's size. The runtime plugs a size table into the reserved
    /// index, and here we compute (bytes in the table − the array's start offset) ÷ element size.
    private func arrayLengthExpression(_ arguments: [WGSLExpression]) throws -> String {
        let argument = arguments.first ?? .identifier("")
        guard let place = pointerTarget(of: argument) ?? storagePlace(of: argument) else {
            throw WGPUError.unsupported(
                "WGSL: arrayLength() can only be used on a storage buffer variable (or a member of one)"
            )
        }
        guard let count = try runtimeArrayElementCount(at: place) else {
            throw WGPUError.unsupported("WGSL: the target of arrayLength() is not a runtime-sized array")
        }
        return count
    }

    /// The element count of the runtime-sized array at `place`, or nil when that place is not one.
    ///
    /// The bound size can be **smaller than the array's start offset** (nothing validates a binding
    /// against the struct's fixed part). Subtracting in uint32 would then wrap round to ~4e9 and turn
    /// every clamp built on this count into a no-op, so the subtraction saturates at 0.
    private func runtimeArrayElementCount(at place: StoragePlace) throws -> String? {
        guard let global = globalsByName[place.global] else { return nil }

        let element: WGSLType
        var offset = 0
        switch (global.type, place.member) {
        case (.array(let inner, nil), nil):
            element = inner
        case (.named(let typeName), let memberName?):
            // Through `resolveStruct`, so `alias B = Buf; var<storage> b: B;` finds `Buf`'s members.
            guard let structure = WGSLLayout.resolveStruct(typeName, module: module) else { return nil }
            let placement = WGSLLayout.layout(
                of: structure, module: module, uniform: uniformStructs.contains(structure.name)
            )
            guard let member = placement.members.first(where: { $0.name == memberName }),
                  case .array(let inner, nil) = member.type else { return nil }
            element = inner
            offset = member.offset
        default:
            return nil
        }

        guard let group = global.group, let binding = global.binding,
              let index = bindings.index(group: group, binding: binding) else {
            throw WGPUError.validation("runtime-sized array '\(place.global)' is not a bound storage buffer")
        }
        let elementType = try MSLTypeMapping.type(element, module: module)
        let size = "\(Self.bufferSizesName)[\(index)]"
        let total = offset == 0 ? size : "(\(size) > \(offset)u ? \(size) - \(offset)u : 0u)"
        return "(\(total) / uint(sizeof(\(elementType))))"
    }

    /// The module-scope place an expression designates — `g`, `g.member`, and the same reached
    /// **through a `let` pointer** (`*p`, `(*p).member`). nil when it is rooted in a local or is any
    /// other shape; such a target is never given a bound, and the prelude refuses to index a runtime-sized
    /// array without one (a compile error rather than an unchecked access).
    private func storagePlace(of expression: WGSLExpression) -> StoragePlace? {
        switch expression {
        case .paren(let inner):
            return storagePlace(of: inner)
        case .identifier(let name):
            guard localBindings[name] == nil, globalsByName[name] != nil else { return nil }
            return StoragePlace(global: name, member: nil)
        case .member(let base, let field):
            guard let place = storagePlace(of: base), place.member == nil else { return nil }
            return StoragePlace(global: place.global, member: field)
        case .dereference(let pointer):
            return pointerTarget(of: pointer)
        default:
            return nil
        }
    }

    /// The place a pointer-valued expression points at — `&place`, or a `let` pointer holding one.
    private func pointerTarget(of expression: WGSLExpression) -> StoragePlace? {
        switch expression {
        case .paren(let inner):
            return pointerTarget(of: inner)
        case .addressOf(let place):
            return storagePlace(of: place)
        case .identifier(let name):
            guard case .pointer(let place)? = localBindings[name] else { return nil }
            return place
        default:
            return nil
        }
    }

    /// Function parameters are locals — a parameter shadowing a global is not that global. A pointer
    /// parameter points at the caller's choice of place, so it is never resolved to one.
    private static func parameterBindings(of function: WGSLFunction) -> [String: LocalBinding] {
        Dictionary(function.parameters.map { parameter in
            if case .pointer(_, let pointee, _) = parameter.type {
                return (parameter.name, LocalBinding.parameterPointer(pointee))
            }
            return (parameter.name, LocalBinding.value)
        }, uniquingKeysWith: { first, _ in first })
    }

    /// The pointee type of a pointer that leads back to a pointer **parameter** (`q`, `&(*q)`,
    /// `&(*q).member`, or a `let` holding one of those). nil for anything else.
    private func parameterPointee(of expression: WGSLExpression) -> WGSLType? {
        switch expression {
        case .paren(let inner):
            return parameterPointee(of: inner)
        case .identifier(let name):
            guard case .parameterPointer(let pointee)? = localBindings[name] else { return nil }
            return pointee
        case .addressOf(let place):
            return parameterPlaceType(of: place)
        default:
            return nil
        }
    }

    /// The type of a place reached through a pointer parameter (`*q`, `(*q).member`). nil for anything else.
    private func parameterPlaceType(of expression: WGSLExpression) -> WGSLType? {
        switch expression {
        case .paren(let inner):
            return parameterPlaceType(of: inner)
        case .dereference(let pointer):
            return parameterPointee(of: pointer)
        case .member(let base, let field):
            guard case .named(let typeName)? = parameterPlaceType(of: base),
                  let structure = WGSLLayout.resolveStruct(typeName, module: module) else { return nil }
            return structure.members.first(where: { $0.name == field })?.type
        default:
            return nil
        }
    }

    /// Builtins whose result is a scalar of the argument's (float) component type.
    private static let scalarResultBuiltins: Set<String> = ["length", "dot", "distance", "determinant"]
    /// Builtins that keep the first argument's shape and make it float (`sqrt(4)` is f32).
    private static let floatShapedBuiltins: Set<String> = [
        "sqrt", "inverseSqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh",
        "exp", "exp2", "log", "log2", "pow", "floor", "ceil", "round", "trunc", "fract", "radians", "degrees",
        "mix", "smoothstep", "step", "normalize", "fma", "saturate",
    ]
    /// Builtins that keep the first argument's type exactly.
    private static let shapePreservingBuiltins: Set<String> = ["abs", "min", "max", "clamp", "sign"]

    private static func componentScalar(of type: WGSLType) -> WGSLType {
        if case .vector(_, let element) = type { return element }
        return type
    }

    /// An integer scalar or vector read as its float counterpart (an AbstractInt argument to a float
    /// builtin converts to AbstractFloat).
    private static func floatened(_ type: WGSLType) -> WGSLType {
        switch type {
        case .scalar("i32"), .scalar("u32"): return .scalar("f32")
        case .vector(let size, .scalar("i32")), .vector(let size, .scalar("u32")):
            return .vector(size: size, element: .scalar("f32"))
        default: return type
        }
    }

    /// Guesses the type from the initializer's **syntax** alone (reading constructor names, not inferring types).
    /// Used for module constants and for `array(…)` constructors with the element type omitted.
    private func inferredType(of expression: WGSLExpression) -> WGSLType? {
        switch expression {
        case .floatLiteral:
            return .scalar("f32")
        case .intLiteral(let text):
            return .scalar(text.hasSuffix("u") ? "u32" : "i32")
        case .boolLiteral:
            return .scalar("bool")
        case .paren(let inner), .unary(_, let inner):
            return inferredType(of: inner)
        case .binary(let op, let left, let right):
            // Enough shape inference to give an arithmetic constant a type — a typeless module
            // constant becomes a `#define`, which is textually unhygienic: `const scale = 1.0 / 255.0;`
            // then rewrote a struct member named `scale` into the expression.
            let leftType = inferredType(of: left)
            let rightType = inferredType(of: right)
            if ["==", "!=", "<", "<=", ">", ">=", "&&", "||"].contains(op) {
                if case .vector(let size, _)? = leftType ?? rightType {
                    return .vector(size: size, element: .scalar("bool"))
                }
                return .scalar("bool")
            }
            if op == "<<" || op == ">>" { return leftType }
            switch (leftType, rightType) {
            case (.matrix?, _), (_, .matrix?):
                return nil
            case (.vector?, _):
                return leftType
            case (_, .vector?):
                return rightType
            case (.scalar(let leftScalar)?, .scalar(let rightScalar)?):
                if leftScalar == rightScalar { return leftType }
                // AbstractInt meets AbstractFloat and converts (`1.0 * 2`).
                if leftScalar == "f32" || rightScalar == "f32" { return .scalar("f32") }
                return nil
            default:
                return nil
            }
        case .call(let callee, let typeArguments, let arguments):
            if structNames.contains(callee) { return .named(callee) }
            if Self.scalarResultBuiltins.contains(callee), let first = arguments.first.flatMap(inferredType(of:)) {
                return Self.floatened(Self.componentScalar(of: first))
            }
            if Self.floatShapedBuiltins.contains(callee), let first = arguments.first.flatMap(inferredType(of:)) {
                return Self.floatened(first)
            }
            if Self.shapePreservingBuiltins.contains(callee), let first = arguments.first.flatMap(inferredType(of:)) {
                return first
            }
            if MSLTypeMapping.scalarConstructors.contains(callee) { return .scalar(callee) }
            if let shorthand = WGSLParser.shorthandType(callee) { return shorthand }
            if callee.hasPrefix("vec"), let size = Int(callee.dropFirst(3)) {
                if case .scalar? = typeArguments.first {
                    return .vector(size: size, element: typeArguments[0])
                }
                // With the component type omitted, guess from the arguments (`vec2(-1.0, -1.0)` → vec2<f32>).
                // If every argument is an AbstractInt constant expression the emitter emits a proxy, so no
                // type can be written (it must flow through the macro/auto path to freeze from context).
                if Self.isAbstractIntegerArguments(arguments) { return nil }
                guard let element = arguments.first.flatMap(inferredType(of:)),
                      case .scalar = element else { return nil }
                return .vector(size: size, element: element)
            }
            if callee == "array" {
                return arrayConstructorType(typeArguments: typeArguments, argumentCount: arguments.count)
                    ?? arguments.first.flatMap(inferredType(of:)).map {
                        WGSLType.array(element: $0, count: .intLiteral(String(arguments.count)))
                    }
            }
            return nil
        default:
            return nil
        }
    }

    // MARK: - Output helpers

    private mutating func line(_ text: String = "") {
        if text.isEmpty {
            output += "\n"
        } else {
            output += String(repeating: "    ", count: indentLevel) + text + "\n"
        }
    }

    private mutating func indented(_ body: (inout MSLEmitter) throws -> Void) rethrows {
        indentLevel += 1
        try body(&self)
        indentLevel -= 1
    }

    /// Block scope boundary — keeps shadowing renames registered inside from leaking out of the block.
    private mutating func scoped(_ body: (inout MSLEmitter) throws -> Void) rethrows {
        let saved = localRenames
        let savedBindings = localBindings
        try indented(body)
        localRenames = saved
        localBindings = savedBindings
    }

    // MARK: - Structs (padded to the WGSL layout)

    private mutating func emitStruct(_ structure: WGSLStruct) throws {
        let isUniform = uniformStructs.contains(structure.name)
        let placement = WGSLLayout.layout(of: structure, module: module, uniform: isUniform)

        line("struct alignas(\(placement.align)) \(MSLTypeMapping.identifier(structure.name)) {")
        var cursor = 0
        var padIndex = 0
        try indented { emitter in
            for member in placement.members {
                if cursor < member.offset {
                    emitter.line("char wgpu_pad\(padIndex)[\(member.offset - cursor)];")
                    padIndex += 1
                    cursor = member.offset
                }
                let declaration = try emitter.memberDeclaration(member, in: structure)
                emitter.line(declaration.text)
                cursor = member.offset + declaration.byteSize
            }
            if cursor < placement.size {
                emitter.line("char wgpu_pad\(padIndex)[\(placement.size - cursor)];")
            }
        }
        line("};")
        line("")
    }

    /// One member line and the bytes that member actually occupies in MSL.
    private func memberDeclaration(
        _ member: WGSLLayout.MemberPlacement,
        in structure: WGSLStruct
    ) throws -> (text: String, byteSize: Int) {
        // A runtime-sized array has no MSL counterpart. Only as the last member do we emit a length-1
        // array and let the caller guarantee the real length (docs/WGSL.md §4).
        if case .array(let element, nil) = member.type {
            let elementType = try MSLTypeMapping.type(element, module: module)
            return ("\(elementType) \(MSLTypeMapping.identifier(member.name))[1];", member.size)
        }
        if member.needsPackedVector, case .vector(3, .scalar(let scalar)) = member.type {
            // A WGSL vec3 has size 12 — MSL float3 (16 bytes) cannot line up the members after it.
            return ("packed_\(MSLTypeMapping.scalar(scalar))3 \(MSLTypeMapping.identifier(member.name));", 12)
        }
        let type = try MSLTypeMapping.type(member.type, module: module)
        if case .vector(3, _) = member.type {
            return ("\(type) \(MSLTypeMapping.identifier(member.name));", 16)
        }
        return ("\(type) \(MSLTypeMapping.identifier(member.name));", member.size)
    }

    /// Aggregate initialization matching the padded declaration `emitStruct` produced: the same
    /// cursor walk decides where a `char wgpu_padN[…]` member sits, and each such slot gets a
    /// zero-filling `{}` so the positional arguments land on the real members.
    private func structConstructor(_ name: String, arguments: [String]) throws -> String {
        let mslName = MSLTypeMapping.identifier(name)
        guard !arguments.isEmpty, let structure = module.structs.first(where: { $0.name == name }) else {
            return "\(mslName){\(arguments.joined(separator: ", "))}"   // zero-value construction zero-fills everything
        }
        let placement = WGSLLayout.layout(of: structure, module: module, uniform: uniformStructs.contains(name))
        var parts: [String] = []
        var cursor = 0
        for (index, member) in placement.members.enumerated() {
            if cursor < member.offset { parts.append("{}") }
            parts.append(index < arguments.count ? arguments[index] : "{}")
            cursor = member.offset + (try memberDeclaration(member, in: structure)).byteSize
        }
        return "\(mslName){\(parts.joined(separator: ", "))}"
    }

    // MARK: - Resource threading

    /// Returns the module-scope variables a function uses (transitively) in deterministic order.
    private func threadedGlobals(for functionName: String) -> [WGSLGlobalVariable] {
        let used = usage[functionName] ?? []
        return module.globals
            .filter { used.contains($0.name) }
            .sorted {
                ($0.group ?? Int.max, $0.binding ?? Int.max, $0.name)
                    < ($1.group ?? Int.max, $1.binding ?? Int.max, $1.name)
            }
    }

    /// The declaration used when passing a global as a function argument (`constant Uniforms& u`).
    private func parameterDeclaration(for global: WGSLGlobalVariable) throws -> String {
        switch global.type {
        case .texture, .sampler:
            return "\(try MSLTypeMapping.type(global.type, module: module)) \(MSLTypeMapping.identifier(global.name))"
        default:
            break
        }
        let space = MSLTypeMapping.addressSpace(global.addressSpace ?? "private")
        let isReadOnly = global.addressSpace == "storage" && (global.access == nil || global.access == "read")
        let qualifier = isReadOnly ? "const \(space)" : space

        // When a storage buffer's stored type is a runtime-sized array, take it as a pointer.
        if case .array(let element, nil) = global.type {
            return "\(qualifier) \(try MSLTypeMapping.type(element, module: module))* \(MSLTypeMapping.identifier(global.name))"
        }
        return "\(qualifier) \(try MSLTypeMapping.type(global.type, module: module))& \(MSLTypeMapping.identifier(global.name))"
    }

    /// The Metal index attribute attached when an entry point receives a resource.
    private func bindingAttribute(for global: WGSLGlobalVariable) throws -> String {
        guard let group = global.group, let binding = global.binding else { return "" }
        guard let index = bindings.index(group: group, binding: binding) else {
            throw WGPUError.validation(
                "@group(\(group)) @binding(\(binding)) (\(global.name)), used by the shader, "
                    + "is not in the pipeline layout"
            )
        }
        guard let layout = WGSLReflectionBuilder.bindingLayout(for: global) else { return "" }
        if let expected = bindings.kind(group: group, binding: binding), expected != layout.metalSlotKind {
            throw WGPUError.validation(
                "@group(\(group)) @binding(\(binding)) (\(global.name)) is a \(layout.metalSlotKind) in the "
                    + "shader but a \(expected) in the pipeline layout"
            )
        }
        switch layout.metalSlotKind {
        case .buffer: return " [[buffer(\(index))]]"
        case .texture: return " [[texture(\(index))]]"
        case .sampler: return " [[sampler(\(index))]]"
        }
    }

    // MARK: - Ordinary functions

    private mutating func emitFunction(_ function: WGSLFunction) throws {
        let previousScope = textureScope
        defer {
            textureScope = previousScope
            injectedGlobalNames = []
            localRenames = [:]
            localBindings = [:]
        }
        registerTextures(of: function)
        injectedGlobalNames = Set(threadedGlobals(for: function.name).map(\.name))
        localRenames = [:]
        localBindings = Self.parameterBindings(of: function)

        var parameters = try function.parameters.map { parameter in
            "\(try MSLTypeMapping.type(parameter.type, module: module)) \(MSLTypeMapping.identifier(parameter.name))"
        }
        parameters += try threadedGlobals(for: function.name).map(parameterDeclaration(for:))
        if needsBufferSizes.contains(function.name) {
            parameters.append("constant uint* \(Self.bufferSizesName)")
        }

        let returnType = try function.returnType.map { try MSLTypeMapping.type($0, module: module) } ?? "void"
        line("\(returnType) \(MSLTypeMapping.functionName(function.name))(\(parameters.joined(separator: ", ")))")
        line("{")
        try indented { emitter in
            try emitter.statements(function.body)
        }
        line("}")
        line("")
    }

    private mutating func registerTextures(of function: WGSLFunction) {
        for global in module.globals {
            if case .texture(let texture) = global.type { textureScope[global.name] = texture }
        }
        for parameter in function.parameters {
            if case .texture(let texture) = parameter.type { textureScope[parameter.name] = texture }
        }
    }

    // MARK: - Entry points

    private struct EntryInterface {
        var stageInFields: [String] = []
        var builtinParameters: [String] = []
        var prelude: [String] = []
        var innerArguments: [String] = []
        var outFields: [String] = []
        var packStatements: [String] = []
    }

    private mutating func emitEntryPoint(_ function: WGSLFunction) throws {
        guard let stage = function.stage else { return }
        let previousScope = textureScope
        defer { textureScope = previousScope }
        registerTextures(of: function)

        // Names derived from the entry point carry their own `wgpu_ep_` tag: a bare `wgpu_<entry>_…`
        // could spell a renamed user identifier (`wgpu_id_…`) for an entry point named `id_…`.
        let innerName = "wgpu_ep_\(function.name)_inner"
        let resources = threadedGlobals(for: function.name)

        // 1) The inner function keeping the original signature.
        var innerParameters = try function.parameters.map { parameter in
            "\(try MSLTypeMapping.type(parameter.type, module: module)) \(MSLTypeMapping.identifier(parameter.name))"
        }
        innerParameters += try resources.map(parameterDeclaration(for:))
        if needsBufferSizes.contains(function.name) {
            innerParameters.append("constant uint* \(Self.bufferSizesName)")
        }
        let returnType = try function.returnType.map { try MSLTypeMapping.type($0, module: module) } ?? "void"
        line("\(returnType) \(innerName)(\(innerParameters.joined(separator: ", ")))")
        line("{")
        injectedGlobalNames = Set(resources.map(\.name))
        localRenames = [:]
        localBindings = Self.parameterBindings(of: function)
        try indented { emitter in
            try emitter.statements(function.body)
        }
        // Wrapper emission is outside the injection scope — a leaked rename would make wgpu_out packing use the wrong name.
        injectedGlobalNames = []
        localRenames = [:]
        localBindings = [:]
        line("}")
        line("")

        // 2) The stage I/O interface.
        var interface = EntryInterface()
        try buildInputs(of: function, stage: stage, into: &interface)
        try buildOutputs(of: function, stage: stage, into: &interface)

        // WGSL **guarantees** zero-initialization of `var<workgroup>`. MSL's threadgroup storage does
        // not, so the leftovers of a previous dispatch show through — a bug that reproduces irregularly
        // and is very hard to track down. Splitting the zeroing across threads needs each thread's
        // number within the threadgroup.
        let workgroupGlobals = resources.filter { !$0.isResource && $0.addressSpace == "workgroup" }
        if stage == .compute, !workgroupGlobals.isEmpty {
            let declaration = "uint \(Self.localInvocationIndexName) [[thread_index_in_threadgroup]]"
            if !interface.builtinParameters.contains(declaration) {
                interface.builtinParameters.append(declaration)
            }
        }

        let inputStructName = "wgpu_ep_\(function.name)_in"
        if !interface.stageInFields.isEmpty {
            line("struct \(inputStructName) {")
            indented { emitter in
                for field in interface.stageInFields { emitter.line(field) }
            }
            line("};")
            line("")
        }

        let outputStructName = "wgpu_ep_\(function.name)_out"
        if !interface.outFields.isEmpty {
            line("struct \(outputStructName) {")
            indented { emitter in
                for field in interface.outFields { emitter.line(field) }
            }
            line("};")
            line("")
        }

        // 3) The wrapper.
        var wrapperParameters: [String] = []
        if !interface.stageInFields.isEmpty {
            wrapperParameters.append("\(inputStructName) wgpu_in [[stage_in]]")
        }
        wrapperParameters += interface.builtinParameters
        for global in resources where global.isResource {
            wrapperParameters.append("\(try parameterDeclaration(for: global))\(try bindingAttribute(for: global))")
        }
        if needsBufferSizes.contains(function.name) {
            // The runtime plugs the byte sizes of the bound buffers in at this index.
            wrapperParameters.append(
                "constant uint* \(Self.bufferSizesName) "
                    + "[[buffer(\(WGSLMetalLimits.bufferSizesIndex))]]"
            )
        }

        let qualifier: String
        switch stage {
        case .vertex: qualifier = "vertex"
        case .fragment: qualifier = "fragment"
        case .compute: qualifier = "kernel"
        }
        let wrapperReturn = interface.outFields.isEmpty ? "void" : outputStructName
        let wrapperName = MSLTypeMapping.functionName(function.name)
        line("\(qualifier) \(wrapperReturn) \(wrapperName)(\(wrapperParameters.joined(separator: ", ")))")
        line("{")
        try indented { emitter in
            // workgroup and private globals are local storage inside the entry point, not bindings.
            for global in resources where !global.isResource {
                try emitter.emitLocalGlobal(global)
            }
            if stage == .compute, !workgroupGlobals.isEmpty {
                // The module here has the pipeline constants applied — the same size the dispatch uses.
                let threadsPerGroup = try emitter.module.workgroupSize(of: function) ?? (1, 1, 1)
                try emitter.emitWorkgroupZeroInit(workgroupGlobals, threadsPerGroup: threadsPerGroup)
            }
            for statement in interface.prelude { emitter.line(statement) }

            var arguments = interface.innerArguments + resources.map { MSLTypeMapping.identifier($0.name) }
            if emitter.needsBufferSizes.contains(function.name) { arguments.append(Self.bufferSizesName) }
            let call = "\(innerName)(\(arguments.joined(separator: ", ")))"
            // The actual discard happens **here** — the body only raised the flag and masked writes.
            // Calling it last is the point: derivatives that neighbours in the same quad use must stay alive.
            let discardTail = !emitter.discardFlagUsers.isEmpty && resources.contains {
                $0.name == Self.discardFlagName
            }
            if interface.outFields.isEmpty {
                emitter.line("\(call);")
                if discardTail { emitter.line("if (\(Self.discardFlagMSLName)) { discard_fragment(); }") }
            } else {
                emitter.line("\(returnType) wgpu_result = \(call);")
                if discardTail { emitter.line("if (\(Self.discardFlagMSLName)) { discard_fragment(); }") }
                emitter.line("\(outputStructName) wgpu_out{};")
                for statement in interface.packStatements { emitter.line(statement) }
                emitter.line("return wgpu_out;")
            }
        }
        line("}")
        line("")
    }

    /// Zeroes `var<workgroup>` globals — establishing the initial state the spec guarantees.
    ///
    /// Threadgroup storage cannot be brace-initialized, so **the threads share the work.** Striding by
    /// the group's thread count keeps any one thread from doing it all, however large the array.
    /// Without the final barrier another thread **reads a slot not yet zeroed** — a common trap where
    /// the symptom survives even after adding the initialization.
    private mutating func emitWorkgroupZeroInit(
        _ globals: [WGSLGlobalVariable],
        threadsPerGroup: (Int, Int, Int)
    ) throws {
        // `translateToMSL` does not apply the device limits, so the product is checked here: it must
        // not overflow, and it is emitted as a `uint` literal.
        let (partial, overflowXY) = threadsPerGroup.0.multipliedReportingOverflow(by: threadsPerGroup.1)
        let (product, overflowXYZ) = partial.multipliedReportingOverflow(by: threadsPerGroup.2)
        guard !overflowXY, !overflowXYZ, product <= Int(UInt32.max) else {
            throw WGPUError.validation(
                "@workgroup_size (\(threadsPerGroup.0), \(threadsPerGroup.1), \(threadsPerGroup.2)) — "
                    + "the invocation count does not fit in u32"
            )
        }
        let total = max(product, 1)
        let index = Self.localInvocationIndexName
        line("// WGSL guarantees zero-initialization of var<workgroup> (MSL threadgroup storage does not).")
        for global in globals {
            let name = MSLTypeMapping.identifier(global.name)
            if case .array(let element, let count) = global.type, let count,
               let length = WGSLLayout.constantValue(count, module: module), length > 0 {
                line("for (uint wgpu_zi = \(index); wgpu_zi < \(length)u; wgpu_zi += \(total)u)")
                line("{")
                try indented { emitter in
                    emitter.line(try Self.zeroStatement(for: element, target: "\(name)[wgpu_zi]", module: emitter.module))
                }
                line("}")
            } else {
                // For a non-array one thread suffices — the rest meet at the barrier.
                let statement = try Self.zeroStatement(for: global.type, target: name, module: module)
                line("if (\(index) == 0u) { \(statement) }")
            }
        }
        line("threadgroup_barrier(mem_flags::mem_threadgroup);")
    }

    /// The statement zeroing one slot. An atomic can only be written with store, not assignment.
    private static func zeroStatement(for type: WGSLType, target: String, module: WGSLModule) throws -> String {
        if case .atomic = type {
            return "atomic_store_explicit(&\(target), 0, memory_order_relaxed);"
        }
        return "\(target) = \(try MSLTypeMapping.type(type, module: module)){};"
    }

    private mutating func emitLocalGlobal(_ global: WGSLGlobalVariable) throws {
        let type = try MSLTypeMapping.type(global.type, module: module)
        if global.addressSpace == "workgroup" {
            line("threadgroup \(type) \(MSLTypeMapping.identifier(global.name));")
        } else if let initializer = global.initializer {
            line("\(type) \(MSLTypeMapping.identifier(global.name)) = \(try expression(initializer));")
        } else {
            line("\(type) \(MSLTypeMapping.identifier(global.name)){};")
        }
    }

    private mutating func buildInputs(
        of function: WGSLFunction,
        stage: WGSLStage,
        into interface: inout EntryInterface
    ) throws {
        for parameter in function.parameters {
            if case .named(let typeName) = parameter.type,
               let structure = WGSLLayout.resolveStruct(typeName, module: module) {
                let local = "wgpu_arg_\(MSLTypeMapping.identifier(parameter.name))"
                interface.prelude.append("\(MSLTypeMapping.identifier(typeName)) \(local){};")
                for member in structure.members {
                    let source = try inputSource(
                        attributes: member.attributes,
                        type: member.type,
                        stage: stage,
                        into: &interface
                    )
                    interface.prelude.append("\(local).\(MSLTypeMapping.identifier(member.name)) = \(source);")
                }
                interface.innerArguments.append(local)
            } else {
                if case .named(let typeName) = parameter.type {
                    throw WGPUError.validation(
                        "could not find type '\(typeName)' of entry point parameter '\(parameter.name)' "
                            + "— the struct declaration must be in the same shader module"
                    )
                }
                let source = try inputSource(
                    attributes: parameter.attributes,
                    type: parameter.type,
                    stage: stage,
                    into: &interface
                )
                interface.innerArguments.append(source)
            }
        }
    }

    /// Registers one input as a stage_in field or a builtin parameter, and returns the expression to pass to the inner function.
    private mutating func inputSource(
        attributes: [WGSLAttribute],
        type: WGSLType,
        stage: WGSLStage,
        into interface: inout EntryInterface
    ) throws -> String {
        if let builtinName = attributes.builtin {
            let builtin = try MSLTypeMapping.builtin(builtinName, stage: stage, isInput: true)
            let name = "wgpu_bi_\(builtinName)"
            let declaration = "\(builtin.type) \(name) \(builtin.attribute)"
            if !interface.builtinParameters.contains(declaration) {
                interface.builtinParameters.append(declaration)
            }
            let target = try MSLTypeMapping.type(type, module: module)
            return target == builtin.type ? name : "\(target)(\(name))"
        }
        guard let location = attributes.location else {
            throw WGPUError.validation("an entry point input needs @location or @builtin")
        }
        let fieldName = "f\(location)"
        let attribute = stage == .vertex ? "[[attribute(\(location))]]" : "[[user(locn\(location))]]"
        let interpolation = stage == .vertex ? "" : MSLTypeMapping.interpolation(attributes.first(named: "interpolate"))
        let mslType = try MSLTypeMapping.type(type, module: module)
        let field = "\(mslType) \(fieldName) \(attribute)\(interpolation);"
        if !interface.stageInFields.contains(field) { interface.stageInFields.append(field) }
        return "wgpu_in.\(fieldName)"
    }

    private mutating func buildOutputs(
        of function: WGSLFunction,
        stage: WGSLStage,
        into interface: inout EntryInterface
    ) throws {
        guard let returnType = function.returnType else { return }

        if case .named(let typeName) = returnType,
           let structure = WGSLLayout.resolveStruct(typeName, module: module) {
            for member in structure.members {
                let field = try outputField(attributes: member.attributes, type: member.type, stage: stage)
                interface.outFields.append(field.declaration)
                interface.packStatements.append("wgpu_out.\(field.name) = wgpu_result.\(MSLTypeMapping.identifier(member.name));")
            }
            return
        }
        let field = try outputField(attributes: function.returnAttributes, type: returnType, stage: stage)
        interface.outFields.append(field.declaration)
        interface.packStatements.append("wgpu_out.\(field.name) = wgpu_result;")
    }

    private func outputField(
        attributes: [WGSLAttribute],
        type: WGSLType,
        stage: WGSLStage
    ) throws -> (name: String, declaration: String) {
        let mslType = try MSLTypeMapping.type(type, module: module)
        if let builtinName = attributes.builtin {
            let builtin = try MSLTypeMapping.builtin(builtinName, stage: stage, isInput: false)
            let name = "wgpu_b_\(builtinName)"
            return (name, "\(builtin.type) \(name) \(builtin.attribute);")
        }
        guard let location = attributes.location else {
            throw WGPUError.validation("an entry point output needs @location or @builtin")
        }
        let name = "f\(location)"
        if stage == .fragment {
            return (name, "\(mslType) \(name) [[color(\(location))]];")
        }
        let interpolation = MSLTypeMapping.interpolation(attributes.first(named: "interpolate"))
        return (name, "\(mslType) \(name) [[user(locn\(location))]]\(interpolation);")
    }

    // MARK: - Statements

    private mutating func statements(_ list: [WGSLStatement]) throws {
        for statement in list { try emitStatement(statement) }
    }

    private mutating func emitStatement(_ statement: WGSLStatement) throws {
        switch statement {
        case .block(let inner):
            guard !inner.isEmpty else { return }
            line("{")
            try scoped { emitter in try emitter.statements(inner) }
            line("}")

        case .letDeclaration, .constDeclaration, .varDeclaration, .assignment, .increment, .decrement:
            let text = try simpleStatement(statement)
            if masksAfterDiscard(statement) {
                line("if (!\(Self.discardFlagMSLName)) { \(text); }")
            } else {
                line("\(text);")
            }

        case .expressionStatement(let expression):
            let text = try self.expression(expression)
            if masksAfterDiscard(statement) {
                line("if (!\(Self.discardFlagMSLName)) { \(text); }")
            } else {
                line("\(text);")
            }

        case .ifStatement(let clauses, let elseBody):
            // The chain stays flat in MSL too: `if … else if … else …`, one loop iteration per arm.
            for (index, clause) in clauses.enumerated() {
                line("\(index == 0 ? "if" : "else if") (\(try expression(clause.condition)))")
                line("{")
                try scoped { emitter in try emitter.statements(clause.body) }
                line("}")
            }
            if let elseBody {
                line("else")
                line("{")
                try scoped { emitter in try emitter.statements(elseBody) }
                line("}")
            }

        case .forStatement(let initializer, let condition, let update, let body):
            // A header declaration's rename scopes over the condition, increment and body — undone when the for ends.
            let savedRenames = localRenames
            let savedBindings = localBindings
            let savedBreakFlag = breakFlag
            breakFlag = nil   // a `break` inside targets this loop
            let initializerText = try initializer.map { try simpleStatement($0) } ?? ""
            let conditionText = try condition.map { try expression($0) } ?? ""
            let updateText = try update.map { try simpleStatement($0) } ?? ""
            line("for (\(initializerText); \(conditionText); \(updateText))")
            line("{")
            try scoped { emitter in try emitter.statements(body) }
            line("}")
            localRenames = savedRenames
            localBindings = savedBindings
            breakFlag = savedBreakFlag

        case .whileStatement(let condition, let body):
            let conditionText = try expression(condition)
            // A literal `true` condition looks like a non-terminating loop to the compiler — add a guard.
            let guardName = conditionText == "true" ? declareLoopGuard() : nil
            let savedBreakFlag = breakFlag
            breakFlag = nil
            line("while (\(conditionText))")
            line("{")
            try scoped { emitter in
                if let guardName { emitter.emitLoopGuardCheck(guardName) }
                try emitter.statements(body)
            }
            line("}")
            breakFlag = savedBreakFlag

        case .loopStatement(let body, let continuing):
            let guardName = declareLoopGuard()
            let savedBreakFlag = breakFlag
            breakFlag = nil
            line("while (true)")
            line("{")
            try scoped { emitter in
                emitter.emitLoopGuardCheck(guardName)
                if let continuing, emitter.containsContinue(body) {
                    // `continue` must run the `continuing` block, so the body goes into a
                    // `do { … } while (false)`: a C++ `continue` there leaves the `do` and falls into
                    // the continuing statements. A `break` would only leave the `do` as well, so the
                    // body's own breaks set a flag that is checked right after it (`break` in a nested
                    // loop or switch targets that construct and stays a plain `break`). This is the
                    // shape Naga and Tint emit for every `for`/`while` that uses `continue`, so it
                    // reaches here from any shader that went through wgpu, Bevy or SPIR-V.
                    let flag = emitter.declareBreakFlag()
                    emitter.line("do")
                    emitter.line("{")
                    emitter.breakFlag = flag
                    try emitter.scoped { inner in try inner.statements(body) }
                    emitter.breakFlag = nil
                    emitter.line("} while (false);")
                    emitter.line("if (\(flag)) { break; }")
                    try emitter.statements(continuing)
                } else {
                    try emitter.statements(body)
                    if let continuing { try emitter.statements(continuing) }
                }
            }
            line("}")
            breakFlag = savedBreakFlag

        case .switchStatement(let subject, let cases):
            // A WGSL `break` inside a case leaves the switch, as in C++ — not the loop around it.
            let savedBreakFlag = breakFlag
            breakFlag = nil
            defer { breakFlag = savedBreakFlag }
            line("switch (\(try expression(subject)))")
            line("{")
            try indented { emitter in
                for switchCase in cases {
                    if switchCase.isDefault {
                        emitter.line("default:")
                    }
                    for selector in switchCase.selectors {
                        emitter.line("case \(try emitter.expression(selector)):")
                    }
                    emitter.line("{")
                    try emitter.scoped { inner in
                        try inner.statements(switchCase.body)
                        // WGSL cases do not fall through.
                        inner.line("break;")
                    }
                    emitter.line("}")
                }
            }
            line("}")

        case .returnStatement(let value):
            if let value {
                line("return \(try expression(value));")
            } else {
                line("return;")
            }

        case .breakStatement:
            if let breakFlag {
                // Inside the `do { … } while (false)` wrapper of a `loop` with `continuing` — see `.loopStatement`.
                line("{ \(breakFlag) = true; break; }")
            } else {
                line("break;")
            }
        case .continueStatement:
            line("continue;")
        case .discardStatement:
            // MSL's `discard_fragment()` is **not an immediate return** — the code after it keeps
            // running and the discarded fragment corrupts storage buffers and textures. The spec
            // forbids that, so here we only mark, mask writes with the flag, and discard for real at
            // the end of the entry point. (Ending with an immediate `return` would break the
            // derivatives neighbouring threads in the same quad rely on.)
            if discardFlagUsers.isEmpty {
                line("discard_fragment();")
            } else {
                line("\(Self.discardFlagMSLName) = true;")
            }
        }
    }

    /// Whether this statement is **a write that must not run after `discard`**.
    ///
    /// The targets are exactly what the spec names — storage buffers, textures and atomics.
    /// Writes to locals never leave this fragment, so they are not masked.
    private func masksAfterDiscard(_ statement: WGSLStatement) -> Bool {
        guard !discardFlagUsers.isEmpty else { return false }
        switch statement {
        case .assignment(let target, _, _), .increment(let target), .decrement(let target):
            guard let root = Self.rootIdentifier(of: target) else { return false }
            return storageGlobalNames.contains(root)
        case .expressionStatement(.call(let callee, _, _)):
            return callee == "textureStore" || (callee.hasPrefix("atomic") && callee != "atomicLoad")
        default:
            return false
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

    /// Attaches **one more exit condition** to a loop that may not terminate.
    ///
    /// An infinite loop is undefined behaviour in C++. The compiler may assume "this loop ends
    /// eventually" and, on that assumption, delete or reorder the surrounding code — with no guarantee
    /// about the resulting code when the loop really does not end. Tint adds this for the same reason.
    ///
    /// The bound is where u32 stops counting, so **the behaviour of a correct shader is unchanged**
    /// (a loop reaching it has already stalled the GPU for minutes).
    private mutating func declareLoopGuard() -> String {
        loopGuardCounter += 1
        let name = "wgpu_loop_guard_\(loopGuardCounter)"
        line("uint \(name) = 0u;")
        return name
    }

    /// The flag a `loop { … continuing { … } }` body's `break` sets (see `.loopStatement`).
    private mutating func declareBreakFlag() -> String {
        loopGuardCounter += 1
        let name = "wgpu_break_\(loopGuardCounter)"
        line("bool \(name) = false;")
        return name
    }

    private mutating func emitLoopGuardCheck(_ name: String) {
        line("if (\(name) >= 4294967294u) { break; }")
        line("\(name) = \(name) + 1u;")
    }

    /// A simple statement without a semicolon (also used in a for header).
    ///
    /// A declaration **emits its initializer first** — under WGSL's point-of-declaration rule the
    /// initializer sees the outer (unshadowed) name, so registering the rename before the initializer
    /// would break `var v = v;`.
    private mutating func simpleStatement(_ statement: WGSLStatement) throws -> String {
        switch statement {
        case .letDeclaration(let name, let type, let value), .constDeclaration(let name, let type, let value):
            let typeText = try type.map { try MSLTypeMapping.type($0, module: module) } ?? "auto"
            let valueText = try expression(value)
            // Resolved before the name is declared — the initializer sees the outer meaning of every name.
            let target = pointerTarget(of: value)
            let parameterTarget = target == nil ? parameterPointee(of: value) : nil
            let declared = declaredName(name)
            localBindings[name] = target.map(LocalBinding.pointer) ?? parameterTarget.map(LocalBinding.parameterPointer) ?? .value
            return "const \(typeText) \(declared) = \(valueText)"

        case .varDeclaration(let name, let type, let value):
            guard let type else {
                guard let value else {
                    throw WGPUError.validation("WGSL: var '\(name)' has neither a type nor an initializer")
                }
                let valueText = try expression(value)
                localBindings[name] = .value
                return "auto \(declaredName(name)) = \(valueText)"
            }
            let typeText = try MSLTypeMapping.type(type, module: module)
            let valueText = try value.map { try expression($0) }
            localBindings[name] = .value
            guard let valueText else { return "\(typeText) \(declaredName(name)){}" }
            return "\(typeText) \(declaredName(name)) = \(valueText)"

        case .assignment(let target, let op, let value):
            if case .index(let base, let subscriptExpression) = target {
                return try storeStatement(base: base, subscript: subscriptExpression, op: op) {
                    try $0.expression(value)
                }
            }
            return "\(try expression(target)) \(op) \(try expression(value))"

        case .increment(let target):
            if case .index(let base, let subscriptExpression) = target {
                return try storeStatement(base: base, subscript: subscriptExpression, op: "+=") { _ in "1" }
            }
            return "\(try expression(target)) += 1"
        case .decrement(let target):
            if case .index(let base, let subscriptExpression) = target {
                return try storeStatement(base: base, subscript: subscriptExpression, op: "-=") { _ in "1" }
            }
            return "\(try expression(target)) -= 1"
        case .expressionStatement(let inner):
            return try expression(inner)
        default:
            throw WGPUError.validation("WGSL: this statement cannot appear here")
        }
    }

    /// The emitted name of a local declaration.
    ///
    /// When it collides with an injected global (a name that became a function parameter) we register
    /// a substitute — C++ does not allow redefining a parameter in the function's top-level block, and
    /// the argument list passing globals at the call site must still refer to the original name.
    /// References in this scope afterwards use the substitute.
    private mutating func declaredName(_ name: String) -> String {
        guard injectedGlobalNames.contains(name) else { return MSLTypeMapping.identifier(name) }
        let renamed = "wgpu_shadow_\(name)"
        localRenames[name] = renamed
        return renamed
    }

    private func containsContinue(_ statements: [WGSLStatement]) -> Bool {
        for statement in statements {
            switch statement {
            case .continueStatement:
                return true
            case .block(let inner), .whileStatement(_, let inner), .forStatement(_, _, _, let inner):
                if containsContinue(inner) { return true }
            case .ifStatement(let clauses, let elseBody):
                for clause in clauses where containsContinue(clause.body) { return true }
                if let elseBody, containsContinue(elseBody) { return true }
            case .switchStatement(_, let cases):
                for switchCase in cases where containsContinue(switchCase.body) { return true }
            default:
                // A `continue` in a nested loop belongs to that loop.
                break
            }
        }
        return false
    }

    // MARK: - Expressions

    private mutating func expression(_ expression: WGSLExpression) throws -> String {
        switch expression {
        case .intLiteral(let text):
            return normalizeIntLiteral(text)
        case .floatLiteral(let text):
            return normalizeFloatLiteral(text)
        case .boolLiteral(let value):
            return value ? "true" : "false"
        case .identifier(let name):
            if let renamed = localRenames[name] { return renamed }
            return MSLTypeMapping.identifier(name)
        case .unary(let op, let operand):
            // A space between stacked prefix operators: `- -b` glued together is C++'s `--b`, a
            // pre-decrement that mutates `b` and yields `b - 1` (`- - -b` is `-(--b)`).
            if case .unary = operand { return "\(op) \(try self.expression(operand))" }
            return "\(op)\(try self.expression(operand))"
        case .binary(let op, let left, let right):
            // Operations WGSL defines but C++ leaves undefined — routed through a prelude helper.
            // Which type it is cannot be known here, so the judgement goes to C++ overloading:
            // integers get the guard, floats and abstract ints pass straight through (`MSLPrelude`).
            if let helper = Self.guardedBinaryOperators[op] {
                return "\(helper)(\(try self.expression(left)), \(try self.expression(right)))"
            }
            return "\(try self.expression(left)) \(op) \(try self.expression(right))"
        case .paren(let inner):
            return "(\(try self.expression(inner)))"
        case .member(let base, let name):
            return "\(try concreteExpression(base)).\(MSLTypeMapping.identifier(name))"
        case .index(let base, let subscriptExpression):
            return try indexExpression(base: base, subscript: subscriptExpression)
        case .addressOf(let operand):
            return "&\(try self.expression(operand))"
        case .dereference(let operand):
            return "*\(try self.expression(operand))"
        case .call(let callee, let typeArguments, let arguments):
            return try call(callee: callee, typeArguments: typeArguments, arguments: arguments)
        }
    }

    /// Indexing — **prevents out-of-range access** (the WebGPU spec's robustness).
    ///
    /// Indices usually come from uniforms or storage, so they are ultimately values the bundle (JS)
    /// decides. Left alone, a shader reads or overwrites adjacent GPU memory, and a large excursion
    /// kills the command buffer with a page fault.
    ///
    /// It splits on whether the bound is **in the type**:
    /// - fixed-size arrays, vectors and matrices → `wgpu_at` knows the size as a C++ template argument;
    /// - runtime-sized arrays (`array<T>`) → they arrive as a pointer with no size in the type, so the
    ///   element count comes from the existing **buffer size table** (the one `arrayLength()` uses).
    private mutating func indexExpression(
        base: WGSLExpression,
        subscript subscriptExpression: WGSLExpression
    ) throws -> String {
        let target = try concreteExpression(base)
        let index = try expression(subscriptExpression)
        if let count = try runtimeArrayCount(of: base) {
            return "wgpu_at_n(\(target), \(index), \(count))"
        }
        return "wgpu_at(\(target), \(index))"
    }

    /// A statement **writing** in index position (`a[i] = v`, `a[i] += v`).
    ///
    /// A separate helper is needed, unlike reads, because of **vector components** — in MSL `v[i]`
    /// cannot bind to a reference (`non-const reference cannot bind to vector element`), so returning
    /// a clamped reference does not work. Finishing the assignment inside the helper covers arrays,
    /// matrices and vectors under the same name; a runtime-sized array takes `wgpu_store_n` instead.
    ///
    /// A compound assignment (`+=`) builds the read expression once more — the target is an identifier or member, so there is no side effect.
    private mutating func storeStatement(
        base: WGSLExpression,
        subscript subscriptExpression: WGSLExpression,
        op: String,
        value: (inout MSLEmitter) throws -> String
    ) throws -> String {
        let target = try concreteExpression(base)
        let index = try expression(subscriptExpression)
        var valueText = try value(&self)
        if op != "=" {
            let read = try indexExpression(base: base, subscript: subscriptExpression)
            valueText = "\(read) \(String(op.dropLast())) (\(valueText))"
        }
        if let count = try runtimeArrayCount(of: base) {
            return "wgpu_store_n(\(target), \(index), \(count), \(valueText))"
        }
        return "wgpu_store(\(target), \(index), \(valueText))"
    }

    /// The MSL expression for the element count, when this expression is a **runtime-sized array**.
    ///
    /// The same calculation as `arrayLength(&x)` — the buffer's byte count divided by the element size.
    /// So the list of functions using the table (`needsBufferSizes`) counts this use as well. A pointer
    /// can only be formed from a global named in the same function, so that list already covers `(*p)[i]`.
    ///
    /// A runtime-sized array behind a **pointer parameter** has no findable bound (the caller picks the
    /// buffer), so it is refused here with a WGSL-level message.
    ///
    /// nil sends the index down the plain `wgpu_at` path — where a runtime-sized array **does not compile**
    /// (the prelude deletes the unbounded pointer overloads), so a shape missed here still fails loudly.
    private func runtimeArrayCount(of expression: WGSLExpression) throws -> String? {
        guard let place = storagePlace(of: expression) else {
            if case .array(_, nil)? = parameterPlaceType(of: expression) {
                throw WGPUError.unsupported(
                    "WGSL: indexing a runtime-sized array through a pointer parameter is not supported"
                )
            }
            return nil
        }
        return try runtimeArrayElementCount(at: place)
    }

    /// The expression used as the target of a swizzle or index.
    ///
    /// An AbstractInt constant-expression vector is normally emitted as a proxy (so the type freezes
    /// from context), but a proxy has no component access such as `.xyz`. In that position it is
    /// emitted as a settled f32 vector.
    private mutating func concreteExpression(_ expression: WGSLExpression) throws -> String {
        if case .call(let callee, let typeArguments, let arguments) = expression,
           MSLPrelude.inferredVectorConstructors.contains(callee),
           typeArguments.isEmpty,
           Self.isAbstractIntegerArguments(arguments) {
            let size = Int(callee.dropFirst(3)) ?? 4
            let emitted = try arguments.map { try self.expression($0) }
            return "float\(size)(\(emitted.joined(separator: ", ")))"
        }
        return try self.expression(expression)
    }

    /// `1u` → `1u`, `1i` → `1`, `0xFFi` → `0xFF`.
    private func normalizeIntLiteral(_ text: String) -> String {
        if text.hasSuffix("i") { return String(text.dropLast()) }
        return text
    }

    /// `1.0f` → `1.0f`, `1.0h` → `1.0h`, `1.` → `1.0`.
    private func normalizeFloatLiteral(_ text: String) -> String {
        var value = text
        if value.hasSuffix(".") { value += "0" }
        if value.hasSuffix("f"), !value.contains(".") , !value.lowercased().contains("e") {
            value = String(value.dropLast()) + ".0f"
        }
        return value
    }

    private mutating func call(
        callee: String,
        typeArguments: [WGSLType],
        arguments: [WGSLExpression]
    ) throws -> String {
        if let reason = MSLTypeMapping.unsupportedBuiltins[callee] {
            throw WGPUError.unsupported("WGSL: \(callee)() — \(reason)")
        }
        // A user function or struct constructor wins over every builtin spelling below. WGSL lets a
        // module declare `fn textureBicubic(…)`, `fn atomicIncrement(…)` or `fn max(…)` and a
        // `struct textureInfo`; dispatching on the name's prefix first rejected the former as
        // "unsupported texture builtin" and handed the latter to the prelude's redirect.
        if module.functionNamed(callee) != nil {
            let emitted = try arguments.map { try expression($0) }
            var extra = threadedGlobals(for: callee).map { MSLTypeMapping.identifier($0.name) }
            if needsBufferSizes.contains(callee) { extra.append(Self.bufferSizesName) }
            return "\(MSLTypeMapping.functionName(callee))(\((emitted + extra).joined(separator: ", ")))"
        }
        if structNames.contains(callee) {
            // A struct constructor is aggregate initialization in MSL (C++).
            return try structConstructor(callee, arguments: try arguments.map { try expression($0) })
        }

        // Before anything below picks arguments out by index.
        try Self.checkArity(callee, count: arguments.count)

        if callee == "bitcast" {
            guard let target = typeArguments.first else {
                throw WGPUError.validation("WGSL: the form must be bitcast<T>(x)")
            }
            return "as_type<\(try MSLTypeMapping.type(target, module: module))>(\(try expression(arguments[0])))"
        }

        if callee == "arrayLength" {
            return try arrayLengthExpression(arguments)
        }
        if callee.hasPrefix("texture") {
            return try textureCall(callee: callee, arguments: arguments)
        }
        if callee.hasPrefix("atomic") {
            return try atomicCall(callee: callee, arguments: arguments)
        }
        switch callee {
        case "workgroupBarrier":
            return "threadgroup_barrier(mem_flags::mem_threadgroup)"
        case "storageBarrier":
            return "threadgroup_barrier(mem_flags::mem_device)"
        case "textureBarrier":
            return "threadgroup_barrier(mem_flags::mem_texture)"
        case "quantizeToF16":
            return "float(half(\(try expression(arguments[0]))))"
        default:
            break
        }

        let emitted = try arguments.map { try expression($0) }

        // A vector constructor with the component type omitted.
        if MSLPrelude.inferredVectorConstructors.contains(callee), typeArguments.isEmpty {
            // With every argument an integer literal there is no basis to infer. WGSL's AbstractInt
            // follows the context type, and in a vector constructor an f32 context is overwhelmingly
            // common, so we choose that (`vec3(1)` = white). Spell out `vec3u(…)` when an integer
            // vector is needed — docs/WGSL.md §4.
            if Self.isAbstractIntegerArguments(arguments) {
                // WGSL's AbstractInt constant expressions freeze into the context type. Emit a proxy
                // and hand that decision to C++ conversion operators (docs/WGSL.md §2-1).
                let size = Int(callee.dropFirst(3)) ?? 4
                return "wgpu_aint\(size)(\(emitted.joined(separator: ", ")))"
            }
            return "wgpu_\(callee)(\(emitted.joined(separator: ", ")))"
        }
        // The same goes for builtins needing literal promotion (`max(x, 0)`).
        if let helper = MSLPrelude.redirectedBuiltins[callee] {
            return "\(helper)(\(emitted.joined(separator: ", ")))"
        }
        if callee == "array" {
            // With the element type omitted, guess it from the first argument's constructor name (`array(vec2f(…), …)`).
            let arrayType = arrayConstructorType(typeArguments: typeArguments, argumentCount: emitted.count)
                ?? arguments.first.flatMap(inferredType(of:)).map {
                    WGSLType.array(element: $0, count: .intLiteral(String(emitted.count)))
                }
            guard let arrayType else {
                throw WGPUError.unsupported(
                    "WGSL: cannot determine the element type of an array(…) constructor — spell it out as `array<T, N>(…)`"
                )
            }
            // An empty brace list is value-initialization — the zero value `array<T, N>()` asks for.
            return "\(try MSLTypeMapping.type(arrayType, module: module)){\(emitted.joined(separator: ", "))}"
        }
        // WGSL defines f32 → i32/u32 conversion as **saturating** (a C++ cast is UB out of range).
        // Only a single argument is a conversion; a constructor listing components already passed each argument through its own.
        if emitted.count == 1, let helper = Self.saturatingConversion(callee, typeArguments: typeArguments) {
            return "\(helper)(\(emitted[0]))"
        }
        if let constructor = try vectorOrMatrixConstructor(callee, typeArguments: typeArguments) {
            return "\(constructor)(\(emitted.joined(separator: ", ")))"
        }
        if MSLTypeMapping.scalarConstructors.contains(callee) {
            return "\(MSLTypeMapping.scalar(callee))(\(emitted.joined(separator: ", ")))"
        }

        // Anything else passes through by name (builtins, alias constructors) — through `identifier(_:)`
        // so it matches the declaration it refers to and cannot reach an internal `wgpu_` helper.
        let name = MSLTypeMapping.renamedBuiltins[callee] ?? MSLTypeMapping.identifier(callee)
        return "\(name)(\(emitted.joined(separator: ", ")))"
    }

    /// Rejects a builtin in `MSLTypeMapping.builtinArities` called with the wrong argument count —
    /// as a compile error, not an out-of-range trap in the lowering that indexes the arguments.
    private static func checkArity(_ callee: String, count: Int) throws {
        guard let arity = MSLTypeMapping.builtinArities[callee], !arity.contains(count) else { return }
        let expected = arity.lowerBound == arity.upperBound
            ? "\(arity.lowerBound)"
            : "\(arity.lowerBound) to \(arity.upperBound)"
        let noun = arity.upperBound == 1 ? "argument" : "arguments"
        throw WGPUError.validation("WGSL: \(callee)() expects \(expected) \(noun), got \(count)")
    }

    /// The type an `array<T, N>(…)` / `array<T>(…)` constructor builds — the parser keeps the written
    /// array type as the sole type argument. Its length is the declared one (the zero-argument form
    /// has nothing else to go by); when there is none, or it cannot be computed here, the argument count.
    private func arrayConstructorType(typeArguments: [WGSLType], argumentCount: Int) -> WGSLType? {
        guard case .array(let element, let count)? = typeArguments.first else { return nil }
        if let count, argumentCount == 0 || WGSLLayout.constantValue(count, module: module) != nil {
            return .array(element: element, count: count)
        }
        return .array(element: element, count: .intLiteral(String(argumentCount)))
    }

    /// The saturating conversion helper name, when this is an integer-producing constructor (`i32`, `u32`, `vec3i`, `vec2<u32>`, …).
    ///
    /// The decision is made on strings **because there is no type inferencer**. Whether the argument
    /// is really f32 cannot be known here, so it goes to C++ overloading — if it is not f32 the helper just casts.
    private static func saturatingConversion(_ callee: String, typeArguments: [WGSLType]) -> String? {
        func target(_ scalar: String) -> String? {
            scalar == "i32" ? "wgpu_ftoi" : (scalar == "u32" ? "wgpu_ftou" : nil)
        }
        if MSLTypeMapping.scalarConstructors.contains(callee) { return target(callee) }
        // `vec3i(…)` / `vec3<i32>(…)` — passing the size as a template argument also accepts scalar broadcast.
        guard callee.hasPrefix("vec"), callee.count >= 4,
              let size = Int(callee.dropFirst(3).prefix(1)) else { return nil }
        let suffix = String(callee.dropFirst(4))
        let scalar: String
        if suffix.isEmpty {
            guard case .scalar(let named)? = typeArguments.first else { return nil }
            scalar = named
        } else if suffix == "i" {
            scalar = "i32"
        } else if suffix == "u" {
            scalar = "u32"
        } else {
            return nil
        }
        return target(scalar).map { "\($0)_n<\(size)>" }
    }

    private func vectorOrMatrixConstructor(_ callee: String, typeArguments: [WGSLType]) throws -> String? {
        if callee == "vec2" || callee == "vec3" || callee == "vec4" {
            let size = Int(callee.dropFirst(3))!
            // With the component type omitted, as in `vec3(…)`, treat it as f32 (by far the most common case in real shaders).
            guard case .scalar(let scalar)? = typeArguments.first ?? .scalar("f32") else { return nil }
            return "\(MSLTypeMapping.scalar(scalar))\(size)"
        }
        if callee.hasPrefix("mat"), callee.count >= 6 {
            let dimensions = callee.dropFirst(3).prefix(3).split(separator: "x")
            guard dimensions.count == 2, let columns = Int(dimensions[0]), let rows = Int(dimensions[1]) else {
                return nil
            }
            guard case .scalar(let scalar)? = typeArguments.first ?? .scalar("f32") else { return nil }
            return "\(MSLTypeMapping.scalar(scalar))\(columns)x\(rows)"
        }
        if let shorthand = WGSLParser.shorthandType(callee) {
            return try MSLTypeMapping.type(shorthand, module: module)
        }
        return nil
    }

    // MARK: - Texture / atomic builtins

    private mutating func textureCall(callee: String, arguments: [WGSLExpression]) throws -> String {
        guard let first = arguments.first else {
            throw WGPUError.validation("WGSL: \(callee)() has no texture argument")
        }
        let receiver = try expression(first)
        let texture = textureType(of: first)
        let dimension = texture?.dimension ?? "2d"
        let isArrayed = dimension.hasSuffix("_array")
        var rest = try arguments.dropFirst().map { try expression($0) }

        func integerCast(_ value: String) -> String {
            switch dimension {
            case "1d": return "uint(\(value))"
            case "3d": return "uint3(\(value))"
            default: return "uint2(\(value))"
            }
        }

        /// The optional trailing `offset` (a const-expression `vec2<i32>` / `vec3<i32>`). Metal takes it
        /// as the **last** argument, after any LOD option — dropping it silently samples the wrong texel.
        /// WGSL allows it only on 2d, 2d_array and 3d textures (depth ones included), never on cubes.
        func appendOffset(to parameters: inout [String]) throws {
            guard !rest.isEmpty else { return }
            guard rest.count == 1, ["2d", "2d_array", "3d"].contains(dimension) else {
                throw WGPUError.validation("WGSL: too many arguments to \(callee)() for a \(dimension) texture")
            }
            parameters.append("\(dimension == "3d" ? "int3" : "int2")(\(rest.removeFirst()))")
        }

        switch callee {
        case "textureSample", "textureSampleBias", "textureSampleLevel", "textureSampleGrad":
            guard rest.count >= 2 else {
                throw WGPUError.validation("WGSL: the form must be \(callee)(t, s, coords, …)")
            }
            let sampler = rest.removeFirst()
            let coordinates = rest.removeFirst()
            var parameters = [sampler, coordinates]
            if isArrayed, !rest.isEmpty { parameters.append(rest.removeFirst()) }
            switch callee {
            case "textureSampleLevel":
                guard !rest.isEmpty else { throw WGPUError.validation("WGSL: textureSampleLevel has no level") }
                parameters.append("level(\(rest.removeFirst()))")
            case "textureSampleBias":
                guard !rest.isEmpty else { throw WGPUError.validation("WGSL: textureSampleBias has no bias") }
                parameters.append("bias(\(rest.removeFirst()))")
            case "textureSampleGrad":
                guard rest.count >= 2 else { throw WGPUError.validation("WGSL: textureSampleGrad has no derivatives") }
                let dx = rest.removeFirst()
                let dy = rest.removeFirst()
                let gradient: String
                switch dimension {
                case "3d": gradient = "gradient3d"
                case "cube", "cube_array": gradient = "gradientcube"
                default: gradient = "gradient2d"
                }
                parameters.append("\(gradient)(\(dx), \(dy))")
            default:
                break
            }
            try appendOffset(to: &parameters)
            return "\(receiver).sample(\(parameters.joined(separator: ", ")))"

        case "textureSampleCompare", "textureSampleCompareLevel":
            guard rest.count >= 3 else {
                throw WGPUError.validation("WGSL: the form must be \(callee)(t, s, coords, depth_ref)")
            }
            let sampler = rest.removeFirst()
            let coordinates = rest.removeFirst()
            var parameters = [sampler, coordinates]
            if isArrayed, rest.count >= 2 { parameters.append(rest.removeFirst()) }
            parameters.append(rest.removeFirst())
            // `…CompareLevel` samples mip level 0 — spelled out, because Metal's option-less overload
            // takes an implicit LOD (fragment-only) and the offset must follow the LOD option.
            if callee == "textureSampleCompareLevel" { parameters.append("level(0)") }
            try appendOffset(to: &parameters)
            return "\(receiver).sample_compare(\(parameters.joined(separator: ", ")))"

        case "textureLoad":
            guard !rest.isEmpty else { throw WGPUError.validation("WGSL: textureLoad has no coordinates") }
            var parameters = [integerCast(rest.removeFirst())]
            parameters += rest
            return "\(receiver).read(\(parameters.joined(separator: ", ")))"

        case "textureStore":
            guard rest.count >= 2 else {
                throw WGPUError.validation("WGSL: the form must be textureStore(t, coords, value)")
            }
            let coordinates = integerCast(rest.removeFirst())
            let value = rest.removeLast()
            var parameters = [value, coordinates]
            parameters += rest   // the array layer index
            return "\(receiver).write(\(parameters.joined(separator: ", ")))"

        case "textureDimensions":
            let level = rest.first.map { "\($0)" } ?? ""
            switch dimension {
            case "1d":
                return "uint(\(receiver).get_width(\(level)))"
            case "3d":
                return "uint3(\(receiver).get_width(\(level)), \(receiver).get_height(\(level)), "
                    + "\(receiver).get_depth(\(level)))"
            default:
                return "uint2(\(receiver).get_width(\(level)), \(receiver).get_height(\(level)))"
            }

        case "textureSampleBaseClampToEdge":
            guard rest.count >= 2 else {
                throw WGPUError.validation("WGSL: the form must be textureSampleBaseClampToEdge(t, s, coords)")
            }
            let clampSampler = rest.removeFirst()
            let clampCoordinates = rest.removeFirst()
            return "wgpu_sample_base_clamp(\(receiver), \(clampSampler), \(clampCoordinates))"

        case "textureNumLayers":
            return "\(receiver).get_array_size()"
        case "textureNumLevels":
            return "\(receiver).get_num_mip_levels()"
        case "textureNumSamples":
            return "\(receiver).get_num_samples()"
        default:
            throw WGPUError.unsupported("WGSL: unsupported texture builtin \(callee)()")
        }
    }

    private func textureType(of expression: WGSLExpression) -> WGSLTextureType? {
        guard case .identifier(let name) = expression else { return nil }
        return textureScope[name]
    }

    private mutating func atomicCall(callee: String, arguments: [WGSLExpression]) throws -> String {
        let emitted = try arguments.map { try expression($0) }
        // Argument counts were checked against `builtinArities` by `call`.
        if let operation = MSLTypeMapping.atomicFetchOperations[callee] {
            return "\(operation)(\(emitted[0]), \(emitted[1]), memory_order_relaxed)"
        }
        switch callee {
        case "atomicLoad":
            return "atomic_load_explicit(\(emitted[0]), memory_order_relaxed)"
        case "atomicStore":
            return "atomic_store_explicit(\(emitted[0]), \(emitted[1]), memory_order_relaxed)"
        case "atomicExchange":
            return "atomic_exchange_explicit(\(emitted[0]), \(emitted[1]), memory_order_relaxed)"
        default:
            throw WGPUError.unsupported("WGSL: unsupported atomic builtin \(callee)()")
        }
    }
}
