import Foundation

// The WGSL syntax tree. It carries only what transpilation needs —
// this is a **syntax translator**, not a type inferencer, so most expression types are not preserved.
// (MSL is C++-based, so most operator and constructor meanings carry across unchanged.)

/// An attribute such as `@vertex`, `@group(0)` or `@location(2)`.
struct WGSLAttribute: Equatable {
    var name: String
    var arguments: [String]

    var firstInt: Int? { arguments.first.flatMap { Int($0.filter(\.isNumber)) } }
}

extension Array where Element == WGSLAttribute {
    func first(named name: String) -> WGSLAttribute? { first { $0.name == name } }
    func has(_ name: String) -> Bool { contains { $0.name == name } }
    var location: Int? { first(named: "location")?.firstInt }
    var builtin: String? { first(named: "builtin")?.arguments.first }
    var group: Int? { first(named: "group")?.firstInt }
    var binding: Int? { first(named: "binding")?.firstInt }
}

indirect enum WGSLType: Equatable {
    /// f32 / i32 / u32 / bool / f16
    case scalar(String)
    case vector(size: Int, element: WGSLType)
    case matrix(columns: Int, rows: Int, element: WGSLType)
    /// count == nil means a runtime-sized array (storage buffers only).
    case array(element: WGSLType, count: WGSLExpression?)
    case atomic(WGSLType)
    case pointer(space: String, element: WGSLType, access: String?)
    case texture(WGSLTextureType)
    case sampler(comparison: Bool)
    /// A struct name or a type alias.
    case named(String)
    case void

    /// Whether it is a runtime-sized array (used to identify storage buffer bindings).
    var isRuntimeArray: Bool {
        if case .array(_, let count) = self { return count == nil }
        return false
    }
}

struct WGSLTextureType: Equatable {
    enum Kind: String {
        case sampled, storage, depth, multisampled, depthMultisampled, external
    }

    var kind: Kind
    /// "1d" "2d" "2d_array" "3d" "cube" "cube_array"
    var dimension: String
    /// Component type of a sampled/multisampled texture (f32/i32/u32).
    var sampleType: WGSLType?
    /// Format name of a storage texture ("rgba8unorm", …).
    var format: String?
    /// Access mode of a storage texture ("write" / "read" / "read_write").
    var access: String?
}

indirect enum WGSLExpression: Equatable {
    case intLiteral(String)
    case floatLiteral(String)
    case boolLiteral(Bool)
    case identifier(String)
    case unary(op: String, WGSLExpression)
    case binary(op: String, WGSLExpression, WGSLExpression)
    /// Covers `vec4<f32>(…)`, `textureSample(…)` and user function calls alike.
    case call(callee: String, typeArguments: [WGSLType], arguments: [WGSLExpression])
    case member(WGSLExpression, String)
    case index(WGSLExpression, WGSLExpression)
    case paren(WGSLExpression)
    case addressOf(WGSLExpression)
    case dereference(WGSLExpression)
}

indirect enum WGSLStatement: Equatable {
    /// `let x: T = e;` — an immutable binding.
    case letDeclaration(name: String, type: WGSLType?, value: WGSLExpression)
    /// `var x: T = e;` — a function-local mutable variable.
    case varDeclaration(name: String, type: WGSLType?, value: WGSLExpression?)
    /// `const x = e;`
    case constDeclaration(name: String, type: WGSLType?, value: WGSLExpression)
    case assignment(target: WGSLExpression, op: String, value: WGSLExpression)
    case increment(WGSLExpression)
    case decrement(WGSLExpression)
    case ifStatement(condition: WGSLExpression, then: [WGSLStatement], elseBranch: WGSLElseBranch?)
    case forStatement(
        initializer: WGSLStatement?,
        condition: WGSLExpression?,
        update: WGSLStatement?,
        body: [WGSLStatement]
    )
    case whileStatement(condition: WGSLExpression, body: [WGSLStatement])
    case loopStatement(body: [WGSLStatement], continuing: [WGSLStatement]?)
    case switchStatement(subject: WGSLExpression, cases: [WGSLSwitchCase])
    case returnStatement(WGSLExpression?)
    case breakStatement
    case continueStatement
    case discardStatement
    /// A function call whose value is discarded (`textureStore(…);`).
    case expressionStatement(WGSLExpression)
    case block([WGSLStatement])
}

indirect enum WGSLElseBranch: Equatable {
    case block([WGSLStatement])
    case chained(WGSLStatement)
}

struct WGSLSwitchCase: Equatable {
    var selectors: [WGSLExpression]
    var isDefault: Bool
    var body: [WGSLStatement]
}

struct WGSLStructMember: Equatable {
    var attributes: [WGSLAttribute]
    var name: String
    var type: WGSLType
}

struct WGSLStruct: Equatable {
    var name: String
    var members: [WGSLStructMember]
}

/// A module-scope variable — uniform/storage buffers, textures, samplers, workgroup and private variables.
struct WGSLGlobalVariable: Equatable {
    var attributes: [WGSLAttribute]
    var name: String
    /// uniform / storage / workgroup / private / (nil = a handle type: texture or sampler)
    var addressSpace: String?
    var access: String?
    var type: WGSLType
    var initializer: WGSLExpression?

    var group: Int? { attributes.group }
    var binding: Int? { attributes.binding }
    /// Whether the variable occupies a bind group slot (workgroup and private do not).
    var isResource: Bool { group != nil && binding != nil }
}

/// A module-scope constant (`const PI = 3.14;`) or a pipeline constant (`override scale: f32 = 1.0;`).
struct WGSLModuleConstant: Equatable {
    var name: String
    var type: WGSLType?
    /// An `override` may be declared without a default (`override size: f32;`).
    /// In that case a value must be supplied through `constants` at pipeline creation.
    var value: WGSLExpression?
    var isOverride: Bool
}

struct WGSLParameter: Equatable {
    var attributes: [WGSLAttribute]
    var name: String
    var type: WGSLType
}

struct WGSLFunction: Equatable {
    var attributes: [WGSLAttribute]
    var name: String
    var parameters: [WGSLParameter]
    var returnAttributes: [WGSLAttribute]
    var returnType: WGSLType?
    var body: [WGSLStatement]

    var stage: WGSLStage? {
        if attributes.has("vertex") { return .vertex }
        if attributes.has("fragment") { return .fragment }
        if attributes.has("compute") { return .compute }
        return nil
    }

    var workgroupSize: (Int, Int, Int)? {
        guard let attribute = attributes.first(named: "workgroup_size") else { return nil }
        let values = attribute.arguments.map { Int($0) ?? 1 }
        return (values.count > 0 ? values[0] : 1, values.count > 1 ? values[1] : 1, values.count > 2 ? values[2] : 1)
    }
}

public enum WGSLStage: String, Equatable, Sendable {
    case vertex, fragment, compute
}

struct WGSLModule {
    var structs: [WGSLStruct] = []
    var globals: [WGSLGlobalVariable] = []
    var constants: [WGSLModuleConstant] = []
    var aliases: [(name: String, type: WGSLType)] = []
    var functions: [WGSLFunction] = []

    func structNamed(_ name: String) -> WGSLStruct? { structs.first { $0.name == name } }
    func functionNamed(_ name: String) -> WGSLFunction? { functions.first { $0.name == name } }
}
