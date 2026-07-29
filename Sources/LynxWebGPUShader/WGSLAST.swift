import Foundation

// WGSL 구문 트리. 트랜스파일에 필요한 만큼만 담는다 —
// 타입 추론기가 아니라 **구문 번역기**이므로 표현식의 타입은 대부분 보존하지 않는다.
// (MSL이 C++ 기반이라 대부분의 연산자·생성자 의미가 그대로 옮겨진다.)

/// `@vertex`, `@group(0)`, `@location(2)` 같은 속성.
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
    /// count == nil 이면 런타임 크기 배열 (storage 버퍼 전용).
    case array(element: WGSLType, count: WGSLExpression?)
    case atomic(WGSLType)
    case pointer(space: String, element: WGSLType, access: String?)
    case texture(WGSLTextureType)
    case sampler(comparison: Bool)
    /// 구조체 이름 또는 타입 별칭.
    case named(String)
    case void

    /// 런타임 크기 배열인가 (storage 버퍼 바인딩 판별용).
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
    /// sampled/multisampled 텍스처의 성분 타입 (f32/i32/u32).
    var sampleType: WGSLType?
    /// storage 텍스처의 포맷 이름 ("rgba8unorm" …).
    var format: String?
    /// storage 텍스처의 접근 모드 ("write" / "read" / "read_write").
    var access: String?
}

indirect enum WGSLExpression: Equatable {
    case intLiteral(String)
    case floatLiteral(String)
    case boolLiteral(Bool)
    case identifier(String)
    case unary(op: String, WGSLExpression)
    case binary(op: String, WGSLExpression, WGSLExpression)
    /// `vec4<f32>(…)`, `textureSample(…)`, 사용자 함수 호출을 모두 포함한다.
    case call(callee: String, typeArguments: [WGSLType], arguments: [WGSLExpression])
    case member(WGSLExpression, String)
    case index(WGSLExpression, WGSLExpression)
    case paren(WGSLExpression)
    case addressOf(WGSLExpression)
    case dereference(WGSLExpression)
}

indirect enum WGSLStatement: Equatable {
    /// `let x: T = e;` — 불변 바인딩.
    case letDeclaration(name: String, type: WGSLType?, value: WGSLExpression)
    /// `var x: T = e;` — 함수 지역 가변 변수.
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
    /// 값을 버리는 함수 호출 (`textureStore(…);`).
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

/// 모듈 스코프 변수 — 유니폼/스토리지 버퍼, 텍스처, 샘플러, workgroup·private 변수.
struct WGSLGlobalVariable: Equatable {
    var attributes: [WGSLAttribute]
    var name: String
    /// uniform / storage / workgroup / private / (nil = handle 타입: 텍스처·샘플러)
    var addressSpace: String?
    var access: String?
    var type: WGSLType
    var initializer: WGSLExpression?

    var group: Int? { attributes.group }
    var binding: Int? { attributes.binding }
    /// 바인드 그룹 슬롯을 차지하는 변수인가 (workgroup/private는 아니다).
    var isResource: Bool { group != nil && binding != nil }
}

/// 모듈 스코프 상수 (`const PI = 3.14;`) 또는 파이프라인 상수 (`override scale: f32 = 1.0;`).
struct WGSLModuleConstant: Equatable {
    var name: String
    var type: WGSLType?
    /// `override`는 기본값 없이 선언될 수 있다 (`override size: f32;`).
    /// 그 경우 파이프라인 생성 시 `constants`로 값을 받아야 한다.
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
