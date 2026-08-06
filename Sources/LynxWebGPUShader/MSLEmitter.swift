import Foundation
import LynxWebGPUCore

/// WGSL 구문 트리를 Metal Shading Language 소스로 방출한다.
///
/// 두 가지 구조적 변환이 핵심이다:
///
/// 1. **리소스 스레딩** — MSL에는 가변 전역이 없다. WGSL의 모듈 스코프 변수(유니폼/스토리지/텍스처/
///    샘플러/workgroup/private)는 진입점의 인자로 받아 호출 그래프를 따라 **함수 인자로 내려보낸다**.
///    (`WGSLReflectionBuilder.transitiveGlobalUsage`가 어느 함수가 무엇을 쓰는지 계산한다.)
/// 2. **진입점 래핑** — WGSL 진입점의 시그니처를 그대로 두고(`…_inner`), 스테이지 I/O 속성을 붙인
///    래퍼를 따로 만든다. 구조체를 유니폼 버퍼와 정점 I/O에 동시에 쓰는 셰이더에서
///    `[[attribute]]`/`[[user]]` 같은 속성이 버퍼 레이아웃을 오염시키지 않게 하기 위해서다.
struct MSLEmitter {
    private let module: WGSLModule
    private let reflection: WGSLShaderReflection
    private let bindings: WGSLBindingAssignment
    private let uniformStructs: Set<String>
    private let usage: [String: Set<String>]
    private let globalsByName: [String: WGSLGlobalVariable]
    private let structNames: Set<String>
    /// 버퍼 크기 표를 인자로 받아야 하는 함수들 — `arrayLength()`를 쓰거나
    /// **런타임 크기 배열을 인덱싱**한다 (후자는 범위를 자르려면 상한이 필요하다).
    private let needsBufferSizes: Set<String>

    /// 버퍼 크기 표 인자 이름.
    private static let bufferSizesName = "wgpu_buffer_sizes"

    /// threadgroup 안의 스레드 번호 — `var<workgroup>` 0 초기화가 이 값으로 일을 나눈다.
    /// 셰이더가 이미 `@builtin(local_invocation_index)`를 받고 있으면 같은 선언이라 합쳐진다.
    private static let localInvocationIndexName = "wgpu_bi_local_invocation_index"

    /// WGSL은 정의하는데 C++에서는 **정의되지 않은** 이항 연산 → 프렐류드 헬퍼.
    ///
    /// - `/`·`%` — 0으로 나누기와 `INT_MIN / -1`
    /// - `<<`·`>>` — 폭 이상의 시프트
    ///
    /// 셋 다 그대로 두면 드라이버가 무엇을 하든 이상하지 않고, 최적화가 "일어날 수 없는 일"로
    /// 보고 주변 코드를 지워 버리기도 한다.
    private static let guardedBinaryOperators = [
        "%": "wgpu_mod", "/": "wgpu_div", "<<": "wgpu_shl", ">>": "wgpu_shr",
    ]

    /// 현재 함수 스코프에서 이름 → 텍스처 타입 (텍스처 내장 함수를 메서드 호출로 바꿀 때 필요).
    private var textureScope: [String: WGSLTextureType] = [:]
    /// 현재 함수에 인자로 주입된 전역 이름들.
    ///
    /// WGSL은 전역을 가리는 지역 선언이 합법이지만, 주입 때문에 그 전역이 **매개변수**가 되면
    /// C++에서는 함수 최상위 블록의 재정의라 불법이다. 그래서 겹치는 지역 선언을 리네임한다.
    private var injectedGlobalNames: Set<String> = []
    /// 활성 섀도잉 리네임 (원래 이름 → 방출 이름). 블록 경계에서 저장/복원된다.
    private var localRenames: [String: String] = [:]
    /// `discard` 뒤의 쓰기를 가리는 합성 전역 이름. 모듈에 `discard`가 없으면 만들지 않는다.
    private static let discardFlagName = "wgpu_discarded"
    /// 그 플래그를 인자로 받아야 하는 함수들 (비어 있으면 변환 자체가 꺼진다).
    private let discardFlagUsers: Set<String>
    /// 스토리지 주소 공간 전역 이름 — `discard` 뒤에 쓰기를 가릴 대상을 고를 때 쓴다.
    private let storageGlobalNames: Set<String>

    /// 무한 루프 가드 변수 이름이 겹치지 않게 하는 번호 (중첩 루프).
    private var loopGuardCounter = 0
    private var output = ""
    private var indentLevel = 0

    init(module: WGSLModule, reflection: WGSLShaderReflection, bindings: WGSLBindingAssignment) {
        var module = module
        var usage = WGSLReflectionBuilder.transitiveGlobalUsage(module)

        // `discard` 뒤의 쓰기를 막을 플래그를 **합성 전역으로 끼워 넣는다.**
        //
        // 직접 인자로 흘려보내는 코드를 새로 쓰는 대신, 이미 있는 리소스 스레딩에 태운다 —
        // `private` 전역은 진입점 안의 지역 변수로 선언되고 호출 그래프를 따라 참조로 내려가므로,
        // 필요한 것이 "전역 하나와 그것을 쓰는 함수 목록"뿐이다.
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
        self.uniformStructs = WGSLLayout.uniformStructNames(module)
        self.usage = usage
        self.globalsByName = Dictionary(uniqueKeysWithValues: module.globals.map { ($0.name, $0) })
        self.structNames = Set(module.structs.map(\.name))
        self.needsBufferSizes = WGSLReflectionBuilder.functionsNeedingBufferSizes(in: module)
        self.storageGlobalNames = Set(module.globals.filter { $0.addressSpace == "storage" }.map(\.name))
    }

    // MARK: - 진입

    mutating func emit(entryPoints requested: [String]) throws -> String {
        output = ""
        line("#include <metal_stdlib>")
        line("#include <simd/simd.h>")
        line("using namespace metal;")
        line("")
        output += MSLPrelude.source + "\n\n"

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

        // 요청한 진입점에서 **호출로 닿는** 함수만 내보낸다. 모듈의 함수를 전부 내보내면,
        // 이 진입점이 부르지도 않는 함수가 참조하는 리소스까지 바인딩 표에 있어야 한다 —
        // `layout: "auto"`는 쓰는 것만 담으므로 그런 함수는 없는 바인딩을 찾다 실패한다
        // (`WGSLReflectionBuilder.functionsReachable` 참고).
        let reachable = WGSLReflectionBuilder.functionsReachable(from: requested, in: module)
        for function in module.functions where function.stage == nil && reachable.contains(function.name) {
            try emitFunction(function)
        }

        for name in requested {
            guard let function = module.functionNamed(name), function.stage != nil else {
                throw WGPUError.validation(
                    "셰이더에 진입점 '\(name)'이(가) 없다 (있는 것: "
                        + "\(reflection.entryPoints.map(\.name).joined(separator: ", ")))"
                )
            }
            try emitEntryPoint(function)
        }
        return output
    }

    /// 모듈 스코프 상수 / 파이프라인 상수.
    ///
    /// MSL은 `constant auto x = …`를 허용하지 않는다. 타입 주석이 없으면 매크로로 편다 —
    /// WGSL의 모듈 상수는 컴파일 타임 값이라 의미가 같다.
    private mutating func emitModuleConstant(_ constant: WGSLModuleConstant) throws {
        guard let value = constant.value else {
            throw WGPUError.validation(
                "override '\(constant.name)'에 값이 없다 — 파이프라인 생성 시 "
                    + "`constants: { \(constant.name): … }` 로 넘겨야 한다"
            )
        }
        if let type = constant.type ?? inferredType(of: value) {
            line("constant \(try MSLTypeMapping.type(type, module: module)) \(MSLTypeMapping.identifier(constant.name)) = \(try expression(value));")
        } else {
            line("#define \(MSLTypeMapping.identifier(constant.name)) (\(try expression(value)))")
        }
    }

    /// 인자가 전부 **접미사 없는 정수 상수식**인가 (WGSL의 AbstractInt).
    ///
    /// `vec2(4, 1)`, `vec4(1741651 * 1009, …)` 처럼 성분 타입을 정할 근거가 인자에 없는 경우다.
    /// `4u`처럼 접미사가 붙었거나 식별자가 섞이면 타입이 정해지므로 템플릿 추론에 맡긴다.
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

    /// `arrayLength(&buffer)` / `arrayLength(&buffer.member)` → 버퍼 크기 표 조회.
    ///
    /// Metal 셰이더는 버퍼 크기를 알 수 없다. 런타임이 예약 인덱스에 크기 표를 꽂아 주고,
    /// 여기서 (표에 담긴 바이트 수 − 배열 시작 오프셋) ÷ 원소 크기로 계산한다.
    private func arrayLengthExpression(_ arguments: [WGSLExpression]) throws -> String {
        var target = arguments.first ?? .identifier("")
        if case .addressOf(let inner) = target { target = inner }

        var globalName = ""
        var memberName: String?
        switch target {
        case .identifier(let name):
            globalName = name
        case .member(.identifier(let name), let field):
            globalName = name
            memberName = field
        default:
            throw WGPUError.unsupported(
                "WGSL: arrayLength()는 스토리지 버퍼 변수(또는 그 멤버)에만 쓸 수 있다"
            )
        }

        guard let global = globalsByName[globalName],
              let group = global.group, let binding = global.binding,
              let index = bindings.index(group: group, binding: binding) else {
            throw WGPUError.validation("arrayLength(): '\(globalName)'은(는) 바인딩된 스토리지 버퍼가 아니다")
        }

        var element: WGSLType
        var offset = 0
        if case .array(let inner, nil) = global.type, memberName == nil {
            element = inner
        } else if case .named(let structName) = global.type, let memberName,
                  let structure = WGSLLayout.resolveStruct(structName, module: module) {
            let placement = WGSLLayout.layout(
                of: structure, module: module, uniform: uniformStructs.contains(structName)
            )
            guard let member = placement.members.first(where: { $0.name == memberName }),
                  case .array(let inner, nil) = member.type else {
                throw WGPUError.unsupported("WGSL: arrayLength()의 대상이 런타임 크기 배열이 아니다")
            }
            element = inner
            offset = member.offset
        } else {
            throw WGPUError.unsupported("WGSL: arrayLength()의 대상이 런타임 크기 배열이 아니다")
        }

        let elementType = try MSLTypeMapping.type(element, module: module)
        let total = offset == 0
            ? "\(Self.bufferSizesName)[\(index)]"
            : "(\(Self.bufferSizesName)[\(index)] - \(offset)u)"
        return "(\(total) / uint(sizeof(\(elementType))))"
    }

    /// 초기값의 **구문**만 보고 타입을 짚어 본다 (타입 추론기가 아니라 생성자 이름을 읽는 것).
    /// 모듈 상수와 성분 타입이 생략된 `array(…)` 생성자에 쓴다.
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
        case .call(let callee, let typeArguments, let arguments):
            if structNames.contains(callee) { return .named(callee) }
            if MSLTypeMapping.scalarConstructors.contains(callee) { return .scalar(callee) }
            if let shorthand = WGSLParser.shorthandType(callee) { return shorthand }
            if callee.hasPrefix("vec"), let size = Int(callee.dropFirst(3)) {
                if case .scalar? = typeArguments.first {
                    return .vector(size: size, element: typeArguments[0])
                }
                // 성분 타입이 생략되면 인자에서 짚어 본다 (`vec2(-1.0, -1.0)` → vec2<f32>).
                // 인자가 전부 AbstractInt 상수식이면 방출기가 프록시를 내보내므로 타입을 못 적는다
                // (매크로/auto 경로로 흘려보내야 문맥에서 굳는다).
                if Self.isAbstractIntegerArguments(arguments) { return nil }
                guard let element = arguments.first.flatMap(inferredType(of:)),
                      case .scalar = element else { return nil }
                return .vector(size: size, element: element)
            }
            if callee == "array" {
                let element = typeArguments.first ?? arguments.first.flatMap(inferredType(of:))
                guard let element else { return nil }
                return .array(element: element, count: .intLiteral(String(arguments.count)))
            }
            return nil
        default:
            return nil
        }
    }

    // MARK: - 출력 보조

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

    /// 블록 스코프 경계 — 안에서 등록된 섀도잉 리네임이 블록 밖으로 새지 않게 한다.
    private mutating func scoped(_ body: (inout MSLEmitter) throws -> Void) rethrows {
        let saved = localRenames
        try indented(body)
        localRenames = saved
    }

    // MARK: - 구조체 (WGSL 배치에 맞춘 패딩)

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

    /// 멤버 한 줄과 그 멤버가 MSL에서 실제로 차지하는 바이트 수.
    private func memberDeclaration(
        _ member: WGSLLayout.MemberPlacement,
        in structure: WGSLStruct
    ) throws -> (text: String, byteSize: Int) {
        // 런타임 크기 배열은 MSL에 대응 문법이 없다. 마지막 멤버일 때만 길이 1 배열로 두고
        // 실제 길이는 호출 측이 보장한다 (docs/WGSL.md §4).
        if case .array(let element, nil) = member.type {
            let elementType = try MSLTypeMapping.type(element, module: module)
            return ("\(elementType) \(MSLTypeMapping.identifier(member.name))[1];", member.size)
        }
        if member.needsPackedVector, case .vector(3, .scalar(let scalar)) = member.type {
            // WGSL vec3는 크기 12 — MSL float3(16바이트)로는 뒤 멤버 자리를 못 맞춘다.
            return ("packed_\(MSLTypeMapping.scalar(scalar))3 \(MSLTypeMapping.identifier(member.name));", 12)
        }
        let type = try MSLTypeMapping.type(member.type, module: module)
        if case .vector(3, _) = member.type {
            return ("\(type) \(MSLTypeMapping.identifier(member.name));", 16)
        }
        return ("\(type) \(MSLTypeMapping.identifier(member.name));", member.size)
    }

    // MARK: - 리소스 스레딩

    /// 함수가 (전이적으로) 쓰는 모듈 스코프 변수를 결정적 순서로 돌려준다.
    private func threadedGlobals(for functionName: String) -> [WGSLGlobalVariable] {
        let used = usage[functionName] ?? []
        return module.globals
            .filter { used.contains($0.name) }
            .sorted {
                ($0.group ?? Int.max, $0.binding ?? Int.max, $0.name)
                    < ($1.group ?? Int.max, $1.binding ?? Int.max, $1.name)
            }
    }

    /// 전역 변수를 함수 인자로 넘길 때의 선언 (`constant Uniforms& u`).
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

        // storage 버퍼의 저장 타입이 런타임 배열이면 포인터로 받는다.
        if case .array(let element, nil) = global.type {
            return "\(qualifier) \(try MSLTypeMapping.type(element, module: module))* \(MSLTypeMapping.identifier(global.name))"
        }
        return "\(qualifier) \(try MSLTypeMapping.type(global.type, module: module))& \(MSLTypeMapping.identifier(global.name))"
    }

    /// 진입점에서 리소스를 받을 때 붙는 Metal 인덱스 속성.
    private func bindingAttribute(for global: WGSLGlobalVariable) throws -> String {
        guard let group = global.group, let binding = global.binding else { return "" }
        guard let index = bindings.index(group: group, binding: binding) else {
            throw WGPUError.validation(
                "셰이더가 쓰는 @group(\(group)) @binding(\(binding)) (\(global.name))이(가) "
                    + "파이프라인 레이아웃에 없다"
            )
        }
        guard let layout = WGSLReflectionBuilder.bindingLayout(for: global) else { return "" }
        switch layout.metalSlotKind {
        case .buffer: return " [[buffer(\(index))]]"
        case .texture: return " [[texture(\(index))]]"
        case .sampler: return " [[sampler(\(index))]]"
        }
    }

    // MARK: - 일반 함수

    private mutating func emitFunction(_ function: WGSLFunction) throws {
        let previousScope = textureScope
        defer {
            textureScope = previousScope
            injectedGlobalNames = []
            localRenames = [:]
        }
        registerTextures(of: function)
        injectedGlobalNames = Set(threadedGlobals(for: function.name).map(\.name))
        localRenames = [:]

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

    // MARK: - 진입점

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

        let innerName = "wgpu_\(function.name)_inner"
        let resources = threadedGlobals(for: function.name)

        // 1) 원래 시그니처를 유지한 내부 함수.
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
        try indented { emitter in
            try emitter.statements(function.body)
        }
        // 래퍼 방출은 주입 스코프 밖이다 — 리네임이 새면 wgpu_out 패킹이 엉뚱한 이름을 쓴다.
        injectedGlobalNames = []
        localRenames = [:]
        line("}")
        line("")

        // 2) 스테이지 I/O 인터페이스.
        var interface = EntryInterface()
        try buildInputs(of: function, stage: stage, into: &interface)
        try buildOutputs(of: function, stage: stage, into: &interface)

        // WGSL은 `var<workgroup>`의 0 초기화를 **보장한다.** MSL의 threadgroup 저장소는 그렇지
        // 않아 이전 디스패치의 잔여 값이 그대로 보인다 — 재현이 불규칙해 추적이 매우 어려운
        // 종류의 버그다. 스레드들이 나눠 0을 채우려면 threadgroup 안의 자기 번호가 필요하다.
        let workgroupGlobals = resources.filter { !$0.isResource && $0.addressSpace == "workgroup" }
        if stage == .compute, !workgroupGlobals.isEmpty {
            let declaration = "uint \(Self.localInvocationIndexName) [[thread_index_in_threadgroup]]"
            if !interface.builtinParameters.contains(declaration) {
                interface.builtinParameters.append(declaration)
            }
        }

        let inputStructName = "wgpu_\(function.name)_in"
        if !interface.stageInFields.isEmpty {
            line("struct \(inputStructName) {")
            indented { emitter in
                for field in interface.stageInFields { emitter.line(field) }
            }
            line("};")
            line("")
        }

        let outputStructName = "wgpu_\(function.name)_out"
        if !interface.outFields.isEmpty {
            line("struct \(outputStructName) {")
            indented { emitter in
                for field in interface.outFields { emitter.line(field) }
            }
            line("};")
            line("")
        }

        // 3) 래퍼.
        var wrapperParameters: [String] = []
        if !interface.stageInFields.isEmpty {
            wrapperParameters.append("\(inputStructName) wgpu_in [[stage_in]]")
        }
        wrapperParameters += interface.builtinParameters
        for global in resources where global.isResource {
            wrapperParameters.append("\(try parameterDeclaration(for: global))\(try bindingAttribute(for: global))")
        }
        if needsBufferSizes.contains(function.name) {
            // 런타임이 바인딩된 버퍼들의 바이트 크기를 이 인덱스에 꽂아 준다.
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
            // workgroup/private 전역은 바인딩이 아니라 진입점 안의 지역 저장소다.
            for global in resources where !global.isResource {
                try emitter.emitLocalGlobal(global)
            }
            if stage == .compute, !workgroupGlobals.isEmpty {
                try emitter.emitWorkgroupZeroInit(
                    workgroupGlobals, threadsPerGroup: function.workgroupSize ?? (1, 1, 1)
                )
            }
            for statement in interface.prelude { emitter.line(statement) }

            var arguments = interface.innerArguments + resources.map { MSLTypeMapping.identifier($0.name) }
            if emitter.needsBufferSizes.contains(function.name) { arguments.append(Self.bufferSizesName) }
            let call = "\(innerName)(\(arguments.joined(separator: ", ")))"
            // 실제 폐기는 **여기서** 한다 — 본문에서는 플래그만 세우고 쓰기를 가렸다.
            // 마지막에 부르는 것이 요점이다: 같은 쿼드의 이웃이 쓰는 미분값이 살아 있어야 한다.
            let discardTail = !emitter.discardFlagUsers.isEmpty && resources.contains {
                $0.name == Self.discardFlagName
            }
            if interface.outFields.isEmpty {
                emitter.line("\(call);")
                if discardTail { emitter.line("if (\(Self.discardFlagName)) { discard_fragment(); }") }
            } else {
                emitter.line("\(returnType) wgpu_result = \(call);")
                if discardTail { emitter.line("if (\(Self.discardFlagName)) { discard_fragment(); }") }
                emitter.line("\(outputStructName) wgpu_out{};")
                for statement in interface.packStatements { emitter.line(statement) }
                emitter.line("return wgpu_out;")
            }
        }
        line("}")
        line("")
    }

    /// `var<workgroup>` 전역을 0으로 깐다 — 명세가 보장하는 초기 상태를 만든다.
    ///
    /// threadgroup 저장소는 브레이스 초기화가 안 되므로 **스레드들이 나눠 채운다.** 그룹의
    /// 스레드 수만큼 건너뛰며 훑으면 배열이 아무리 커도 한 스레드가 몰아 쓰지 않는다.
    /// 마지막 배리어가 없으면 다른 스레드가 **아직 안 깔린 자리를 읽는다** — 초기화를 넣고도
    /// 같은 증상이 남는 흔한 실수 자리다.
    private mutating func emitWorkgroupZeroInit(
        _ globals: [WGSLGlobalVariable],
        threadsPerGroup: (Int, Int, Int)
    ) throws {
        let total = max(threadsPerGroup.0 * threadsPerGroup.1 * threadsPerGroup.2, 1)
        let index = Self.localInvocationIndexName
        line("// WGSL은 var<workgroup>의 0 초기화를 보장한다 (MSL의 threadgroup 저장소는 아니다).")
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
                // 배열이 아니면 한 스레드만 쓰면 된다 — 나머지는 배리어에서 만난다.
                let statement = try Self.zeroStatement(for: global.type, target: name, module: module)
                line("if (\(index) == 0u) { \(statement) }")
            }
        }
        line("threadgroup_barrier(mem_flags::mem_threadgroup);")
    }

    /// 한 자리를 0으로 만드는 문장. atomic은 대입이 아니라 store로만 쓸 수 있다.
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
                        "진입점 매개변수 '\(parameter.name)'의 타입 '\(typeName)'을(를) 찾을 수 없다 "
                            + "— 구조체 선언이 같은 셰이더 모듈 안에 있어야 한다"
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

    /// 입력 하나를 stage_in 필드 또는 builtin 파라미터로 등록하고, 내부 함수에 넘길 식을 돌려준다.
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
            throw WGPUError.validation("진입점 입력에는 @location 또는 @builtin이 필요하다")
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
            throw WGPUError.validation("진입점 출력에는 @location 또는 @builtin이 필요하다")
        }
        let name = "f\(location)"
        if stage == .fragment {
            return (name, "\(mslType) \(name) [[color(\(location))]];")
        }
        let interpolation = MSLTypeMapping.interpolation(attributes.first(named: "interpolate"))
        return (name, "\(mslType) \(name) [[user(locn\(location))]]\(interpolation);")
    }

    // MARK: - 문장

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
                line("if (!\(Self.discardFlagName)) { \(text); }")
            } else {
                line("\(text);")
            }

        case .expressionStatement(let expression):
            let text = try self.expression(expression)
            if masksAfterDiscard(statement) {
                line("if (!\(Self.discardFlagName)) { \(text); }")
            } else {
                line("\(text);")
            }

        case .ifStatement(let condition, let then, let elseBranch):
            line("if (\(try expression(condition)))")
            line("{")
            try scoped { emitter in try emitter.statements(then) }
            line("}")
            switch elseBranch {
            case .block(let statements)?:
                line("else")
                line("{")
                try scoped { emitter in try emitter.statements(statements) }
                line("}")
            case .chained(let nested)?:
                // `else if` — 중첩 if를 else 뒤에 붙인다.
                line("else")
                try scoped { emitter in try emitter.emitStatement(nested) }
            case nil:
                break
            }

        case .forStatement(let initializer, let condition, let update, let body):
            // 헤더 선언의 리네임은 조건·증감·본문까지가 스코프다 — for 문이 끝나면 되돌린다.
            let savedRenames = localRenames
            let initializerText = try initializer.map { try simpleStatement($0) } ?? ""
            let conditionText = try condition.map { try expression($0) } ?? ""
            let updateText = try update.map { try simpleStatement($0) } ?? ""
            line("for (\(initializerText); \(conditionText); \(updateText))")
            line("{")
            try scoped { emitter in try emitter.statements(body) }
            line("}")
            localRenames = savedRenames

        case .whileStatement(let condition, let body):
            let conditionText = try expression(condition)
            // 조건이 리터럴 `true`면 컴파일러가 보기에 끝나지 않는 루프다 — 가드를 붙인다.
            let guardName = conditionText == "true" ? declareLoopGuard() : nil
            line("while (\(conditionText))")
            line("{")
            try scoped { emitter in
                if let guardName { emitter.emitLoopGuardCheck(guardName) }
                try emitter.statements(body)
            }
            line("}")

        case .loopStatement(let body, let continuing):
            if continuing != nil, containsContinue(body) {
                throw WGPUError.unsupported(
                    "WGSL: `continuing` 블록과 `continue`를 함께 쓰는 loop는 지원하지 않는다 (docs/WGSL.md §4)"
                )
            }
            let guardName = declareLoopGuard()
            line("while (true)")
            line("{")
            try scoped { emitter in
                emitter.emitLoopGuardCheck(guardName)
                try emitter.statements(body)
                if let continuing { try emitter.statements(continuing) }
            }
            line("}")

        case .switchStatement(let subject, let cases):
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
                        // WGSL의 case는 fallthrough 하지 않는다.
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
            line("break;")
        case .continueStatement:
            line("continue;")
        case .discardStatement:
            // MSL의 `discard_fragment()`는 **즉시 종료가 아니다** — 뒤의 코드가 계속 돌아
            // 버려진 프래그먼트가 스토리지 버퍼·텍스처를 오염시킨다. 명세는 이를 금지하므로
            // 여기서는 표시만 하고, 쓰기는 이 플래그로 가리며, 실제 폐기는 진입점 끝에서 한다.
            // (즉시 `return`으로 끝내면 같은 쿼드의 이웃 스레드가 쓰는 미분값이 깨진다.)
            if discardFlagUsers.isEmpty {
                line("discard_fragment();")
            } else {
                line("\(Self.discardFlagName) = true;")
            }
        }
    }

    /// 이 문장이 **`discard` 뒤에는 실행되면 안 되는 쓰기**인가.
    ///
    /// 대상은 명세가 정한 그대로 — 스토리지 버퍼와 텍스처, 그리고 원자 연산이다.
    /// 지역 변수 쓰기는 이 프래그먼트 밖으로 나가지 않으므로 가리지 않는다.
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

    /// 대입 대상의 뿌리 식별자 (`out[i].x` → `out`).
    private static func rootIdentifier(of expression: WGSLExpression) -> String? {
        switch expression {
        case .identifier(let name): return name
        case .index(let base, _), .member(let base, _), .dereference(let base), .paren(let base):
            return rootIdentifier(of: base)
        default: return nil
        }
    }

    /// 끝나지 않을 수 있는 루프에 **탈출 조건을 하나 더** 붙인다.
    ///
    /// C++에서 무한 루프는 정의되지 않은 동작이다. 컴파일러는 "이 루프는 언젠가 끝난다"고
    /// 가정할 수 있고, 그 가정 위에서 주변 코드를 지우거나 순서를 바꾼다 — 실제로는 끝나지
    /// 않는 루프였을 때 어떤 코드가 나올지 아무 보장이 없다. Tint도 같은 이유로 넣는다.
    ///
    /// 상한은 u32가 셀 수 있는 끝이라 **정상 셰이더의 동작은 바뀌지 않는다** (거기 닿는 루프는
    /// 이미 GPU를 몇 분씩 멈춰 세우고 있다).
    private mutating func declareLoopGuard() -> String {
        loopGuardCounter += 1
        let name = "wgpu_loop_guard_\(loopGuardCounter)"
        line("uint \(name) = 0u;")
        return name
    }

    private mutating func emitLoopGuardCheck(_ name: String) {
        line("if (\(name) >= 4294967294u) { break; }")
        line("\(name) = \(name) + 1u;")
    }

    /// 세미콜론 없는 단문 (for 헤더에서도 쓴다).
    ///
    /// 선언은 **초기값을 먼저 방출**한다 — WGSL의 point-of-declaration 규칙상 초기값은
    /// 바깥(가려지기 전) 이름을 보므로, 리네임 등록이 초기값보다 먼저면 `var v = v;`가 깨진다.
    private mutating func simpleStatement(_ statement: WGSLStatement) throws -> String {
        switch statement {
        case .letDeclaration(let name, let type, let value), .constDeclaration(let name, let type, let value):
            let typeText = try type.map { try MSLTypeMapping.type($0, module: module) } ?? "auto"
            let valueText = try expression(value)
            return "const \(typeText) \(declaredName(name)) = \(valueText)"

        case .varDeclaration(let name, let type, let value):
            guard let type else {
                guard let value else {
                    throw WGPUError.validation("WGSL: var '\(name)'에 타입도 초기값도 없다")
                }
                let valueText = try expression(value)
                return "auto \(declaredName(name)) = \(valueText)"
            }
            let typeText = try MSLTypeMapping.type(type, module: module)
            guard let value else { return "\(typeText) \(declaredName(name)){}" }
            let valueText = try expression(value)
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
            throw WGPUError.validation("WGSL: 이 위치에 올 수 없는 문장")
        }
    }

    /// 지역 선언의 방출 이름.
    ///
    /// 주입된 전역(함수 매개변수가 된 이름)과 겹치면 대체 이름을 등록한다 — C++은 함수
    /// 최상위 블록에서 매개변수 재정의를 허용하지 않고, 호출 시 전역을 넘기는 인자 목록도
    /// 원래 이름을 참조해야 하기 때문이다. 이후 이 스코프의 참조는 대체 이름으로 나간다.
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
            case .ifStatement(_, let then, let elseBranch):
                if containsContinue(then) { return true }
                switch elseBranch {
                case .block(let inner)?: if containsContinue(inner) { return true }
                case .chained(let nested)?: if containsContinue([nested]) { return true }
                case nil: break
                }
            case .switchStatement(_, let cases):
                for switchCase in cases where containsContinue(switchCase.body) { return true }
            default:
                // 중첩된 loop의 continue는 그쪽 loop 소속이다.
                break
            }
        }
        return false
    }

    // MARK: - 표현식

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
            return "\(op)\(try self.expression(operand))"
        case .binary(let op, let left, let right):
            // WGSL이 정의하는데 C++에서는 정의되지 않은 연산들 — 프렐류드 헬퍼로 우회한다.
            // 어느 타입인지는 여기서 알 수 없으므로 판단을 C++ 오버로드에 넘긴다:
            // 정수면 가드가 붙고, 부동소수·추상 정수면 그대로 지나간다 (`MSLPrelude`).
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

    /// 인덱싱 — **범위를 벗어난 접근을 막는다** (WebGPU 명세의 robustness).
    ///
    /// 인덱스는 대개 유니폼·스토리지에서 오므로 결국 번들(JS)이 정하는 값이다. 그대로 두면
    /// 셰이더가 인접 GPU 메모리를 읽거나 덮어쓰고, 크게 벗어나면 페이지 폴트로 커맨드 버퍼가
    /// 죽는다.
    ///
    /// 상한이 **타입에 있는가**로 두 갈래다:
    /// - 고정 크기 배열·벡터·행렬 → `wgpu_at`이 C++ 템플릿 인자로 크기를 안다.
    /// - 런타임 크기 배열(`array<T>`) → 포인터로 내려와 타입에 크기가 없다. 이미 있는
    ///   **버퍼 크기 표**(`arrayLength()`가 쓰는 것)로 원소 수를 구해 넘긴다.
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

    /// 인덱스 자리로 **쓰는** 문장 (`a[i] = v`, `a[i] += v`).
    ///
    /// 읽기와 달리 별도 헬퍼가 필요한 이유는 **벡터 성분** 때문이다 — MSL에서 `v[i]`는 참조로
    /// 묶을 수 없어(`non-const reference cannot bind to vector element`) 클램프한 참조를
    /// 돌려주는 방식이 통하지 않는다. 대입 자체를 헬퍼 안에서 끝내면 배열·행렬·포인터까지
    /// 같은 이름으로 덮인다.
    ///
    /// 복합 대입(`+=`)은 읽기 식을 한 번 더 만든다 — 대상이 식별자·멤버라 부작용이 없다.
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

    /// 이 표현식이 **런타임 크기 배열**이면 그 원소 수를 구하는 MSL 식.
    ///
    /// `arrayLength(&x)`와 같은 계산이다 — 버퍼의 바이트 수를 원소 크기로 나눈다.
    /// 그래서 표를 쓰는 함수 목록(`needsBufferSizes`)도 이쪽 사용을 함께 센다.
    private func runtimeArrayCount(of expression: WGSLExpression) throws -> String? {
        switch expression {
        case .identifier(let name):
            guard let global = globalsByName[name], case .array(_, nil) = global.type else { return nil }
            return try? arrayLengthExpression([.addressOf(.identifier(name))])
        case .member(.identifier(let name), let field):
            guard let global = globalsByName[name], case .named(let structName) = global.type,
                  let structure = module.structNamed(structName),
                  let member = structure.members.first(where: { $0.name == field }),
                  case .array(_, nil) = member.type else { return nil }
            return try? arrayLengthExpression([.addressOf(.member(.identifier(name), field))])
        default:
            return nil
        }
    }

    /// 스위즐·인덱싱의 대상으로 쓸 표현식.
    ///
    /// AbstractInt 상수식 벡터는 보통 프록시로 내보내지만(문맥에서 타입이 굳도록), 프록시에는
    /// `.xyz` 같은 성분 접근이 없다. 그 자리에서는 f32 벡터로 확정해 내보낸다.
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

        if callee == "bitcast" {
            guard let target = typeArguments.first, arguments.count == 1 else {
                throw WGPUError.validation("WGSL: bitcast<T>(x) 형태여야 한다")
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

        // 구조체 생성자는 MSL(C++)에서 집합 초기화다.
        if structNames.contains(callee) {
            return "\(MSLTypeMapping.identifier(callee)){\(emitted.joined(separator: ", "))}"
        }
        // 성분 타입이 생략된 벡터 생성자.
        if MSLPrelude.inferredVectorConstructors.contains(callee), typeArguments.isEmpty {
            // 인자가 전부 정수 리터럴이면 추론할 근거가 없다. WGSL의 AbstractInt는 문맥 타입을
            // 따르는데, 벡터 생성자에서는 f32 문맥이 압도적으로 흔하므로 그쪽을 택한다
            // (`vec3(1)` = 흰색). 정수 벡터가 필요하면 `vec3u(…)`처럼 명시할 것 — docs/WGSL.md §4.
            if Self.isAbstractIntegerArguments(arguments) {
                // WGSL의 AbstractInt 상수식은 문맥 타입으로 굳는다. 프록시로 내보내
                // 그 결정을 C++ 변환 연산자에 넘긴다 (docs/WGSL.md §2-1).
                let size = Int(callee.dropFirst(3)) ?? 4
                return "wgpu_aint\(size)(\(emitted.joined(separator: ", ")))"
            }
            return "wgpu_\(callee)(\(emitted.joined(separator: ", ")))"
        }
        // 리터럴 승격이 필요한 내장 함수도 마찬가지 (`max(x, 0)`).
        if let helper = MSLPrelude.redirectedBuiltins[callee] {
            return "\(helper)(\(emitted.joined(separator: ", ")))"
        }
        if callee == "array" {
            // 성분 타입이 생략되면 첫 인자의 생성자 이름에서 짚어 낸다 (`array(vec2f(…), …)`).
            let element = typeArguments.first ?? arguments.first.flatMap(inferredType(of:))
            guard let element, let elementType = try? MSLTypeMapping.type(element, module: module) else {
                throw WGPUError.unsupported(
                    "WGSL: array(…) 생성자의 성분 타입을 알 수 없다 — `array<T, N>(…)`로 명시할 것"
                )
            }
            return "array<\(elementType), \(emitted.count)>{\(emitted.joined(separator: ", "))}"
        }
        // f32 → i32/u32 변환은 WGSL이 **포화**로 정의한다 (C++ 캐스트는 범위 밖이면 UB).
        // 인자가 하나일 때만 변환이고, 성분을 나열한 생성자는 각 인자가 이미 자기 변환을 지난다.
        if emitted.count == 1, let helper = Self.saturatingConversion(callee, typeArguments: typeArguments) {
            return "\(helper)(\(emitted[0]))"
        }
        if let constructor = try vectorOrMatrixConstructor(callee, typeArguments: typeArguments) {
            return "\(constructor)(\(emitted.joined(separator: ", ")))"
        }
        if MSLTypeMapping.scalarConstructors.contains(callee) {
            return "\(MSLTypeMapping.scalar(callee))(\(emitted.joined(separator: ", ")))"
        }

        // 사용자 함수는 스레딩된 리소스를 뒤에 덧붙여 넘긴다.
        if module.functionNamed(callee) != nil {
            var extra = threadedGlobals(for: callee).map { MSLTypeMapping.identifier($0.name) }
            if needsBufferSizes.contains(callee) { extra.append(Self.bufferSizesName) }
            return "\(MSLTypeMapping.functionName(callee))(\((emitted + extra).joined(separator: ", ")))"
        }

        let name = MSLTypeMapping.renamedBuiltins[callee] ?? callee
        return "\(name)(\(emitted.joined(separator: ", ")))"
    }

    /// 정수로 바꾸는 생성자면 포화 변환 헬퍼 이름 (`i32` `u32` `vec3i` `vec2<u32>` …).
    ///
    /// 판단을 문자열로 하는 것은 **타입 추론기가 없기 때문**이다. 인자가 실제로 f32인지는
    /// 여기서 알 수 없으므로 C++ 오버로드에 넘긴다 — f32가 아니면 헬퍼가 그냥 캐스트한다.
    private static func saturatingConversion(_ callee: String, typeArguments: [WGSLType]) -> String? {
        func target(_ scalar: String) -> String? {
            scalar == "i32" ? "wgpu_ftoi" : (scalar == "u32" ? "wgpu_ftou" : nil)
        }
        if MSLTypeMapping.scalarConstructors.contains(callee) { return target(callee) }
        // `vec3i(…)` / `vec3<i32>(…)` — 크기를 템플릿 인자로 넘겨 스칼라 브로드캐스트도 받는다.
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
            // `vec3(…)`처럼 성분 타입이 생략되면 f32로 본다 (실제 셰이더에서 가장 흔한 경우).
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

    // MARK: - 텍스처 / 아토믹 내장 함수

    private mutating func textureCall(callee: String, arguments: [WGSLExpression]) throws -> String {
        guard let first = arguments.first else {
            throw WGPUError.validation("WGSL: \(callee)()에 텍스처 인자가 없다")
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

        switch callee {
        case "textureSample", "textureSampleBias", "textureSampleLevel", "textureSampleGrad":
            guard rest.count >= 2 else {
                throw WGPUError.validation("WGSL: \(callee)(t, s, coords, …) 형태여야 한다")
            }
            let sampler = rest.removeFirst()
            let coordinates = rest.removeFirst()
            var parameters = [sampler, coordinates]
            if isArrayed, !rest.isEmpty { parameters.append(rest.removeFirst()) }
            switch callee {
            case "textureSampleLevel":
                guard !rest.isEmpty else { throw WGPUError.validation("WGSL: textureSampleLevel에 level이 없다") }
                parameters.append("level(\(rest.removeFirst()))")
            case "textureSampleBias":
                guard !rest.isEmpty else { throw WGPUError.validation("WGSL: textureSampleBias에 bias가 없다") }
                parameters.append("bias(\(rest.removeFirst()))")
            case "textureSampleGrad":
                guard rest.count >= 2 else { throw WGPUError.validation("WGSL: textureSampleGrad에 미분값이 없다") }
                let dx = rest.removeFirst()
                let dy = rest.removeFirst()
                parameters.append("gradient2d(\(dx), \(dy))")
            default:
                break
            }
            return "\(receiver).sample(\(parameters.joined(separator: ", ")))"

        case "textureSampleCompare", "textureSampleCompareLevel":
            guard rest.count >= 3 else {
                throw WGPUError.validation("WGSL: \(callee)(t, s, coords, depth_ref) 형태여야 한다")
            }
            let sampler = rest.removeFirst()
            let coordinates = rest.removeFirst()
            var parameters = [sampler, coordinates]
            if isArrayed, rest.count >= 2 { parameters.append(rest.removeFirst()) }
            parameters.append(rest.removeFirst())
            return "\(receiver).sample_compare(\(parameters.joined(separator: ", ")))"

        case "textureLoad":
            guard !rest.isEmpty else { throw WGPUError.validation("WGSL: textureLoad에 좌표가 없다") }
            var parameters = [integerCast(rest.removeFirst())]
            parameters += rest
            return "\(receiver).read(\(parameters.joined(separator: ", ")))"

        case "textureStore":
            guard rest.count >= 2 else {
                throw WGPUError.validation("WGSL: textureStore(t, coords, value) 형태여야 한다")
            }
            let coordinates = integerCast(rest.removeFirst())
            let value = rest.removeLast()
            var parameters = [value, coordinates]
            parameters += rest   // 배열 레이어 인덱스
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
                throw WGPUError.validation("WGSL: textureSampleBaseClampToEdge(t, s, coords) 형태여야 한다")
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
            throw WGPUError.unsupported("WGSL: 지원하지 않는 텍스처 내장 함수 \(callee)()")
        }
    }

    private func textureType(of expression: WGSLExpression) -> WGSLTextureType? {
        guard case .identifier(let name) = expression else { return nil }
        return textureScope[name]
    }

    private mutating func atomicCall(callee: String, arguments: [WGSLExpression]) throws -> String {
        let emitted = try arguments.map { try expression($0) }
        if let operation = MSLTypeMapping.atomicFetchOperations[callee] {
            guard emitted.count == 2 else {
                throw WGPUError.validation("WGSL: \(callee)(&atomic, value) 형태여야 한다")
            }
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
            throw WGPUError.unsupported("WGSL: 지원하지 않는 아토믹 내장 함수 \(callee)()")
        }
    }
}
