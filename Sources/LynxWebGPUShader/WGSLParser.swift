import Foundation
import LynxWebGPUCore

/// WGSL 재귀 하강 파서.
///
/// WGSL은 `<`가 제네릭 인자인지 비교 연산자인지 문맥으로 갈린다. 완전한 템플릿 판별
/// 알고리즘 대신, **템플릿을 받는 이름 집합**(`vec3`, `array`, `bitcast` …)을 알고 있는 것으로
/// 대신한다 — 실제 셰이더에서 사용자 타입이 제네릭을 받는 경우는 없으므로 충분하다.
struct WGSLParser {
    private var tokens: [WGSLToken]
    private var index = 0
    /// 제네릭 인자 안에서 표현식을 파싱 중인 깊이 (`array<f32, 64>`의 길이 식).
    /// 0보다 크면 `>`를 비교 연산자가 아니라 **템플릿 닫기**로 본다.
    private var templateDepth = 0

    /// `<`를 제네릭 시작으로 해석해야 하는 이름들.
    private static let templateNames: Set<String> = [
        "vec2", "vec3", "vec4",
        "mat2x2", "mat2x3", "mat2x4",
        "mat3x2", "mat3x3", "mat3x4",
        "mat4x2", "mat4x3", "mat4x4",
        "array", "atomic", "ptr", "bitcast",
        "texture_1d", "texture_2d", "texture_2d_array", "texture_3d",
        "texture_cube", "texture_cube_array", "texture_multisampled_2d",
        "texture_storage_1d", "texture_storage_2d", "texture_storage_2d_array", "texture_storage_3d",
    ]

    /// 제네릭 인자 안에서는 이 토큰들이 이항 연산자가 아니라 템플릿 닫기다.
    private static let templateClosers: Set<String> = ["<", ">", "<=", ">=", ">>"]

    private static let statementKeywords: Set<String> = [
        "let", "var", "const", "if", "for", "while", "loop", "switch",
        "return", "break", "continue", "discard", "const_assert",
    ]

    static func parse(_ source: String) throws -> WGSLModule {
        var parser = WGSLParser(tokens: try WGSLLexer.tokenize(source))
        return try parser.parseModule()
    }

    private init(tokens: [WGSLToken]) {
        self.tokens = tokens
    }

    // MARK: - 토큰 조작

    private var current: WGSLToken { tokens[index] }
    private var isAtEnd: Bool { current.kind == .endOfFile }

    private func peek(_ offset: Int = 0) -> WGSLToken {
        let target = index + offset
        return target < tokens.count ? tokens[target] : tokens[tokens.count - 1]
    }

    private func check(_ text: String) -> Bool { current.text == text && current.kind != .endOfFile }

    @discardableResult
    private mutating func advance() -> WGSLToken {
        let token = current
        if !isAtEnd { index += 1 }
        return token
    }

    private mutating func match(_ text: String) -> Bool {
        guard check(text) else { return false }
        advance()
        return true
    }

    @discardableResult
    private mutating func expect(_ text: String, _ context: String) throws -> WGSLToken {
        guard check(text) else {
            throw error("'\(text)' 이(가) 필요하다 (\(context)) — 실제: '\(current.text)'")
        }
        return advance()
    }

    private mutating func expectIdentifier(_ context: String) throws -> String {
        guard current.kind == .identifier else {
            throw error("식별자가 필요하다 (\(context)) — 실제: '\(current.text)'")
        }
        return advance().text
    }

    /// 제네릭 닫기. `array<vec4<f32>,3>` 처럼 중첩되면 렉서가 `>>`로 붙여 놓으므로 하나만 떼어낸다.
    private mutating func expectGenericClose(_ context: String) throws {
        if check(">") {
            advance()
        } else if check(">>") {
            tokens[index] = .punctuation(">", line: current.line)
        } else if check(">=") {
            tokens[index] = .punctuation("=", line: current.line)
        } else {
            throw error("'>' 이(가) 필요하다 (\(context)) — 실제: '\(current.text)'")
        }
    }

    private func error(_ message: String) -> WGPUError {
        .validation("WGSL 파싱 실패 (line \(current.line)): \(message)")
    }

    // MARK: - 모듈

    private mutating func parseModule() throws -> WGSLModule {
        var module = WGSLModule()
        while !isAtEnd {
            if match(";") { continue }
            let attributes = try parseAttributes()

            if check("struct") {
                advance()
                module.structs.append(try parseStruct())
            } else if check("fn") {
                advance()
                module.functions.append(try parseFunction(attributes: attributes))
            } else if check("var") {
                advance()
                module.globals.append(try parseGlobalVariable(attributes: attributes))
            } else if check("const") || check("override") {
                let isOverride = current.text == "override"
                advance()
                module.constants.append(try parseModuleConstant(isOverride: isOverride))
            } else if check("alias") || check("type") {
                advance()
                let name = try expectIdentifier("alias 이름")
                try expect("=", "alias")
                let type = try parseType()
                try expect(";", "alias")
                module.aliases.append((name, type))
            } else if check("const_assert") {
                // 컴파일 타임 단언은 번역 대상이 아니다 — 세미콜론까지 건너뛴다.
                while !isAtEnd, !check(";") { advance() }
                try expect(";", "const_assert")
            } else {
                throw error("모듈 스코프에 올 수 없는 토큰 '\(current.text)'")
            }
        }
        return module
    }

    private mutating func parseAttributes() throws -> [WGSLAttribute] {
        var attributes: [WGSLAttribute] = []
        while check("@") {
            advance()
            let name = try expectIdentifier("속성 이름")
            var arguments: [String] = []
            if match("(") {
                repeat {
                    if check(")") { break }
                    var text = ""
                    var depth = 0
                    while !isAtEnd {
                        if depth == 0, check(",") || check(")") { break }
                        if check("(") { depth += 1 }
                        if check(")") { depth -= 1 }
                        text += advance().text
                    }
                    arguments.append(text)
                } while match(",")
                try expect(")", "속성 인자")
            }
            attributes.append(WGSLAttribute(name: name, arguments: arguments))
        }
        return attributes
    }

    private mutating func parseStruct() throws -> WGSLStruct {
        let name = try expectIdentifier("구조체 이름")
        try expect("{", "구조체 본문")
        var members: [WGSLStructMember] = []
        while !check("}"), !isAtEnd {
            let attributes = try parseAttributes()
            let memberName = try expectIdentifier("멤버 이름")
            try expect(":", "멤버 타입")
            let type = try parseType()
            members.append(WGSLStructMember(attributes: attributes, name: memberName, type: type))
            if !match(",") { break }
        }
        try expect("}", "구조체 끝")
        _ = match(";")
        return WGSLStruct(name: name, members: members)
    }

    private mutating func parseGlobalVariable(attributes: [WGSLAttribute]) throws -> WGSLGlobalVariable {
        var addressSpace: String?
        var access: String?
        if match("<") {
            addressSpace = try expectIdentifier("주소 공간")
            if match(",") { access = try expectIdentifier("접근 모드") }
            try expectGenericClose("var 주소 공간")
        }
        let name = try expectIdentifier("변수 이름")
        var type: WGSLType = .void
        if match(":") { type = try parseType() }
        var initializer: WGSLExpression?
        if match("=") { initializer = try parseExpression() }
        try expect(";", "전역 변수")
        return WGSLGlobalVariable(
            attributes: attributes,
            name: name,
            addressSpace: addressSpace,
            access: access,
            type: type,
            initializer: initializer
        )
    }

    private mutating func parseModuleConstant(isOverride: Bool) throws -> WGSLModuleConstant {
        let name = try expectIdentifier("상수 이름")
        var type: WGSLType?
        if match(":") { type = try parseType() }
        try expect("=", "상수 초기값")
        let value = try parseExpression()
        try expect(";", "상수")
        return WGSLModuleConstant(name: name, type: type, value: value, isOverride: isOverride)
    }

    private mutating func parseFunction(attributes: [WGSLAttribute]) throws -> WGSLFunction {
        let name = try expectIdentifier("함수 이름")
        try expect("(", "매개변수 목록")
        var parameters: [WGSLParameter] = []
        while !check(")"), !isAtEnd {
            let parameterAttributes = try parseAttributes()
            let parameterName = try expectIdentifier("매개변수 이름")
            try expect(":", "매개변수 타입")
            let type = try parseType()
            parameters.append(WGSLParameter(attributes: parameterAttributes, name: parameterName, type: type))
            if !match(",") { break }
        }
        try expect(")", "매개변수 목록 끝")

        var returnAttributes: [WGSLAttribute] = []
        var returnType: WGSLType?
        if match("->") {
            returnAttributes = try parseAttributes()
            returnType = try parseType()
        }
        let body = try parseBlock()
        return WGSLFunction(
            attributes: attributes,
            name: name,
            parameters: parameters,
            returnAttributes: returnAttributes,
            returnType: returnType,
            body: body
        )
    }

    // MARK: - 타입

    private mutating func parseType() throws -> WGSLType {
        let name = try expectIdentifier("타입 이름")

        switch name {
        case "f32", "i32", "u32", "bool", "f16":
            return .scalar(name)
        case "void":
            return .void
        case "vec2", "vec3", "vec4":
            let size = Int(name.dropFirst(3))!
            try expect("<", "벡터 성분 타입")
            let element = try parseType()
            try expectGenericClose("벡터")
            return .vector(size: size, element: element)
        case "array":
            try expect("<", "배열 원소 타입")
            let element = try parseType()
            var count: WGSLExpression?
            if match(",") {
                templateDepth += 1
                defer { templateDepth -= 1 }
                count = try parseExpression()
            }
            try expectGenericClose("배열")
            return .array(element: element, count: count)
        case "atomic":
            try expect("<", "atomic 성분 타입")
            let element = try parseType()
            try expectGenericClose("atomic")
            return .atomic(element)
        case "ptr":
            try expect("<", "포인터 주소 공간")
            let space = try expectIdentifier("주소 공간")
            try expect(",", "포인터 대상 타입")
            let element = try parseType()
            var access: String?
            if match(",") { access = try expectIdentifier("접근 모드") }
            try expectGenericClose("포인터")
            return .pointer(space: space, element: element, access: access)
        case "sampler":
            return .sampler(comparison: false)
        case "sampler_comparison":
            return .sampler(comparison: true)
        default:
            break
        }

        if let shorthand = Self.shorthandType(name) { return shorthand }
        if name.hasPrefix("mat") { return try parseMatrixType(name) }
        if name.hasPrefix("texture_") { return try parseTextureType(name) }
        return .named(name)
    }

    /// `vec4f` / `vec2u` / `mat4x4f` 같은 축약형 (WGSL 1.0 predeclared alias).
    /// 방출기도 생성자 이름(`vec3f(…)`)을 풀 때 쓴다.
    static func shorthandType(_ name: String) -> WGSLType? {
        let suffixes: [Character: String] = ["f": "f32", "i": "i32", "u": "u32", "h": "f16"]
        guard let last = name.last, let scalar = suffixes[last] else { return nil }
        let base = String(name.dropLast())
        if base == "vec2" || base == "vec3" || base == "vec4" {
            return .vector(size: Int(base.dropFirst(3))!, element: .scalar(scalar))
        }
        if base.hasPrefix("mat"), base.count == 6 {
            let dimensions = base.dropFirst(3)
            let parts = dimensions.split(separator: "x")
            guard parts.count == 2, let columns = Int(parts[0]), let rows = Int(parts[1]) else { return nil }
            return .matrix(columns: columns, rows: rows, element: .scalar(scalar))
        }
        return nil
    }

    private mutating func parseMatrixType(_ name: String) throws -> WGSLType {
        let dimensions = name.dropFirst(3).split(separator: "x")
        guard dimensions.count == 2, let columns = Int(dimensions[0]), let rows = Int(dimensions[1]) else {
            return .named(name)
        }
        try expect("<", "행렬 성분 타입")
        let element = try parseType()
        try expectGenericClose("행렬")
        return .matrix(columns: columns, rows: rows, element: element)
    }

    private mutating func parseTextureType(_ name: String) throws -> WGSLType {
        let body = String(name.dropFirst("texture_".count))

        if body == "external" {
            return .texture(WGSLTextureType(kind: .external, dimension: "2d", sampleType: nil, format: nil, access: nil))
        }
        if body.hasPrefix("depth_") {
            let dimension = String(body.dropFirst("depth_".count))
            let isMultisampled = dimension.hasPrefix("multisampled_")
            return .texture(WGSLTextureType(
                kind: isMultisampled ? .depthMultisampled : .depth,
                dimension: isMultisampled ? String(dimension.dropFirst("multisampled_".count)) : dimension,
                sampleType: .scalar("f32"), format: nil, access: nil
            ))
        }
        if body.hasPrefix("storage_") {
            let dimension = String(body.dropFirst("storage_".count))
            try expect("<", "storage 텍스처 포맷")
            let format = try expectIdentifier("포맷")
            try expect(",", "storage 텍스처 접근 모드")
            let access = try expectIdentifier("접근 모드")
            try expectGenericClose("storage 텍스처")
            return .texture(WGSLTextureType(
                kind: .storage, dimension: dimension, sampleType: nil, format: format, access: access
            ))
        }
        if body.hasPrefix("multisampled_") {
            let dimension = String(body.dropFirst("multisampled_".count))
            try expect("<", "멀티샘플 텍스처 성분 타입")
            let sampleType = try parseType()
            try expectGenericClose("멀티샘플 텍스처")
            return .texture(WGSLTextureType(
                kind: .multisampled, dimension: dimension, sampleType: sampleType, format: nil, access: nil
            ))
        }
        try expect("<", "텍스처 성분 타입")
        let sampleType = try parseType()
        try expectGenericClose("텍스처")
        return .texture(WGSLTextureType(kind: .sampled, dimension: body, sampleType: sampleType, format: nil, access: nil))
    }

    // MARK: - 문장

    private mutating func parseBlock() throws -> [WGSLStatement] {
        try expect("{", "블록 시작")
        var statements: [WGSLStatement] = []
        while !check("}"), !isAtEnd {
            statements.append(try parseStatement())
        }
        try expect("}", "블록 끝")
        return statements
    }

    private mutating func parseStatement() throws -> WGSLStatement {
        // 문장에 붙는 속성(@diagnostic 등)은 번역에 영향이 없다.
        _ = try parseAttributes()

        if check("{") { return .block(try parseBlock()) }
        if match(";") { return .block([]) }

        switch current.text {
        case "let", "const":
            let isConst = current.text == "const"
            advance()
            let name = try expectIdentifier("바인딩 이름")
            var type: WGSLType?
            if match(":") { type = try parseType() }
            try expect("=", "바인딩 초기값")
            let value = try parseExpression()
            try expect(";", "바인딩")
            return isConst
                ? .constDeclaration(name: name, type: type, value: value)
                : .letDeclaration(name: name, type: type, value: value)

        case "var":
            advance()
            if match("<") {
                // 함수 스코프의 `var<function>` — MSL에는 대응 개념이 없어 무시한다.
                while !isAtEnd, !check(">"), !check(">>") { advance() }
                try expectGenericClose("var 주소 공간")
            }
            let name = try expectIdentifier("변수 이름")
            var type: WGSLType?
            if match(":") { type = try parseType() }
            var value: WGSLExpression?
            if match("=") { value = try parseExpression() }
            try expect(";", "변수 선언")
            return .varDeclaration(name: name, type: type, value: value)

        case "if":
            advance()
            return try parseIfStatement()

        case "for":
            advance()
            return try parseForStatement()

        case "while":
            advance()
            let condition = try parseExpression()
            return .whileStatement(condition: condition, body: try parseBlock())

        case "loop":
            advance()
            return try parseLoopStatement()

        case "switch":
            advance()
            return try parseSwitchStatement()

        case "return":
            advance()
            var value: WGSLExpression?
            if !check(";") { value = try parseExpression() }
            try expect(";", "return")
            return .returnStatement(value)

        case "break":
            advance()
            // `break if cond;` (continuing 블록 전용) — 조건을 무시하면 의미가 바뀌므로 거부한다.
            if check("if") {
                throw error("`break if`는 지원하지 않는다 (docs/WGSL.md §4)")
            }
            try expect(";", "break")
            return .breakStatement

        case "continue":
            advance()
            try expect(";", "continue")
            return .continueStatement

        case "discard":
            advance()
            try expect(";", "discard")
            return .discardStatement

        case "const_assert":
            while !isAtEnd, !check(";") { advance() }
            try expect(";", "const_assert")
            return .block([])

        default:
            break
        }

        return try parseAssignmentOrCall()
    }

    private mutating func parseIfStatement() throws -> WGSLStatement {
        let condition = try parseExpression()
        let thenBranch = try parseBlock()
        var elseBranch: WGSLElseBranch?
        if match("else") {
            if check("if") {
                advance()
                elseBranch = .chained(try parseIfStatement())
            } else {
                elseBranch = .block(try parseBlock())
            }
        }
        return .ifStatement(condition: condition, then: thenBranch, elseBranch: elseBranch)
    }

    private mutating func parseForStatement() throws -> WGSLStatement {
        try expect("(", "for 헤더")
        var initializer: WGSLStatement?
        if !check(";") {
            initializer = try parseStatement()   // 세미콜론까지 소비한다
        } else {
            advance()
        }
        var condition: WGSLExpression?
        if !check(";") { condition = try parseExpression() }
        try expect(";", "for 조건")
        var update: WGSLStatement?
        if !check(")") { update = try parseSimpleAssignment() }
        try expect(")", "for 헤더 끝")
        return .forStatement(initializer: initializer, condition: condition, update: update, body: try parseBlock())
    }

    private mutating func parseLoopStatement() throws -> WGSLStatement {
        try expect("{", "loop 본문")
        var body: [WGSLStatement] = []
        var continuing: [WGSLStatement]?
        while !check("}"), !isAtEnd {
            if check("continuing") {
                advance()
                continuing = try parseBlock()
                continue
            }
            body.append(try parseStatement())
        }
        try expect("}", "loop 끝")
        return .loopStatement(body: body, continuing: continuing)
    }

    private mutating func parseSwitchStatement() throws -> WGSLStatement {
        let subject = try parseExpression()
        try expect("{", "switch 본문")
        var cases: [WGSLSwitchCase] = []
        while !check("}"), !isAtEnd {
            if match("case") {
                var selectors: [WGSLExpression] = []
                var isDefault = false
                repeat {
                    if check("default") {
                        advance()
                        isDefault = true
                    } else if !check(":"), !check("{") {
                        selectors.append(try parseExpression())
                    }
                } while match(",")
                _ = match(":")
                cases.append(WGSLSwitchCase(selectors: selectors, isDefault: isDefault, body: try parseBlock()))
            } else if match("default") {
                _ = match(":")
                cases.append(WGSLSwitchCase(selectors: [], isDefault: true, body: try parseBlock()))
            } else {
                throw error("switch 안에는 case/default만 올 수 있다 — 실제: '\(current.text)'")
            }
        }
        try expect("}", "switch 끝")
        return .switchStatement(subject: subject, cases: cases)
    }

    /// 대입 또는 호출 문장. 세미콜론까지 소비한다.
    private mutating func parseAssignmentOrCall() throws -> WGSLStatement {
        let statement = try parseSimpleAssignment()
        try expect(";", "문장")
        return statement
    }

    /// 세미콜론을 소비하지 않는 대입/증감/호출 (for의 update 절에서도 쓴다).
    private mutating func parseSimpleAssignment() throws -> WGSLStatement {
        if check("_"), peek(1).text == "=" {
            advance()
            advance()
            return .expressionStatement(try parseExpression())
        }

        let target = try parseExpression()
        let assignmentOperators = ["=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="]
        if assignmentOperators.contains(current.text) {
            let op = advance().text
            return .assignment(target: target, op: op, value: try parseExpression())
        }
        if match("++") { return .increment(target) }
        if match("--") { return .decrement(target) }
        return .expressionStatement(target)
    }

    // MARK: - 표현식 (우선순위 등반)

    private static let precedence: [[String]] = [
        ["||"],
        ["&&"],
        ["|"],
        ["^"],
        ["&"],
        ["==", "!="],
        ["<", ">", "<=", ">="],
        ["<<", ">>"],
        ["+", "-"],
        ["*", "/", "%"],
    ]

    mutating func parseExpression() throws -> WGSLExpression {
        try parseBinary(level: 0)
    }

    private mutating func parseBinary(level: Int) throws -> WGSLExpression {
        guard level < Self.precedence.count else { return try parseUnary() }
        var left = try parseBinary(level: level + 1)
        while Self.precedence[level].contains(current.text), current.kind == .punctuation,
              !(templateDepth > 0 && Self.templateClosers.contains(current.text)) {
            let op = advance().text
            let right = try parseBinary(level: level + 1)
            left = .binary(op: op, left, right)
        }
        return left
    }

    private mutating func parseUnary() throws -> WGSLExpression {
        if check("-") || check("!") || check("~") {
            let op = advance().text
            return .unary(op: op, try parseUnary())
        }
        if check("&") {
            advance()
            return .addressOf(try parseUnary())
        }
        if check("*") {
            advance()
            return .dereference(try parseUnary())
        }
        return try parsePostfix()
    }

    private mutating func parsePostfix() throws -> WGSLExpression {
        var expression = try parsePrimary()
        while true {
            if match(".") {
                expression = .member(expression, try expectIdentifier("멤버 이름"))
            } else if match("[") {
                let subscriptExpression = try parseExpression()
                try expect("]", "인덱스")
                expression = .index(expression, subscriptExpression)
            } else {
                return expression
            }
        }
    }

    private mutating func parsePrimary() throws -> WGSLExpression {
        if match("(") {
            let inner = try parseExpression()
            try expect(")", "괄호 표현식")
            return .paren(inner)
        }
        if current.kind == .intLiteral { return .intLiteral(advance().text) }
        if current.kind == .floatLiteral { return .floatLiteral(advance().text) }

        guard current.kind == .identifier else {
            throw error("표현식이 필요하다 — 실제: '\(current.text)'")
        }
        let name = advance().text
        if name == "true" { return .boolLiteral(true) }
        if name == "false" { return .boolLiteral(false) }

        var typeArguments: [WGSLType] = []
        if check("<"), Self.templateNames.contains(name) {
            if name == "array" {
                // `array<f32, 3>(…)`의 두 번째 인자는 타입이 아니라 길이 식이다.
                // 타입 파서가 이미 그 형태를 알고 있으므로 이름 토큰까지 되감아 재사용한다.
                index -= 1
                if case .array(let element, _) = try parseType() {
                    typeArguments = [element]
                }
            } else {
                advance()
                repeat {
                    typeArguments.append(try parseType())
                } while match(",")
                try expectGenericClose("제네릭 인자")
            }
        }

        if match("(") {
            var arguments: [WGSLExpression] = []
            while !check(")"), !isAtEnd {
                arguments.append(try parseExpression())
                if !match(",") { break }
            }
            try expect(")", "호출 인자")
            return .call(callee: name, typeArguments: typeArguments, arguments: arguments)
        }

        // 인자 없는 제네릭 이름(`array<f32>` 같은 타입 위치)은 표현식으로 올 수 없다.
        guard typeArguments.isEmpty else {
            throw error("'\(name)' 뒤에 호출 인자가 필요하다")
        }
        return .identifier(name)
    }
}
