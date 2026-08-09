import Foundation
import LynxWebGPUCore

/// Recursive-descent WGSL parser.
///
/// In WGSL, whether `<` opens a generic argument or is a comparison operator depends on context.
/// Instead of the full template-disambiguation algorithm, we stand in **a set of names that take
/// templates** (`vec3`, `array`, `bitcast`, …) — enough in practice, since user types never take generics.
struct WGSLParser {
    private var tokens: [WGSLToken]
    private var index = 0
    /// Depth of expression parsing inside a generic argument (the length expression of `array<f32, 64>`).
    /// Above zero, `>` is read as **closing a template** rather than as a comparison.
    private var templateDepth = 0

    /// Names whose `<` must be read as opening a generic.
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

    /// Inside a generic argument these tokens close the template rather than acting as binary operators.
    private static let templateClosers: Set<String> = ["<", ">", "<=", ">=", ">>"]

    /// Module-scope directives with nothing to carry into MSL.
    private static let moduleDirectives: Set<String> = ["enable", "requires", "diagnostic"]

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

    // MARK: - Token handling

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
            throw error("expected '\(text)' (\(context)) — got: '\(current.text)'")
        }
        return advance()
    }

    private mutating func expectIdentifier(_ context: String) throws -> String {
        guard current.kind == .identifier else {
            throw error("expected an identifier (\(context)) — got: '\(current.text)'")
        }
        return advance().text
    }

    /// Closes a generic. When nested, as in `array<vec4<f32>,3>`, the lexer glues them into `>>`, so we peel off one.
    private mutating func expectGenericClose(_ context: String) throws {
        if check(">") {
            advance()
        } else if check(">>") {
            tokens[index] = .punctuation(">", line: current.line)
        } else if check(">=") {
            tokens[index] = .punctuation("=", line: current.line)
        } else {
            throw error("expected '>' (\(context)) — got: '\(current.text)'")
        }
    }

    private func error(_ message: String) -> WGPUError {
        // Carry the line number **as a number too** — this is the value `getCompilationInfo()` hands the editor.
        WGPUError(
            kind: .validation,
            message: "WGSL parse failed (line \(current.line)): \(message)",
            line: current.line
        )
    }

    // MARK: - Module

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
                let name = try expectIdentifier("alias name")
                try expect("=", "alias")
                let type = try parseType()
                try expect(";", "alias")
                module.aliases.append((name, type))
            } else if check("const_assert") {
                // A compile-time assertion is not translated — skip to the semicolon.
                while !isAtEnd, !check(";") { advance() }
                try expect(";", "const_assert")
            } else if Self.moduleDirectives.contains(current.text) {
                // `enable f16;` `requires readonly_and_readwrite_storage_textures;` `diagnostic(off, …);`
                // An enable declaration has nothing to carry into MSL — skip to the semicolon.
                while !isAtEnd, !check(";") { advance() }
                try expect(";", "an enable declaration")
            } else {
                throw error("token '\(current.text)' cannot appear at module scope")
            }
        }
        return module
    }

    private mutating func parseAttributes() throws -> [WGSLAttribute] {
        var attributes: [WGSLAttribute] = []
        while check("@") {
            advance()
            let name = try expectIdentifier("attribute name")
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
                try expect(")", "attribute arguments")
            }
            attributes.append(WGSLAttribute(name: name, arguments: arguments))
        }
        return attributes
    }

    private mutating func parseStruct() throws -> WGSLStruct {
        let name = try expectIdentifier("struct name")
        try expect("{", "struct body")
        var members: [WGSLStructMember] = []
        while !check("}"), !isAtEnd {
            let attributes = try parseAttributes()
            let memberName = try expectIdentifier("member name")
            try expect(":", "member type")
            let type = try parseType()
            members.append(WGSLStructMember(attributes: attributes, name: memberName, type: type))
            if !match(",") { break }
        }
        try expect("}", "end of struct")
        _ = match(";")
        return WGSLStruct(name: name, members: members)
    }

    private mutating func parseGlobalVariable(attributes: [WGSLAttribute]) throws -> WGSLGlobalVariable {
        var addressSpace: String?
        var access: String?
        if match("<") {
            addressSpace = try expectIdentifier("address space")
            if match(",") { access = try expectIdentifier("access mode") }
            try expectGenericClose("var address space")
        }
        let name = try expectIdentifier("variable name")
        var type: WGSLType = .void
        if match(":") { type = try parseType() }
        var initializer: WGSLExpression?
        if match("=") { initializer = try parseExpression() }
        try expect(";", "global variable")
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
        let name = try expectIdentifier("constant name")
        var type: WGSLType?
        if match(":") { type = try parseType() }
        var value: WGSLExpression?
        if match("=") {
            value = try parseExpression()
        } else if !isOverride {
            throw error("expected '=' (constant initializer) — got: '\(current.text)'")
        }
        try expect(";", "constant")
        return WGSLModuleConstant(name: name, type: type, value: value, isOverride: isOverride)
    }

    private mutating func parseFunction(attributes: [WGSLAttribute]) throws -> WGSLFunction {
        let name = try expectIdentifier("function name")
        try expect("(", "parameter list")
        var parameters: [WGSLParameter] = []
        while !check(")"), !isAtEnd {
            let parameterAttributes = try parseAttributes()
            let parameterName = try expectIdentifier("parameter name")
            try expect(":", "parameter type")
            let type = try parseType()
            parameters.append(WGSLParameter(attributes: parameterAttributes, name: parameterName, type: type))
            if !match(",") { break }
        }
        try expect(")", "end of parameter list")

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

    // MARK: - Types

    private mutating func parseType() throws -> WGSLType {
        let name = try expectIdentifier("type name")

        switch name {
        case "f32", "i32", "u32", "bool", "f16":
            return .scalar(name)
        case "void":
            return .void
        case "vec2", "vec3", "vec4":
            let size = Int(name.dropFirst(3))!
            try expect("<", "vector component type")
            let element = try parseType()
            try expectGenericClose("vector")
            return .vector(size: size, element: element)
        case "array":
            try expect("<", "array element type")
            let element = try parseType()
            var count: WGSLExpression?
            if match(",") {
                templateDepth += 1
                defer { templateDepth -= 1 }
                count = try parseExpression()
            }
            try expectGenericClose("array")
            return .array(element: element, count: count)
        case "atomic":
            try expect("<", "atomic component type")
            let element = try parseType()
            try expectGenericClose("atomic")
            return .atomic(element)
        case "ptr":
            try expect("<", "pointer address space")
            let space = try expectIdentifier("address space")
            try expect(",", "pointer target type")
            let element = try parseType()
            var access: String?
            if match(",") { access = try expectIdentifier("access mode") }
            try expectGenericClose("pointer")
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

    /// Shorthands such as `vec4f` / `vec2u` / `mat4x4f` (WGSL 1.0 predeclared aliases).
    /// The emitter uses this too when resolving constructor names (`vec3f(…)`).
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
        try expect("<", "matrix component type")
        let element = try parseType()
        try expectGenericClose("matrix")
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
            try expect("<", "storage texture format")
            let format = try expectIdentifier("format")
            try expect(",", "storage texture access mode")
            let access = try expectIdentifier("access mode")
            try expectGenericClose("storage texture")
            return .texture(WGSLTextureType(
                kind: .storage, dimension: dimension, sampleType: nil, format: format, access: access
            ))
        }
        if body.hasPrefix("multisampled_") {
            let dimension = String(body.dropFirst("multisampled_".count))
            try expect("<", "multisampled texture component type")
            let sampleType = try parseType()
            try expectGenericClose("multisampled texture")
            return .texture(WGSLTextureType(
                kind: .multisampled, dimension: dimension, sampleType: sampleType, format: nil, access: nil
            ))
        }
        try expect("<", "texture component type")
        let sampleType = try parseType()
        try expectGenericClose("texture")
        return .texture(WGSLTextureType(kind: .sampled, dimension: body, sampleType: sampleType, format: nil, access: nil))
    }

    // MARK: - Statements

    private mutating func parseBlock() throws -> [WGSLStatement] {
        try expect("{", "start of block")
        var statements: [WGSLStatement] = []
        while !check("}"), !isAtEnd {
            statements.append(try parseStatement())
        }
        try expect("}", "end of block")
        return statements
    }

    private mutating func parseStatement() throws -> WGSLStatement {
        // Attributes on a statement (@diagnostic and the like) do not affect translation.
        _ = try parseAttributes()

        if check("{") { return .block(try parseBlock()) }
        if match(";") { return .block([]) }

        switch current.text {
        case "let", "const":
            let isConst = current.text == "const"
            advance()
            let name = try expectIdentifier("binding name")
            var type: WGSLType?
            if match(":") { type = try parseType() }
            try expect("=", "binding initializer")
            let value = try parseExpression()
            try expect(";", "binding")
            return isConst
                ? .constDeclaration(name: name, type: type, value: value)
                : .letDeclaration(name: name, type: type, value: value)

        case "var":
            advance()
            if match("<") {
                // Function-scope `var<function>` — MSL has no counterpart, so it is ignored.
                while !isAtEnd, !check(">"), !check(">>") { advance() }
                try expectGenericClose("var address space")
            }
            let name = try expectIdentifier("variable name")
            var type: WGSLType?
            if match(":") { type = try parseType() }
            var value: WGSLExpression?
            if match("=") { value = try parseExpression() }
            try expect(";", "variable declaration")
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
            // `break if cond;` (continuing blocks only) — ignoring the condition changes meaning, so it is rejected.
            if check("if") {
                throw error("`break if` is not supported (docs/WGSL.md §4)")
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
        try expect("(", "for header")
        var initializer: WGSLStatement?
        if !check(";") {
            initializer = try parseStatement()   // consumes through the semicolon
        } else {
            advance()
        }
        var condition: WGSLExpression?
        if !check(";") { condition = try parseExpression() }
        try expect(";", "for condition")
        var update: WGSLStatement?
        if !check(")") { update = try parseSimpleAssignment() }
        try expect(")", "end of for header")
        return .forStatement(initializer: initializer, condition: condition, update: update, body: try parseBlock())
    }

    private mutating func parseLoopStatement() throws -> WGSLStatement {
        try expect("{", "loop body")
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
        try expect("}", "end of loop")
        return .loopStatement(body: body, continuing: continuing)
    }

    private mutating func parseSwitchStatement() throws -> WGSLStatement {
        let subject = try parseExpression()
        try expect("{", "switch body")
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
                throw error("only case/default may appear inside a switch — got: '\(current.text)'")
            }
        }
        try expect("}", "end of switch")
        return .switchStatement(subject: subject, cases: cases)
    }

    /// An assignment or call statement. Consumes through the semicolon.
    private mutating func parseAssignmentOrCall() throws -> WGSLStatement {
        let statement = try parseSimpleAssignment()
        try expect(";", "statement")
        return statement
    }

    /// An assignment/increment/call that does not consume the semicolon (also used in a for update clause).
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

    // MARK: - Expressions (precedence climbing)

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
                expression = .member(expression, try expectIdentifier("member name"))
            } else if match("[") {
                let subscriptExpression = try parseExpression()
                try expect("]", "index")
                expression = .index(expression, subscriptExpression)
            } else {
                return expression
            }
        }
    }

    private mutating func parsePrimary() throws -> WGSLExpression {
        if match("(") {
            let inner = try parseExpression()
            try expect(")", "parenthesized expression")
            return .paren(inner)
        }
        if current.kind == .intLiteral { return .intLiteral(advance().text) }
        if current.kind == .floatLiteral { return .floatLiteral(advance().text) }

        guard current.kind == .identifier else {
            throw error("expected an expression — got: '\(current.text)'")
        }
        let name = advance().text
        if name == "true" { return .boolLiteral(true) }
        if name == "false" { return .boolLiteral(false) }

        var typeArguments: [WGSLType] = []
        if check("<"), Self.templateNames.contains(name) {
            if name == "array" {
                // The second argument of `array<f32, 3>(…)` is a length expression, not a type.
                // The type parser already knows that shape, so we rewind to the name token and reuse it.
                index -= 1
                if case .array(let element, _) = try parseType() {
                    typeArguments = [element]
                }
            } else {
                advance()
                repeat {
                    typeArguments.append(try parseType())
                } while match(",")
                try expectGenericClose("generic argument")
            }
        }

        if match("(") {
            var arguments: [WGSLExpression] = []
            while !check(")"), !isAtEnd {
                arguments.append(try parseExpression())
                if !match(",") { break }
            }
            try expect(")", "call arguments")
            return .call(callee: name, typeArguments: typeArguments, arguments: arguments)
        }

        // A generic name with no arguments (a type position such as `array<f32>`) cannot appear as an expression.
        guard typeArguments.isEmpty else {
            throw error("call arguments are required after '\(name)'")
        }
        return .identifier(name)
    }
}
