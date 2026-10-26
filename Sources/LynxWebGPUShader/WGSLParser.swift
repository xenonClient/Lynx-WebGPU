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
    /// How deeply types, expressions and statements are nested at the current token. Every one of
    /// those is parsed recursively, so untrusted source (`((((…))))` or `array<array<…>>` a few
    /// hundred thousand deep) would otherwise overflow the stack — `enterNesting()` caps it.
    private var nestingDepth = 0

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
        var module = try parser.parseModule()
        try module.resolveIntegerAttributes()
        try module.validateWorkgroupSizes()
        return module
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

    /// The deepest nesting of types, expressions and statements a module may use — Tint's parser
    /// limit. Real shaders stay far below it; past it parsing fails instead of exhausting the stack.
    static let maximumNestingDepth = 128

    /// The tallest expression tree a module may contain.
    ///
    /// `maximumNestingDepth` counts parser recursion, and a flat chain (`a + a + … + a`,
    /// `v.x.x.…`, `m[0][0]…`) is built in a loop — so it never counts, while the tree it builds is as
    /// tall as the chain is long and **every later walk recurses once per node along it**
    /// (reflection, emission). A hundred-thousand-term chain overflowed the emitter's stack: a
    /// SIGSEGV of the host from shader text in a bundle. So the finished tree's height is measured
    /// (`WGSLExpression.height`, iteratively) and capped here. Real shaders are nowhere near it —
    /// 1024 chained operators is far beyond what hand-written or machine-generated code produces —
    /// and 1024 recursive frames fit the 16 MB `WGSLDeepStack` with room to spare.
    static let maximumExpressionHeight = 1024

    /// How many `parseExpression` calls are on the stack — the height check runs once, at the
    /// outermost one, so a long chain is measured once rather than once per nested operand.
    private var expressionDepth = 0

    /// Rejects a finished expression whose tree is taller than `maximumExpressionHeight`.
    private func checkHeight(of expression: WGSLExpression, line: Int) throws {
        guard expression.height <= Self.maximumExpressionHeight else {
            throw error(
                "the expression is deeper than \(Self.maximumExpressionHeight) levels "
                + "— a chain of that many operators, members or indices is not translatable",
                line: line
            )
        }
    }

    /// Called on entry to every recursive production; pair it with `defer { nestingDepth -= 1 }`.
    private mutating func enterNesting() throws {
        nestingDepth += 1
        guard nestingDepth <= Self.maximumNestingDepth else {
            throw error("nesting is deeper than \(Self.maximumNestingDepth) levels")
        }
    }

    private func error(_ message: String, line: Int? = nil) -> WGPUError {
        // Carry the line number **as a number too** — this is the value `getCompilationInfo()` hands the editor.
        let line = line ?? current.line
        return WGPUError(
            kind: .validation,
            message: "WGSL parse failed (line \(line)): \(message)",
            line: line
        )
    }

    // MARK: - Module

    private mutating func parseModule() throws -> WGSLModule {
        var module = WGSLModule()
        // Module-scope name → the line it was first declared on.
        var declaredLines: [String: Int] = [:]
        while !isAtEnd {
            if match(";") { continue }
            let attributes = try parseAttributes()
            let declarationLine = current.line

            if check("struct") {
                advance()
                let structure = try parseStruct()
                try declare(structure.name, line: declarationLine, in: &declaredLines)
                module.structs.append(structure)
            } else if check("fn") {
                advance()
                let function = try parseFunction(attributes: attributes)
                try declare(function.name, line: declarationLine, in: &declaredLines)
                module.functions.append(function)
            } else if check("var") {
                advance()
                let global = try parseGlobalVariable(attributes: attributes)
                try declare(global.name, line: declarationLine, in: &declaredLines)
                module.globals.append(global)
            } else if check("const") || check("override") {
                let isOverride = current.text == "override"
                advance()
                let constant = try parseModuleConstant(isOverride: isOverride)
                try declare(constant.name, line: declarationLine, in: &declaredLines)
                module.constants.append(constant)
            } else if check("alias") || check("type") {
                advance()
                let name = try expectIdentifier("alias name")
                try expect("=", "alias")
                let type = try parseType()
                try expect(";", "alias")
                try declare(name, line: declarationLine, in: &declaredLines)
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
        // Every module-scope name is in `declaredLines` once (redeclarations were rejected above),
        // so the type checks can point back at a struct or alias by name.
        try Self.validateTypeDeclarations(module, lines: declaredLines)
        return module
    }

    /// Registers a module-scope name — **structs, aliases, functions, variables, constants and overrides
    /// share one namespace** in WGSL, so a second declaration of any kind is a shader-creation error
    /// (Tint rejects it the same way). Letting it through would leave two declarations that every
    /// name-keyed table downstream (reflection, resource threading, emission) silently picks between.
    private func declare(_ name: String, line: Int, in declaredLines: inout [String: Int]) throws {
        if let first = declaredLines[name] {
            throw error(
                "redeclaration of '\(name)' (first declared on line \(first)) — module-scope "
                    + "structs, aliases, functions, variables, constants and overrides share one namespace",
                line: line
            )
        }
        declaredLines[name] = line
    }

    // MARK: - Type declaration checks

    /// The deepest composite type nesting a module may declare (Tint's limit — the WGSL spec only
    /// guarantees 15). It also bounds every recursive walk over a type (layout, emission).
    static let maximumCompositeNesting = 255

    /// Rejects what the grammar cannot: **recursive types** and over-deep composite nesting.
    ///
    /// WGSL forbids a type that contains itself — directly (`struct S { a: S }`), through an array
    /// (`array<S, 2>`), through an alias or through another struct — because it would have infinite
    /// size. The layout calculator and the struct emitter walk these graphs recursively, so a cycle
    /// that got past here would recurse until the stack overflowed. Checked once at module creation,
    /// with the declaration's line, so nothing downstream ever sees one. The depth limit does the same
    /// for long acyclic chains (`S1 { a: S0 }`, `S2 { a: S1 }`, …) an untrusted bundle can declare.
    private static func validateTypeDeclarations(_ module: WGSLModule, lines: [String: Int]) throws {
        var depths: [String: Int] = [:]
        // Declarations currently being walked — meeting one again is a cycle.
        var path: [String] = []

        func failure(_ name: String, _ message: String) -> WGPUError {
            let line = lines[name] ?? 0
            return WGPUError(kind: .validation, message: "WGSL validation failed (line \(line)): \(message)", line: line)
        }

        func nesting(ofDeclaration name: String) throws -> Int {
            if let known = depths[name] { return known }
            if let start = path.firstIndex(of: name) {
                let cycle = (path[start...] + [name]).joined(separator: " → ")
                throw failure(name, "recursive type '\(name)' (\(cycle)) — WGSL types cannot contain themselves")
            }
            guard path.count < maximumCompositeNesting else {
                throw failure(name, "type declarations nest more than \(maximumCompositeNesting) levels deep")
            }
            path.append(name)
            defer { path.removeLast() }

            let depth: Int
            if let structure = module.structNamed(name) {
                depth = 1 + (try structure.members.map { try nesting(of: $0.type) }.max() ?? 0)
            } else if let alias = module.aliases.first(where: { $0.name == name }) {
                depth = try nesting(of: alias.type)
            } else {
                depth = 0   // not a declared type — undefined names are left to later stages
            }
            guard depth <= maximumCompositeNesting else {
                throw failure(name, "'\(name)' nests composite types \(depth) levels deep (limit \(maximumCompositeNesting))")
            }
            depths[name] = depth
            return depth
        }

        func nesting(of type: WGSLType) throws -> Int {
            switch type {
            case .vector: return 1
            case .matrix: return 2
            case .array(let element, _): return 1 + (try nesting(of: element))
            case .named(let name): return try nesting(ofDeclaration: name)
            // Scalars, atomics and handle/pointer types do not embed another type's storage.
            default: return 0
            }
        }

        for structure in module.structs { _ = try nesting(ofDeclaration: structure.name) }
        for alias in module.aliases { _ = try nesting(ofDeclaration: alias.name) }
    }

    private mutating func parseAttributes() throws -> [WGSLAttribute] {
        var attributes: [WGSLAttribute] = []
        while check("@") {
            advance()
            let line = current.line
            let name = try expectIdentifier("attribute name")
            var arguments: [String] = []
            var expressions: [WGSLExpression?] = []
            if match("(") {
                repeat {
                    if check(")") { break }
                    let argument = parseAttributeArgument()
                    arguments.append(argument.text)
                    expressions.append(argument.expression)
                } while match(",")
                try expect(")", "attribute arguments")
            }
            attributes.append(WGSLAttribute(name: name, arguments: arguments, expressions: expressions, line: line))
        }
        return attributes
    }

    /// One attribute argument — as an expression when it parses as one (`0x10`, `BASE + 1`), and as
    /// source text always. An argument that is not an expression is kept as text only, so an attribute
    /// this parser has no grammar for still passes as it did.
    private mutating func parseAttributeArgument() -> (text: String, expression: WGSLExpression?) {
        let start = index
        // `expectGenericClose` rewrites tokens in place, so the text and a rollback read the originals.
        let original = tokens
        let originalTemplateDepth = templateDepth
        if let expression = try? parseExpression(), check(",") || check(")") {
            return (original[start..<index].map(\.text).joined(), expression)
        }
        index = start
        tokens = original
        templateDepth = originalTemplateDepth

        var text = ""
        var depth = 0
        while !isAtEnd {
            if depth == 0, check(",") || check(")") { break }
            if check("(") { depth += 1 }
            if check(")") { depth -= 1 }
            text += advance().text
        }
        return (text, nil)
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
        try enterNesting()
        defer { nestingDepth -= 1 }
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
        try enterNesting()
        defer { nestingDepth -= 1 }
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
            // `break if cond;` (the last statement of a `continuing` block) means exactly
            // `if cond { break; }` — it is parsed as that, so no later pass needs a new case. The
            // `continuing` statements are emitted outside the body wrapper, where `break` leaves the loop.
            if match("if") {
                let condition = try parseExpression()
                try expect(";", "break if")
                return .ifStatement(
                    clauses: [WGSLIfClause(condition: condition, body: [.breakStatement])], elseBody: nil
                )
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

    /// The `else if` arms are read in a loop (as Tint does) — a chain's length is not nesting, so it
    /// neither recurses nor counts toward `maximumNestingDepth`.
    private mutating func parseIfStatement() throws -> WGSLStatement {
        var clauses = [WGSLIfClause(condition: try parseExpression(), body: try parseBlock())]
        var elseBody: [WGSLStatement]?
        while match("else") {
            guard check("if") else {
                elseBody = try parseBlock()
                break
            }
            advance()
            clauses.append(WGSLIfClause(condition: try parseExpression(), body: try parseBlock()))
        }
        return .ifStatement(clauses: clauses, elseBody: elseBody)
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
        try enterNesting()
        defer { nestingDepth -= 1 }
        expressionDepth += 1
        defer { expressionDepth -= 1 }
        let line = current.line
        let expression = try parseBinary(minimumLevel: 0)
        // Only the outermost expression is measured — its height covers every nested operand.
        if expressionDepth == 1 { try checkHeight(of: expression, line: line) }
        return expression
    }

    /// Precedence climbing — one call per operand instead of one per precedence level, so a nested
    /// parenthesis costs a handful of stack frames rather than a dozen (the depth limit has to fit a
    /// 512 KB thread stack). Operators of one level associate left.
    private mutating func parseBinary(minimumLevel: Int) throws -> WGSLExpression {
        var left = try parseUnary()
        while let level = binaryOperatorLevel(), level >= minimumLevel {
            let op = advance().text
            let right = try parseBinary(minimumLevel: level + 1)
            left = .binary(op: op, left, right)
        }
        return left
    }

    /// The precedence level of the binary operator at the current token, if it is one here.
    private func binaryOperatorLevel() -> Int? {
        guard current.kind == .punctuation,
              !(templateDepth > 0 && Self.templateClosers.contains(current.text)) else { return nil }
        return Self.precedence.firstIndex { $0.contains(current.text) }
    }

    private mutating func parseUnary() throws -> WGSLExpression {
        // A prefix operator recurses without passing through parseExpression (`- - - … x`), so it counts too.
        if check("-") || check("!") || check("~") {
            let op = advance().text
            try enterNesting()
            defer { nestingDepth -= 1 }
            return .unary(op: op, try parseUnary())
        }
        if check("&") {
            advance()
            try enterNesting()
            defer { nestingDepth -= 1 }
            return .addressOf(try parseUnary())
        }
        if check("*") {
            advance()
            try enterNesting()
            defer { nestingDepth -= 1 }
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
                // The **whole array type** is kept: the zero-value form `array<u32, 4>()` has no
                // arguments to count, so the length lives only here.
                index -= 1
                typeArguments = [try parseType()]
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
