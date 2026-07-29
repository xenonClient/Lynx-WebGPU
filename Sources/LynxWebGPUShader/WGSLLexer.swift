import Foundation
import LynxWebGPUCore

/// WGSL 토큰.
struct WGSLToken: Equatable {
    enum Kind: Equatable {
        case identifier
        case intLiteral
        case floatLiteral
        case punctuation
        case endOfFile
    }

    let kind: Kind
    let text: String
    let line: Int

    static func punctuation(_ text: String, line: Int = 0) -> WGSLToken {
        WGSLToken(kind: .punctuation, text: text, line: line)
    }
}

/// WGSL 소스를 토큰열로 자른다.
///
/// 키워드를 따로 분류하지 않는다 — 파서가 문맥에서 식별자 텍스트로 판별하는 편이
/// WGSL의 문맥 의존 문법(`<`가 제네릭인지 비교인지)을 다루기 쉽다.
struct WGSLLexer {
    private let source: [Character]
    private var index = 0
    private var line = 1

    init(_ source: String) {
        self.source = Array(source)
    }

    /// 긴 것부터 시도해야 `<<=`가 `<` + `<=`로 잘리지 않는다.
    private static let operators = [
        "<<=", ">>=",
        "->", "&&", "||", "<<", ">>", "<=", ">=", "==", "!=",
        "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "++", "--",
        "+", "-", "*", "/", "%", "=", "<", ">", "!", "&", "|", "^", "~",
        "(", ")", "[", "]", "{", "}", ",", ";", ":", ".", "@",
    ]

    static func tokenize(_ source: String) throws -> [WGSLToken] {
        var lexer = WGSLLexer(source)
        return try lexer.run()
    }

    private mutating func run() throws -> [WGSLToken] {
        var tokens: [WGSLToken] = []
        while true {
            try skipTrivia()
            guard index < source.count else { break }
            let character = source[index]

            if character.isLetter || character == "_" {
                tokens.append(lexIdentifier())
            } else if character.isNumber || (character == "." && index + 1 < source.count && source[index + 1].isNumber) {
                tokens.append(lexNumber())
            } else if let op = matchOperator() {
                tokens.append(WGSLToken(kind: .punctuation, text: op, line: line))
            } else {
                throw WGPUError.validation("WGSL: 알 수 없는 문자 '\(character)' (line \(line))")
            }
        }
        tokens.append(WGSLToken(kind: .endOfFile, text: "", line: line))
        return tokens
    }

    private mutating func skipTrivia() throws {
        while index < source.count {
            let character = source[index]
            if character == "\n" {
                line += 1
                index += 1
            } else if character.isWhitespace {
                index += 1
            } else if character == "/", index + 1 < source.count, source[index + 1] == "/" {
                while index < source.count, source[index] != "\n" { index += 1 }
            } else if character == "/", index + 1 < source.count, source[index + 1] == "*" {
                try skipBlockComment()
            } else {
                return
            }
        }
    }

    /// WGSL의 블록 주석은 중첩된다.
    private mutating func skipBlockComment() throws {
        var depth = 0
        repeat {
            guard index < source.count else {
                throw WGPUError.validation("WGSL: 닫히지 않은 블록 주석 (line \(line))")
            }
            if source[index] == "/", index + 1 < source.count, source[index + 1] == "*" {
                depth += 1
                index += 2
            } else if source[index] == "*", index + 1 < source.count, source[index + 1] == "/" {
                depth -= 1
                index += 2
            } else {
                if source[index] == "\n" { line += 1 }
                index += 1
            }
        } while depth > 0
    }

    private mutating func lexIdentifier() -> WGSLToken {
        let start = index
        while index < source.count, source[index].isLetter || source[index].isNumber || source[index] == "_" {
            index += 1
        }
        return WGSLToken(kind: .identifier, text: String(source[start..<index]), line: line)
    }

    private mutating func lexNumber() -> WGSLToken {
        let start = index
        var isFloat = false

        if source[index] == "0", index + 1 < source.count, source[index + 1] == "x" || source[index + 1] == "X" {
            index += 2
            while index < source.count, source[index].isHexDigit { index += 1 }
        } else {
            while index < source.count, source[index].isNumber { index += 1 }
            if index < source.count, source[index] == "." {
                isFloat = true
                index += 1
                while index < source.count, source[index].isNumber { index += 1 }
            }
            if index < source.count, source[index] == "e" || source[index] == "E" {
                isFloat = true
                index += 1
                if index < source.count, source[index] == "+" || source[index] == "-" { index += 1 }
                while index < source.count, source[index].isNumber { index += 1 }
            }
        }
        // 접미사: u(uint) i(int) f(f32) h(f16)
        if index < source.count, "uifh".contains(source[index]) {
            if source[index] == "f" || source[index] == "h" { isFloat = true }
            index += 1
        }
        return WGSLToken(
            kind: isFloat ? .floatLiteral : .intLiteral,
            text: String(source[start..<index]),
            line: line
        )
    }

    private mutating func matchOperator() -> String? {
        for op in Self.operators {
            let characters = Array(op)
            guard index + characters.count <= source.count else { continue }
            if Array(source[index..<(index + characters.count)]) == characters {
                index += characters.count
                return op
            }
        }
        return nil
    }
}
