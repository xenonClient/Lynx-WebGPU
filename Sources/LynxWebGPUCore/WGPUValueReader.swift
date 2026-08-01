import Foundation

/// Lynx NativeModule이 넘긴 `[String: Any]`를 타입 안전하게 읽는 리더.
///
/// JSON 문자열을 거치지 않는다 — Lynx는 이미 `NSDictionary`/`NSNumber`/`NSString`로 변환해서 주므로
/// 한 번 더 직렬화하면 프레임당 비용만 늘어난다. 대신 실패 시 **경로가 붙은 오류**를 던져
/// 커맨드 스트림 어디가 잘못됐는지 JS 쪽에서 바로 알 수 있게 한다.
public struct WGPUValueReader {
    public let path: String
    private let dictionary: [String: Any]

    public init(_ dictionary: [String: Any], path: String = "") {
        self.dictionary = dictionary
        self.path = path
    }

    public var keys: [String] { Array(dictionary.keys) }

    public func has(_ key: String) -> Bool {
        guard let value = dictionary[key] else { return false }
        return !(value is NSNull)
    }

    private func childPath(_ key: String) -> String {
        path.isEmpty ? key : "\(path).\(key)"
    }

    private func value(_ key: String) -> Any? {
        guard let value = dictionary[key], !(value is NSNull) else { return nil }
        return value
    }

    private func missing(_ key: String) -> WGPUError {
        .validation("필수 필드 누락", path: childPath(key))
    }

    private func mismatch(_ key: String, expected: String) -> WGPUError {
        .validation("\(expected) 이(가) 필요하다", path: childPath(key))
    }

    // MARK: - 스칼라

    public func requiredInt(_ key: String) throws -> Int {
        guard let raw = value(key) else { throw missing(key) }
        guard let number = raw as? NSNumber else { throw mismatch(key, expected: "정수") }
        return number.intValue
    }

    public func int(_ key: String, default fallback: Int) -> Int {
        (value(key) as? NSNumber)?.intValue ?? fallback
    }

    public func optionalInt(_ key: String) -> Int? {
        (value(key) as? NSNumber)?.intValue
    }

    public func requiredDouble(_ key: String) throws -> Double {
        guard let raw = value(key) else { throw missing(key) }
        guard let number = raw as? NSNumber else { throw mismatch(key, expected: "숫자") }
        return number.doubleValue
    }

    public func double(_ key: String, default fallback: Double) -> Double {
        (value(key) as? NSNumber)?.doubleValue ?? fallback
    }

    public func bool(_ key: String, default fallback: Bool) -> Bool {
        (value(key) as? NSNumber)?.boolValue ?? fallback
    }

    public func requiredString(_ key: String) throws -> String {
        guard let raw = value(key) else { throw missing(key) }
        guard let string = raw as? String else { throw mismatch(key, expected: "문자열") }
        return string
    }

    public func string(_ key: String, default fallback: String) -> String {
        (value(key) as? String) ?? fallback
    }

    public func optionalString(_ key: String) -> String? {
        value(key) as? String
    }

    // MARK: - 핸들

    public func requiredHandle(_ key: String) throws -> WGPUHandle {
        WGPUHandle(try requiredInt(key))
    }

    public func optionalHandle(_ key: String) -> WGPUHandle? {
        optionalInt(key).map(WGPUHandle.init)
    }

    // MARK: - 열거형 / 플래그

    public func requiredEnum<T: RawRepresentable & CaseIterable>(_ key: String, _ type: T.Type) throws -> T
    where T.RawValue == String {
        let raw = try requiredString(key)
        guard let parsed = T(rawValue: raw) else {
            throw WGPUError.validation("알 수 없는 값 \"\(raw)\" (가능: \(Self.allowed(type)))", path: childPath(key))
        }
        return parsed
    }

    public func enumValue<T: RawRepresentable & CaseIterable>(_ key: String, default fallback: T) throws -> T
    where T.RawValue == String {
        guard has(key) else { return fallback }
        return try requiredEnum(key, T.self)
    }

    public func optionalEnum<T: RawRepresentable & CaseIterable>(_ key: String, _ type: T.Type) throws -> T?
    where T.RawValue == String {
        guard has(key) else { return nil }
        return try requiredEnum(key, type)
    }

    private static func allowed<T: RawRepresentable & CaseIterable>(_ type: T.Type) -> String
    where T.RawValue == String {
        T.allCases.map { "\($0.rawValue)" }.joined(separator: ", ")
    }

    public func flags<T: OptionSet>(_ key: String, _ type: T.Type, default fallback: T) -> T where T.RawValue == Int {
        guard let number = value(key) as? NSNumber else { return fallback }
        return T(rawValue: number.intValue)
    }

    public func requiredFlags<T: OptionSet>(_ key: String, _ type: T.Type) throws -> T where T.RawValue == Int {
        T(rawValue: try requiredInt(key))
    }

    // MARK: - 중첩 구조

    public func requiredObject(_ key: String) throws -> WGPUValueReader {
        guard let raw = value(key) else { throw missing(key) }
        guard let dictionary = raw as? [String: Any] else { throw mismatch(key, expected: "객체") }
        return WGPUValueReader(dictionary, path: childPath(key))
    }

    public func object(_ key: String) -> WGPUValueReader? {
        guard let dictionary = value(key) as? [String: Any] else { return nil }
        return WGPUValueReader(dictionary, path: childPath(key))
    }

    public func requiredObjects(_ key: String) throws -> [WGPUValueReader] {
        guard let raw = value(key) else { throw missing(key) }
        guard let array = raw as? [Any] else { throw mismatch(key, expected: "배열") }
        return try array.enumerated().map { index, element in
            guard let dictionary = element as? [String: Any] else {
                throw WGPUError.validation("객체가 필요하다", path: "\(childPath(key))[\(index)]")
            }
            return WGPUValueReader(dictionary, path: "\(childPath(key))[\(index)]")
        }
    }

    public func objects(_ key: String) throws -> [WGPUValueReader] {
        guard has(key) else { return [] }
        return try requiredObjects(key)
    }

    public func numbers(_ key: String) throws -> [Double] {
        guard let raw = value(key) else { throw missing(key) }
        guard let array = raw as? [Any] else { throw mismatch(key, expected: "숫자 배열") }
        return try array.enumerated().map { index, element in
            guard let number = element as? NSNumber else {
                throw WGPUError.validation("숫자가 필요하다", path: "\(childPath(key))[\(index)]")
            }
            return number.doubleValue
        }
    }

    public func integers(_ key: String) throws -> [Int] {
        try numbers(key).map { Int($0) }
    }

    /// `{ "name": 값 }` 형태의 숫자 맵 (파이프라인 상수 `constants`).
    public func numberMap(_ key: String) -> [String: Double] {
        guard let raw = value(key) as? [String: Any] else { return [:] }
        return raw.compactMapValues { ($0 as? NSNumber)?.doubleValue }
    }

    public func handles(_ key: String) throws -> [WGPUHandle] {
        try integers(key).map(WGPUHandle.init)
    }

    // MARK: - 바이너리

    /// 바이트열을 읽는다. 세 가지 표현을 모두 받는다:
    ///
    /// - `NSData` — **JS 셰임이 쓰는 경로.** Lynx가 `ArrayBuffer`를 변환해 준 것이다.
    /// - base64 문자열 — 손으로 쓰는 커맨드 스트림(테스트 하네스, `RenderHarness.base64`)용.
    /// - 숫자 배열 — 소량 디버그용.
    public func requiredData(_ key: String) throws -> Data {
        guard let raw = value(key) else { throw missing(key) }
        if let data = raw as? Data { return data }
        if let string = raw as? String {
            guard let decoded = Data(base64Encoded: string) else {
                throw WGPUError.validation("base64 디코딩 실패", path: childPath(key))
            }
            return decoded
        }
        if let array = raw as? [Any] {
            var bytes = [UInt8]()
            bytes.reserveCapacity(array.count)
            for (index, element) in array.enumerated() {
                guard let number = element as? NSNumber else {
                    throw WGPUError.validation("바이트(0~255)가 필요하다", path: "\(childPath(key))[\(index)]")
                }
                bytes.append(UInt8(truncatingIfNeeded: number.intValue))
            }
            return Data(bytes)
        }
        throw mismatch(key, expected: "ArrayBuffer · base64 문자열 · 바이트 배열")
    }

    public func optionalData(_ key: String) throws -> Data? {
        guard has(key) else { return nil }
        return try requiredData(key)
    }

    // MARK: - 복합 값

    /// `{r,g,b,a}` 또는 `[r,g,b,a]` 두 표기를 모두 받는다 (WebGPU `GPUColor`).
    public func color(_ key: String, default fallback: WGPUColor) throws -> WGPUColor {
        guard let raw = value(key) else { return fallback }
        if let dictionary = raw as? [String: Any] {
            let reader = WGPUValueReader(dictionary, path: childPath(key))
            return WGPUColor(
                red: reader.double("r", default: 0),
                green: reader.double("g", default: 0),
                blue: reader.double("b", default: 0),
                alpha: reader.double("a", default: 1)
            )
        }
        if let array = raw as? [Any] {
            let components = array.compactMap { ($0 as? NSNumber)?.doubleValue }
            guard components.count >= 3 else {
                throw WGPUError.validation("색은 성분 3개 이상이 필요하다", path: childPath(key))
            }
            return WGPUColor(
                red: components[0],
                green: components[1],
                blue: components[2],
                alpha: components.count > 3 ? components[3] : 1
            )
        }
        throw mismatch(key, expected: "색 ({r,g,b,a} 또는 [r,g,b,a])")
    }

    /// `{width,height,depthOrArrayLayers}` 또는 `[w,h,d]` (WebGPU `GPUExtent3D`).
    public func requiredExtent(_ key: String) throws -> WGPUExtent3D {
        guard let raw = value(key) else { throw missing(key) }
        if let dictionary = raw as? [String: Any] {
            let reader = WGPUValueReader(dictionary, path: childPath(key))
            return WGPUExtent3D(
                width: try reader.requiredInt("width"),
                height: reader.int("height", default: 1),
                depthOrArrayLayers: reader.int("depthOrArrayLayers", default: 1)
            )
        }
        if let array = raw as? [Any] {
            let components = array.compactMap { ($0 as? NSNumber)?.intValue }
            guard let width = components.first else {
                throw WGPUError.validation("크기는 성분 1개 이상이 필요하다", path: childPath(key))
            }
            return WGPUExtent3D(
                width: width,
                height: components.count > 1 ? components[1] : 1,
                depthOrArrayLayers: components.count > 2 ? components[2] : 1
            )
        }
        throw mismatch(key, expected: "크기 ({width,height,…} 또는 [w,h,d])")
    }

    /// `{x,y,z}` 또는 `[x,y,z]` (WebGPU `GPUOrigin3D`). 없으면 원점.
    public func origin(_ key: String) throws -> WGPUOrigin3D {
        guard let raw = value(key) else { return WGPUOrigin3D() }
        if let dictionary = raw as? [String: Any] {
            let reader = WGPUValueReader(dictionary, path: childPath(key))
            return WGPUOrigin3D(
                x: reader.int("x", default: 0),
                y: reader.int("y", default: 0),
                z: reader.int("z", default: 0)
            )
        }
        if let array = raw as? [Any] {
            let components = array.compactMap { ($0 as? NSNumber)?.intValue }
            return WGPUOrigin3D(
                x: components.count > 0 ? components[0] : 0,
                y: components.count > 1 ? components[1] : 0,
                z: components.count > 2 ? components[2] : 0
            )
        }
        throw mismatch(key, expected: "원점 ({x,y,z} 또는 [x,y,z])")
    }
}

/// WebGPU `GPUColor`.
public struct WGPUColor: Equatable, Sendable {
    public var red: Double
    public var green: Double
    public var blue: Double
    public var alpha: Double

    public init(red: Double = 0, green: Double = 0, blue: Double = 0, alpha: Double = 1) {
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha
    }

    public static let transparent = WGPUColor(red: 0, green: 0, blue: 0, alpha: 0)
    public static let black = WGPUColor(red: 0, green: 0, blue: 0, alpha: 1)
}

/// WebGPU `GPUExtent3D`.
public struct WGPUExtent3D: Equatable, Sendable {
    public var width: Int
    public var height: Int
    public var depthOrArrayLayers: Int

    public init(width: Int, height: Int = 1, depthOrArrayLayers: Int = 1) {
        self.width = width
        self.height = height
        self.depthOrArrayLayers = depthOrArrayLayers
    }
}

/// WebGPU `GPUOrigin3D`.
public struct WGPUOrigin3D: Equatable, Sendable {
    public var x: Int
    public var y: Int
    public var z: Int

    public init(x: Int = 0, y: Int = 0, z: Int = 0) {
        self.x = x
        self.y = y
        self.z = z
    }
}
