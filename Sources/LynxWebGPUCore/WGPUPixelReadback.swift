import Foundation

/// GPU에서 되읽은 2D 픽셀 블록.
///
/// **바이트만 돌려주지 않는다.** 되읽기 결과를 `Data`로만 넘기면 호출 측이 "RGBA8일 것"을
/// 암묵적으로 가정하게 되고, `rgba16float` 표면에서는 오류 없이 조용히 틀린 값을 읽는다.
/// 그래서 포맷·크기·행 간격을 값과 함께 묶어 둔다 — 해석에 필요한 것이 전부 여기 있다.
///
/// 채널 값이 필요하면 `rgba(x:y:)`를 쓴다. 포맷을 직접 다루고 싶으면 `data`를 그대로 읽되
/// **행 간격은 `bytesPerRow`를 봐야 한다** (`width * bytesPerPixel`과 다를 수 있다).
public struct WGPUPixelReadback: Sendable, Equatable {
    /// 픽셀 바이트. 행 하나가 `bytesPerRow` 바이트, 행이 `height`개다.
    public let data: Data
    /// `data`가 어떤 포맷인지. 이것을 보지 않고 해석하면 안 된다.
    public let format: WGPUTextureFormat
    public let width: Int
    public let height: Int
    /// 행 간격. 패딩이 있으면 `width * format.bytesPerPixel`보다 크다.
    public let bytesPerRow: Int

    public init(data: Data, format: WGPUTextureFormat, width: Int, height: Int, bytesPerRow: Int) {
        self.data = data
        self.format = format
        self.width = width
        self.height = height
        self.bytesPerRow = bytesPerRow
    }

    /// 픽셀 1개의 바이트 수.
    public var bytesPerPixel: Int { format.bytesPerPixel }

    /// 픽셀 하나를 RGBA float으로 편다.
    ///
    /// 없는 채널은 RGB가 0, A가 1로 채워진다 (WebGPU가 셰이더에서 텍스처를 읽을 때와 같은 규칙).
    /// `bgra8unorm` 계열은 RGBA 순서로 바꿔 돌려준다.
    ///
    /// **색공간 변환은 하지 않는다.** `-srgb` 포맷도 저장된 값을 그대로 0~1로 정규화할 뿐이라,
    /// 선형 값이 필요하면 호출 측이 변환한다. float 포맷은 정규화하지 않으므로
    /// **1.0을 넘는 값과 음수가 그대로 살아서 나온다** — HDR 결과를 검증하는 근거가 이것이다.
    ///
    /// - Throws: 좌표가 범위를 벗어나거나, 채널로 풀 수 없는 포맷(팩된 포맷·정수 포맷 등)이면
    ///           `WGPUError.validation`. 그런 포맷은 `data`를 직접 해석해야 한다.
    public func rgba(x: Int, y: Int) throws -> SIMD4<Float> {
        guard x >= 0, y >= 0, x < width, y < height else {
            throw WGPUError.validation("픽셀 (\(x), \(y))이 \(width)×\(height) 범위를 벗어났다")
        }
        guard let layout = Self.channelLayout(of: format) else {
            throw WGPUError.validation(
                "\(format.rawValue)은 rgba(x:y:)로 풀 수 없는 포맷이다 — data를 직접 해석할 것"
            )
        }
        let stride = bytesPerPixel
        let offset = y * bytesPerRow + x * stride
        guard offset >= 0, offset + stride <= data.count else {
            throw WGPUError.validation(
                "픽셀 (\(x), \(y))의 바이트가 버퍼(\(data.count)B) 밖이다 — bytesPerRow(\(bytesPerRow))를 확인할 것"
            )
        }

        var result = SIMD4<Float>(0, 0, 0, 1)
        data.withUnsafeBytes { raw in
            for channel in 0..<layout.count {
                switch layout.kind {
                case .unorm8:
                    let byte = raw.load(fromByteOffset: offset + channel, as: UInt8.self)
                    result[channel] = Float(byte) / 255
                case .snorm8:
                    // snorm은 -128과 -127이 모두 -1.0이다 (WebGPU 명세의 클램프 규칙).
                    let byte = raw.load(fromByteOffset: offset + channel, as: Int8.self)
                    result[channel] = max(Float(byte) / 127, -1)
                case .float16:
                    let bits = raw.loadUnaligned(fromByteOffset: offset + channel * 2, as: UInt16.self)
                    result[channel] = Self.float(fromHalf: bits)
                case .float32:
                    let bits = raw.loadUnaligned(fromByteOffset: offset + channel * 4, as: UInt32.self)
                    result[channel] = Float(bitPattern: bits)
                }
            }
            if layout.isBGRA {
                result = SIMD4<Float>(result.z, result.y, result.x, result.w)
            }
        }
        return result
    }

    // MARK: - 채널 배치

    private enum ChannelKind {
        case unorm8, snorm8, float16, float32
    }

    private struct ChannelLayout {
        let kind: ChannelKind
        let count: Int
        let isBGRA: Bool

        init(_ kind: ChannelKind, _ count: Int, isBGRA: Bool = false) {
            self.kind = kind
            self.count = count
            self.isBGRA = isBGRA
        }
    }

    /// 채널이 균일한 크기로 늘어서 있는 포맷만 다룬다.
    /// `rgb10a2unorm`·`rgb10a2uint`·`rg11b10ufloat`·`rgb9e5ufloat`처럼 비트가 채널 경계를
    /// 넘어 팩된 것과 정수 포맷은 nil —
    /// 정규화된 float으로 펴는 것이 오히려 값을 왜곡하므로 호출 측에 넘긴다.
    private static func channelLayout(of format: WGPUTextureFormat) -> ChannelLayout? {
        switch format {
        case .r8unorm: return ChannelLayout(.unorm8, 1)
        case .rg8unorm: return ChannelLayout(.unorm8, 2)
        case .rgba8unorm, .rgba8unormSRGB: return ChannelLayout(.unorm8, 4)
        case .bgra8unorm, .bgra8unormSRGB: return ChannelLayout(.unorm8, 4, isBGRA: true)
        case .r8snorm: return ChannelLayout(.snorm8, 1)
        case .rg8snorm: return ChannelLayout(.snorm8, 2)
        case .rgba8snorm: return ChannelLayout(.snorm8, 4)
        case .r16float: return ChannelLayout(.float16, 1)
        case .rg16float: return ChannelLayout(.float16, 2)
        case .rgba16float: return ChannelLayout(.float16, 4)
        case .r32float: return ChannelLayout(.float32, 1)
        case .rg32float: return ChannelLayout(.float32, 2)
        case .rgba32float: return ChannelLayout(.float32, 4)
        default: return nil
        }
    }

    // MARK: - half → float

    /// IEEE 754 binary16 비트를 `Float`으로 편다.
    ///
    /// 표준 `Float16`을 쓰지 않는 것은 그 타입이 x86_64 macOS에서 빠지기 때문이다.
    /// 되읽기는 CI를 포함해 어디서든 돌아야 하므로 비트 조작으로 직접 편다.
    static func float(fromHalf bits: UInt16) -> Float {
        let sign = UInt32(bits & 0x8000) << 16
        let exponent = Int((bits >> 10) & 0x1F)
        let mantissa = UInt32(bits & 0x03FF)

        // half의 지수 바이어스는 15, float은 127이다.
        let bias = 127 - 15

        if exponent == 0x1F {
            // Inf / NaN — 지수를 전부 1로 채우고 가수는 그대로 옮긴다.
            return Float(bitPattern: sign | 0x7F80_0000 | (mantissa << 13))
        }
        if exponent == 0 {
            if mantissa == 0 { return Float(bitPattern: sign) }   // ±0
            // 서브노멀 — float에서는 정규수가 되므로, 암묵 1이 제자리에 올 때까지 밀어 올린다.
            var shiftedExponent = 0
            var shiftedMantissa = mantissa
            while shiftedMantissa & 0x0400 == 0 {
                shiftedMantissa <<= 1
                shiftedExponent -= 1
            }
            shiftedMantissa &= 0x03FF
            let biased = UInt32(shiftedExponent + 1 + bias)
            return Float(bitPattern: sign | (biased << 23) | (shiftedMantissa << 13))
        }
        return Float(bitPattern: sign | (UInt32(exponent + bias) << 23) | (mantissa << 13))
    }
}
