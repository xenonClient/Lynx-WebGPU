import Foundation

/// A 2D block of pixels read back from the GPU.
///
/// **It does not hand back bare bytes.** Returning a readback as plain `Data` makes the caller
/// implicitly assume "this is RGBA8", and on an `rgba16float` surface it then reads silently wrong
/// values with no error. So format, size and row stride travel with the values — everything needed
/// to interpret them is here.
///
/// Use `rgba(x:y:)` when you want channel values. To handle the format yourself, read `data`
/// directly but **take the row stride from `bytesPerRow`** (it may differ from
/// `width * bytesPerPixel`).
public struct WGPUPixelReadback: Sendable, Equatable {
    /// Pixel bytes. One row is `bytesPerRow` bytes, and there are `height` rows.
    public let data: Data
    /// What format `data` is in. Never interpret the bytes without consulting this.
    public let format: WGPUTextureFormat
    public let width: Int
    public let height: Int
    /// Row stride. With padding it exceeds `width * format.bytesPerPixel`.
    public let bytesPerRow: Int

    public init(data: Data, format: WGPUTextureFormat, width: Int, height: Int, bytesPerRow: Int) {
        self.data = data
        self.format = format
        self.width = width
        self.height = height
        self.bytesPerRow = bytesPerRow
    }

    /// Bytes per pixel.
    public var bytesPerPixel: Int { format.bytesPerPixel }

    /// Expands one pixel into RGBA floats.
    ///
    /// Missing channels are filled with 0 for RGB and 1 for A (the same rule WebGPU uses when a
    /// shader reads a texture). `bgra8unorm` and friends are reordered into RGBA.
    ///
    /// **No color-space conversion happens.** Even a `-srgb` format is merely normalized to 0...1
    /// as stored, so a caller who needs linear values converts them. Float formats are not
    /// normalized at all, so **values above 1.0 and negative values survive intact** — that is what
    /// makes verifying HDR results possible.
    ///
    /// - Throws: `WGPUError.validation` if the coordinate is out of range, or if the format cannot
    ///           be expanded into channels (packed or integer formats). Interpret `data` directly
    ///           for those.
    public func rgba(x: Int, y: Int) throws -> SIMD4<Float> {
        guard x >= 0, y >= 0, x < width, y < height else {
            throw WGPUError.validation("pixel (\(x), \(y)) is outside the \(width)×\(height) bounds")
        }
        guard let layout = Self.channelLayout(of: format) else {
            throw WGPUError.validation(
                "\(format.rawValue) cannot be expanded by rgba(x:y:) — interpret data directly"
            )
        }
        let stride = bytesPerPixel
        let offset = y * bytesPerRow + x * stride
        guard offset >= 0, offset + stride <= data.count else {
            throw WGPUError.validation(
                "bytes for pixel (\(x), \(y)) fall outside the buffer (\(data.count)B) — check bytesPerRow (\(bytesPerRow))"
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
                    // In snorm both -128 and -127 are -1.0 (the spec's clamping rule).
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

    // MARK: - Channel layout

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

    /// Handles only formats whose channels sit in uniform-width slots.
    /// Formats that pack bits across channel boundaries (`rgb10a2unorm`, `rgb10a2uint`,
    /// `rg11b10ufloat`, `rgb9e5ufloat`) and integer formats return nil — expanding those into
    /// normalized floats would distort the values, so we leave them to the caller.
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

    /// Expands IEEE 754 binary16 bits into a `Float`.
    ///
    /// We avoid the standard `Float16` because that type is unavailable on x86_64 macOS. Readback
    /// has to work everywhere, CI included, so we expand it by hand with bit manipulation.
    static func float(fromHalf bits: UInt16) -> Float {
        let sign = UInt32(bits & 0x8000) << 16
        let exponent = Int((bits >> 10) & 0x1F)
        let mantissa = UInt32(bits & 0x03FF)

        // half's exponent bias is 15; float's is 127.
        let bias = 127 - 15

        if exponent == 0x1F {
            // Inf / NaN — fill the exponent with ones and carry the mantissa across.
            return Float(bitPattern: sign | 0x7F80_0000 | (mantissa << 13))
        }
        if exponent == 0 {
            if mantissa == 0 { return Float(bitPattern: sign) }   // ±0
            // Subnormal — normal in float, so shift up until the implicit 1 lands in place.
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
