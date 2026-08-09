import Foundation
import CoreGraphics
import ImageIO

/// A decoded image — this is the spec's `ImageBitmap`.
///
/// It plays the same role as what the web's `createImageBitmap()` returns, and for the same
/// reason it is **not a texture**: it is not on the GPU yet, and at this stage only the size
/// matters. JS creates a texture of that size and uploads it with `copyExternalImageToTexture`.
public final class WGPUImageBitmapObject {
    public let width: Int
    public let height: Int
    /// RGBA8 pixels (row stride = `bytesPerRow`).
    public let pixels: Data
    public let bytesPerRow: Int
    /// Whether the color has alpha premultiplied in (`createImageBitmap`'s `premultiplyAlpha`).
    public let premultiplied: Bool

    init(width: Int, height: Int, pixels: Data, bytesPerRow: Int, premultiplied: Bool) {
        self.width = width
        self.height = height
        self.pixels = pixels
        self.bytesPerRow = bytesPerRow
        self.premultiplied = premultiplied
    }
}

/// Expands encoded image bytes (PNG, JPEG, HEIC, …) into RGBA8 pixels.
///
/// On the web the browser does this. Lynx has neither `ImageBitmap` nor `<img>`, so ImageIO
/// stands in here — dozens of times faster than an app decoding PNG by hand in JS, and it opens
/// formats such as HEIC that have no JS decoder at all.
public enum WGPUImageDecoder {
    /// Decode options are backend-independent values, so they live in `LynxWebGPUCore`
    /// (`WebGPURuntime.decodeImage` takes this type). The alias keeps the old name working.
    public typealias Options = WGPUImageDecodeOptions

    /// - Returns: an RGBA8 image. Throws `WGPUError.validation` on failure.
    public static func decode(_ data: Data, options: Options = Options()) throws -> WGPUImageBitmapObject {
        guard !data.isEmpty else {
            throw WGPUError.validation("image data is empty")
        }
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw WGPUError.validation(
                "could not decode the image (\(data.count)B — unsupported format or corrupt data)"
            )
        }
        let width = options.resize?.width ?? image.width
        let height = options.resize?.height ?? image.height
        guard width > 0, height > 0 else {
            throw WGPUError.validation("image size is zero (\(width)x\(height))")
        }
        return try draw(image, width: width, height: height, options: options)
    }

    private static func draw(
        _ image: CGImage, width: Int, height: Int, options: Options
    ) throws -> WGPUImageBitmapObject {
        let bytesPerRow = width * 4
        var pixels = Data(count: bytesPerRow * height)
        // The texture is rgba8unorm, so we pin the channel order to R,G,B,A — left to the
        // CoreGraphics default it can come out BGRA depending on device and format, inverting colors.
        //
        // Alpha is **always** taken premultiplied. CGBitmapContext does not support unpremultiplied
        // alpha in 8-bit RGBA, so creating the context would fail outright. If unpremultiplied
        // values are needed we undo it below (the browser's `premultiplyAlpha: 'none'` is in the
        // same position).
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue

        let drawn: Bool = pixels.withUnsafeMutableBytes { raw -> Bool in
            guard let base = raw.baseAddress,
                  let context = CGContext(
                      data: base, width: width, height: height,
                      bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                      space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bitmapInfo
                  ) else { return false }
            // Drawing straight into the bitmap context puts the image's **first row** first in
            // memory (top-left origin, as on the web). Flipping the CTM is what flipY means.
            if options.flipY {
                context.translateBy(x: 0, y: CGFloat(height))
                context.scaleBy(x: 1, y: -1)
            }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else {
            throw WGPUError.outOfMemory("could not create the image bitmap context (\(width)x\(height))")
        }
        if !options.premultiplyAlpha { unpremultiply(&pixels) }
        return WGPUImageBitmapObject(
            width: width, height: height, pixels: pixels,
            bytesPerRow: bytesPerRow, premultiplied: options.premultiplyAlpha
        )
    }

    /// Undoes premultiplied alpha (`premultiplyAlpha: 'none'`).
    ///
    /// The lower the alpha the less information survives, so it is **not fully recoverable** — the
    /// browser is no different. When exact values matter, upload the unpremultiplied original with
    /// `writeTexture` in the first place.
    private static func unpremultiply(_ pixels: inout Data) {
        pixels.withUnsafeMutableBytes { raw in
            guard let base = raw.bindMemory(to: UInt8.self).baseAddress else { return }
            for offset in stride(from: 0, to: raw.count, by: 4) {
                let alpha = Int(base[offset + 3])
                guard alpha > 0, alpha < 255 else { continue }
                for channel in 0..<3 {
                    let value = (Int(base[offset + channel]) * 255 + alpha / 2) / alpha
                    base[offset + channel] = UInt8(min(255, value))
                }
            }
        }
    }
}
