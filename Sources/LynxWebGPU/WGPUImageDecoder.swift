import Foundation
import CoreGraphics
import ImageIO
import LynxWebGPUCore

/// 디코딩이 끝난 이미지 — 명세의 `ImageBitmap` 자리다.
///
/// 웹의 `createImageBitmap()`이 돌려주는 것과 같은 역할이고, 같은 이유로 **텍스처가 아니다**:
/// 아직 GPU에 없고, 크기만 알면 되는 단계다. JS는 이 크기로 텍스처를 만든 뒤
/// `copyExternalImageToTexture`로 올린다.
public final class WGPUImageBitmapObject {
    public let width: Int
    public let height: Int
    /// RGBA8 픽셀 (행 간격 = `bytesPerRow`).
    public let pixels: Data
    public let bytesPerRow: Int
    /// 알파가 곱해진 색인가 (`createImageBitmap`의 `premultiplyAlpha`).
    public let premultiplied: Bool

    init(width: Int, height: Int, pixels: Data, bytesPerRow: Int, premultiplied: Bool) {
        self.width = width
        self.height = height
        self.pixels = pixels
        self.bytesPerRow = bytesPerRow
        self.premultiplied = premultiplied
    }
}

/// 인코딩된 이미지 바이트(PNG·JPEG·HEIC …)를 RGBA8 픽셀로 푼다.
///
/// 웹에서는 브라우저가 하는 일이다. Lynx에는 `ImageBitmap`도 `<img>`도 없으므로 여기서
/// ImageIO로 대신한다 — 앱이 JS에서 PNG를 손으로 푸는 것보다 수십 배 빠르고, HEIC처럼
/// JS 디코더가 아예 없는 형식도 열린다.
public enum WGPUImageDecoder {
    /// 디코딩 옵션은 백엔드와 무관한 값이라 `LynxWebGPUCore`에 있다 (`WebGPURuntime`의
    /// `decodeImage`가 이 타입을 받는다). 예전 이름을 그대로 쓸 수 있게 별칭을 남긴다.
    public typealias Options = WGPUImageDecodeOptions

    /// - Returns: RGBA8 이미지. 실패하면 `WGPUError.validation`을 던진다.
    public static func decode(_ data: Data, options: Options = Options()) throws -> WGPUImageBitmapObject {
        guard !data.isEmpty else {
            throw WGPUError.validation("이미지 데이터가 비어 있다")
        }
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw WGPUError.validation(
                "이미지를 디코딩하지 못했다 (\(data.count)B — 지원하지 않는 형식이거나 손상된 데이터)"
            )
        }
        let width = options.resize?.width ?? image.width
        let height = options.resize?.height ?? image.height
        guard width > 0, height > 0 else {
            throw WGPUError.validation("이미지 크기가 0이다 (\(width)x\(height))")
        }
        return try draw(image, width: width, height: height, options: options)
    }

    private static func draw(
        _ image: CGImage, width: Int, height: Int, options: Options
    ) throws -> WGPUImageBitmapObject {
        let bytesPerRow = width * 4
        var pixels = Data(count: bytesPerRow * height)
        // 텍스처는 rgba8unorm이므로 채널 순서를 R,G,B,A로 못 박는다 — CoreGraphics 기본에
        // 맡기면 기기/형식에 따라 BGRA가 나와 색이 뒤집힌다.
        //
        // 알파는 **항상 곱해서** 받는다. CGBitmapContext는 8비트 RGBA에서 곱하지 않은 알파를
        // 지원하지 않아 컨텍스트 생성 자체가 실패한다. 곱하지 않은 값이 필요하면 아래에서
        // 되돌린다 (브라우저의 `premultiplyAlpha: 'none'`도 같은 처지다).
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue

        let drawn: Bool = pixels.withUnsafeMutableBytes { raw -> Bool in
            guard let base = raw.baseAddress,
                  let context = CGContext(
                      data: base, width: width, height: height,
                      bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                      space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bitmapInfo
                  ) else { return false }
            // 비트맵 컨텍스트에 그대로 그리면 메모리 첫 행이 이미지의 **첫 행**이다
            // (웹과 같은 좌상단 기준). CTM을 뒤집는 것이 곧 flipY다.
            if options.flipY {
                context.translateBy(x: 0, y: CGFloat(height))
                context.scaleBy(x: 1, y: -1)
            }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else {
            throw WGPUError.outOfMemory("이미지 비트맵 컨텍스트를 만들지 못했다 (\(width)x\(height))")
        }
        if !options.premultiplyAlpha { unpremultiply(&pixels) }
        return WGPUImageBitmapObject(
            width: width, height: height, pixels: pixels,
            bytesPerRow: bytesPerRow, premultiplied: options.premultiplyAlpha
        )
    }

    /// 곱해진 알파를 되돌린다 (`premultiplyAlpha: 'none'`).
    ///
    /// 알파가 낮을수록 남아 있는 정보가 적어 **완전히 복원되지는 않는다** — 브라우저도 같다.
    /// 정확한 값이 필요하면 애초에 곱하지 않은 원본을 `writeTexture`로 올릴 것.
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
