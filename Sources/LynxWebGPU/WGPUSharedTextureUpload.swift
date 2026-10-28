import Foundation
import CoreGraphics
import ImageIO
import Metal
import LynxWebGPUCore

/// Decodes and uploads one shared texture (`importSharedTexture`) — the `make` behind
/// `WGPUSharedTextureCache`.
///
/// The path keeps the memory peak near the **target** size rather than the source size:
/// - ImageIO's thumbnail API decodes straight to (at least) the target size — a large source asked for
///   at half size never exists at full size in memory.
/// - CoreGraphics draws the pixels once, into a shared-storage `MTLBuffer` — no decoded bitmap object,
///   no crop copy, no staging pool.
/// - A blit on a dedicated per-device queue moves them into a private texture, and the call **waits**
///   for it. The upload does not ride any context's batch: that would commit only when that context
///   submits, and every backend has its own staging pool.
///
/// The wait is what makes the texture safe to read from every context's queue right away — Metal tracks
/// hazards within a queue, not between queues, so a read committed on another queue straight after an
/// asynchronous upload could race it. It runs on the cache's background queue, never the JS thread.
///
/// Color: drawn into device RGB with no color management, as `createImageBitmap` (`WGPUImageDecoder`).
/// Alpha comes out **premultiplied** — CoreGraphics draws 8-bit RGBA no other way, and keeping it saves a
/// lossy division pass over the whole image.
enum WGPUSharedTextureUpload {
    /// - Returns: an `rgba8unorm`, `shaderRead`, private texture at `request.targetSize(…)`, its upload completed.
    static func makeTexture(from data: Data, request: WGPUSharedTextureRequest, device: MTLDevice) throws -> MTLTexture {
        guard !data.isEmpty else {
            throw WGPUError.validation("importSharedTexture: '\(request.name)' is empty")
        }
        guard let source = CGImageSourceCreateWithData(data as CFData, [kCGImageSourceShouldCache: false] as CFDictionary),
              CGImageSourceGetCount(source) > 0,
              let sourceSize = pixelSize(of: source) else {
            throw WGPUError.validation(
                "importSharedTexture could not decode '\(request.name)' (\(data.count)B — unsupported format or corrupt data)"
            )
        }
        // A requested axis is checked before anything is derived from it: bounded by the limit, the
        // derived axis stays inside `Int` (`targetSize`), and the error names the axis the caller gave.
        let limit = WGPUDeviceCapability.maxTextureDimension2D(device)
        for (field, value) in [("resizeWidth", request.resizeWidth), ("resizeHeight", request.resizeHeight)] {
            if let value, value < 1 || value > limit {
                throw WGPUError.validation(
                    "importSharedTexture: \(field) must be between 1 and maxTextureDimension2D (\(limit)) — got \(value)"
                )
            }
        }
        let target = request.targetSize(sourceWidth: sourceSize.width, sourceHeight: sourceSize.height)
        guard target.width <= limit, target.height <= limit else {
            throw WGPUError.validation(
                "importSharedTexture: \(target.width)x\(target.height) exceeds maxTextureDimension2D (\(limit)) "
                    + "— pass resizeWidth/resizeHeight"
            )
        }
        guard let image = decode(source, sourceSize: sourceSize, target: target) else {
            throw WGPUError.validation("importSharedTexture could not decode '\(request.name)'")
        }

        let bytesPerRow = target.width * 4
        guard let staging = device.makeBuffer(length: bytesPerRow * target.height, options: .storageModeShared) else {
            throw WGPUError.outOfMemory("importSharedTexture: no memory for a \(target.width)x\(target.height) upload")
        }
        try draw(image, into: staging, width: target.width, height: target.height, bytesPerRow: bytesPerRow)

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: target.width, height: target.height, mipmapped: false
        )
        descriptor.usage = .shaderRead
        descriptor.storageMode = .private
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw WGPUError.outOfMemory("importSharedTexture: texture creation failed (\(target.width)x\(target.height))")
        }
        texture.label = request.label ?? "shared:\(request.key)"

        guard let queue = uploadQueue(for: device),
              let commandBuffer = queue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw WGPUError.backend("importSharedTexture could not encode its upload")
        }
        blit.copy(
            from: staging, sourceOffset: 0,
            sourceBytesPerRow: bytesPerRow, sourceBytesPerImage: bytesPerRow * target.height,
            sourceSize: MTLSize(width: target.width, height: target.height, depth: 1),
            to: texture, destinationSlice: 0, destinationLevel: 0,
            destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
        )
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else {
            throw WGPUError.backend(
                "importSharedTexture upload failed: \(commandBuffer.error.map { "\($0)" } ?? "status \(commandBuffer.status.rawValue)")"
            )
        }
        return texture
    }

    /// The source's pixel size from its properties — read without decoding.
    private static func pixelSize(of source: CGImageSource) -> (width: Int, height: Int)? {
        guard let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
              let height = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue,
              width > 0, height > 0 else { return nil }
        return (width, height)
    }

    /// The smallest decode that still covers the target on **both** axes — smaller would make the draw
    /// upscale. The thumbnail API bounds the long side only, so that side is scaled by the larger of the
    /// two axis ratios; the draw then fits it to the exact target. A target at or above the source size
    /// decodes the source as is.
    private static func decode(
        _ source: CGImageSource, sourceSize: (width: Int, height: Int), target: (width: Int, height: Int)
    ) -> CGImage? {
        let scale = max(
            Double(target.width) / Double(sourceSize.width),
            Double(target.height) / Double(sourceSize.height)
        )
        guard scale < 1 else {
            return CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCache: false] as CFDictionary)
        }
        let options: [CFString: Any] = [
            // An embedded thumbnail (EXIF, HEIC) is far smaller than any real target — always decode.
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: Int((Double(max(sourceSize.width, sourceSize.height)) * scale).rounded(.up)),
            // No EXIF orientation, the same as `createImageBitmap`.
            kCGImageSourceCreateThumbnailWithTransform: false,
            kCGImageSourceShouldCacheImmediately: true,
        ]
        return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
    }

    /// Draws `image` scaled to exactly `width`×`height` into the buffer, first row first (top-left origin,
    /// as on the web). The channel order is pinned to R,G,B,A for `rgba8unorm`.
    private static func draw(_ image: CGImage, into buffer: MTLBuffer, width: Int, height: Int, bytesPerRow: Int) throws {
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        guard let context = CGContext(
            data: buffer.contents(), width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bitmapInfo
        ) else {
            throw WGPUError.outOfMemory("importSharedTexture could not create its bitmap context (\(width)x\(height))")
        }
        // The buffer's memory is not cleared — copy the source over it instead of blending onto whatever is there.
        context.setBlendMode(.copy)
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    }

    // MARK: - The upload queue

    private static let queueLock = NSLock()
    private static var queues: [ObjectIdentifier: MTLCommandQueue] = [:]

    /// One queue per device for shared uploads, kept for the life of the process.
    private static func uploadQueue(for device: MTLDevice) -> MTLCommandQueue? {
        queueLock.lock()
        defer { queueLock.unlock() }
        let id = ObjectIdentifier(device)
        if let queue = queues[id] { return queue }
        guard let queue = device.makeCommandQueue() else { return nil }
        queue.label = "LynxWebGPU shared texture uploads"
        queues[id] = queue
        return queue
    }
}
