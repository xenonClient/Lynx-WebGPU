#!/usr/bin/env swift
//
//  extract-hdr-asset.swift
//
//  Converts an HDR HEIC (Apple's gain map form) into a single binary the demo can upload to the GPU as-is.
//
//      swift Projects/WebGPUDemo/Tools/extract-hdr-asset.swift \
//          ~/IMG_2243.HEIC Projects/WebGPUDemo/Resources/hdr-sample.bin
//
//  Why at build time rather than runtime:
//  - What the demo verifies is the `rgba16float` path, not an image decoder. Inserting a decoder at
//    runtime blurs what is under test, and unpacking 4032×3024 in JS is merely slow.
//  - Extracting only pixels **strips every capture metadata field** — EXIF, GPS, MakerApple.
//    The original photo carries capture coordinates, so keeping only the output in the repository is safer.
//
//  Output layout (little endian):
//
//      0   "LWGH"        magic
//      4   version u32   = 1
//      8   baseWidth  u32
//      12  baseHeight u32
//      16  gainWidth  u32
//      20  gainHeight u32
//      24  headroom   f32   maximum multiple over SDR white (e.g. 4.89)
//      28  baseOffset u32
//      32  baseLength u32
//      36  gainOffset u32
//      40  gainLength u32
//      44  reserved   u32
//      48  base pixels (RGBA8, sRGB, no alpha → fixed at 255)
//      …   gain map pixels (R8, grayscale)
//

import Foundation
import ImageIO
import CoreGraphics
import UniformTypeIdentifiers

// MARK: - Configuration

/// Target width in pixels for the base image. The gain map is set to half of it (matching the source ratio).
let baseTargetWidth = 1024

let headerSize = 48
let magic: [UInt8] = Array("LWGH".utf8)

// MARK: - Arguments

let arguments = CommandLine.arguments
guard arguments.count >= 3 else {
    FileHandle.standardError.write(Data("usage: extract-hdr-asset.swift <input.HEIC> <output.bin>\n".utf8))
    exit(2)
}
let inputURL = URL(fileURLWithPath: arguments[1])
let outputURL = URL(fileURLWithPath: arguments[2])

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("error: \(message)\n".utf8))
    exit(1)
}

// MARK: - Opening the source

guard let source = CGImageSourceCreateWithURL(inputURL as CFURL, nil) else {
    fail("could not open the image: \(inputURL.path)")
}
guard let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any] else {
    fail("could not read the image properties")
}

// The maximum multiple used when applying the gain map to the SDR base. Without it this is not an HDR photo.
guard let headroom = properties["Headroom"] as? Double else {
    fail("no headroom — this photo carries no HDR gain map")
}

// Rotation metadata is not applied to the pixels. The demo asset accepts upright photos only.
if let orientation = properties["Orientation"] as? Int, orientation != 1 {
    FileHandle.standardError.write(Data(
        "warning: Orientation=\(orientation) but no rotation is applied. Prefer an upright photo.\n".utf8
    ))
}

// MARK: - Base image

guard let baseImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
    fail("could not decode the base image")
}

let baseWidth = baseTargetWidth
let baseHeight = Int((Double(baseTargetWidth) * Double(baseImage.height) / Double(baseImage.width)).rounded())

/// Even a Display P3 source is converted to sRGB here.
/// The canvas output is sRGB-based, so passing P3 values through skews the color, and color space
/// handling is separate from the axis this scene verifies (HDR reconstruction).
guard let baseContext = CGContext(
    data: nil,
    width: baseWidth,
    height: baseHeight,
    bitsPerComponent: 8,
    bytesPerRow: baseWidth * 4,
    space: CGColorSpace(name: CGColorSpace.sRGB)!,
    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
) else {
    fail("could not create the base bitmap context")
}
baseContext.interpolationQuality = .high
baseContext.draw(baseImage, in: CGRect(x: 0, y: 0, width: baseWidth, height: baseHeight))

guard let basePointer = baseContext.data else { fail("could not obtain the base pixels") }
var basePixels = Data(bytes: basePointer, count: baseWidth * baseHeight * 4)

// `noneSkipLast` leaves the fourth byte undefined. It is filled with 255 so a shader reading alpha
// has no problem.
basePixels.withUnsafeMutableBytes { raw in
    guard let bytes = raw.bindMemory(to: UInt8.self).baseAddress else { return }
    for index in stride(from: 3, to: baseWidth * baseHeight * 4, by: 4) {
        bytes[index] = 255
    }
}

// MARK: - Gain map

guard let gainInfo = CGImageSourceCopyAuxiliaryDataInfoAtIndex(
    source, 0, kCGImageAuxiliaryDataTypeHDRGainMap
) as? [String: Any],
      let gainData = gainInfo[kCGImageAuxiliaryDataInfoData as String] as? Data,
      let description = gainInfo[kCGImageAuxiliaryDataInfoDataDescription as String] as? [String: Any],
      let sourceGainWidth = description["Width"] as? Int,
      let sourceGainHeight = description["Height"] as? Int,
      let sourceGainBytesPerRow = description["BytesPerRow"] as? Int
else {
    fail("could not find the HDR gain map auxiliary data")
}

// The auxiliary data's rows carry padding (2016 pixels with a BytesPerRow of 2048, say).
// Telling CGImage that value lets CoreGraphics skip it for us.
guard let gainProvider = CGDataProvider(data: gainData as CFData),
      let gainImage = CGImage(
        width: sourceGainWidth,
        height: sourceGainHeight,
        bitsPerComponent: 8,
        bitsPerPixel: 8,
        bytesPerRow: sourceGainBytesPerRow,
        space: CGColorSpaceCreateDeviceGray(),
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
        provider: gainProvider,
        decode: nil,
        shouldInterpolate: true,
        intent: .defaultIntent
      )
else {
    fail("could not build an image from the gain map")
}

let gainWidth = baseWidth / 2
let gainHeight = baseHeight / 2

guard let gainContext = CGContext(
    data: nil,
    width: gainWidth,
    height: gainHeight,
    bitsPerComponent: 8,
    bytesPerRow: gainWidth,
    space: CGColorSpaceCreateDeviceGray(),
    bitmapInfo: CGImageAlphaInfo.none.rawValue
) else {
    fail("could not create the gain map bitmap context")
}
gainContext.interpolationQuality = .high
gainContext.draw(gainImage, in: CGRect(x: 0, y: 0, width: gainWidth, height: gainHeight))

guard let gainPointer = gainContext.data else { fail("could not obtain the gain map pixels") }
let gainPixels = Data(bytes: gainPointer, count: gainWidth * gainHeight)

// MARK: - Packing

var output = Data()
output.append(contentsOf: magic)

func appendUInt32(_ value: Int) {
    var little = UInt32(value).littleEndian
    withUnsafeBytes(of: &little) { output.append(contentsOf: $0) }
}
func appendFloat32(_ value: Double) {
    var little = Float(value).bitPattern.littleEndian
    withUnsafeBytes(of: &little) { output.append(contentsOf: $0) }
}

let baseOffset = headerSize
let gainOffset = baseOffset + basePixels.count

appendUInt32(1)                 // version
appendUInt32(baseWidth)
appendUInt32(baseHeight)
appendUInt32(gainWidth)
appendUInt32(gainHeight)
appendFloat32(headroom)
appendUInt32(baseOffset)
appendUInt32(basePixels.count)
appendUInt32(gainOffset)
appendUInt32(gainPixels.count)
appendUInt32(0)                 // reserved

precondition(output.count == headerSize, "the header must be \(headerSize) bytes (currently \(output.count))")

output.append(basePixels)
output.append(gainPixels)

do {
    try output.write(to: outputURL)
} catch {
    fail("could not write the output: \(error)")
}

let megabytes = Double(output.count) / 1_048_576
print("""
    \(outputURL.lastPathComponent) created — \(String(format: "%.2f", megabytes)) MB
      base      \(baseWidth)×\(baseHeight) RGBA8  (\(basePixels.count) B)
      gain map  \(gainWidth)×\(gainHeight) R8     (\(gainPixels.count) B)
      headroom \(String(format: "%.5f", headroom))
      Metadata (GPS included) does not survive into the output, since only pixels are extracted.
    """)
