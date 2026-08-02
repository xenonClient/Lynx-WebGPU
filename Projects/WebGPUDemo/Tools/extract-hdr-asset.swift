#!/usr/bin/env swift
//
//  extract-hdr-asset.swift
//
//  HDR HEIC(Apple 게인맵 방식)를 데모가 그대로 GPU에 올릴 수 있는 단일 바이너리로 바꾼다.
//
//      swift Projects/WebGPUDemo/Tools/extract-hdr-asset.swift \
//          ~/IMG_2243.HEIC Projects/WebGPUDemo/Resources/hdr-sample.bin
//
//  왜 런타임이 아니라 빌드 시점에 하는가:
//  - 데모가 검증하려는 것은 `rgba16float` 처리 경로지 이미지 디코더가 아니다.
//    디코더를 런타임에 끼우면 검증 대상이 흐려지고, 4032×3024를 JS에서 푸는 건 느리기만 하다.
//  - 픽셀만 뽑아내므로 EXIF·GPS·MakerApple 같은 촬영 메타데이터가 **전부 떨어져 나간다**.
//    원본 사진에는 촬영 좌표가 들어 있어서, 산출물만 저장소에 두는 편이 안전하다.
//
//  출력 레이아웃 (리틀 엔디언):
//
//      0   "LWGH"        매직
//      4   version u32   = 1
//      8   baseWidth  u32
//      12  baseHeight u32
//      16  gainWidth  u32
//      20  gainHeight u32
//      24  headroom   f32   SDR 흰색 대비 최대 배율 (예: 4.89)
//      28  baseOffset u32
//      32  baseLength u32
//      36  gainOffset u32
//      40  gainLength u32
//      44  reserved   u32
//      48  베이스 픽셀 (RGBA8, sRGB, 알파 없음 → 255 고정)
//      …   게인맵 픽셀 (R8, 그레이스케일)
//

import Foundation
import ImageIO
import CoreGraphics
import UniformTypeIdentifiers

// MARK: - 설정

/// 베이스 이미지의 목표 가로 픽셀. 게인맵은 이것의 절반으로 맞춘다 (원본 비율과 같다).
let baseTargetWidth = 1024

let headerSize = 48
let magic: [UInt8] = Array("LWGH".utf8)

// MARK: - 인자

let arguments = CommandLine.arguments
guard arguments.count >= 3 else {
    FileHandle.standardError.write(Data("사용법: extract-hdr-asset.swift <입력.HEIC> <출력.bin>\n".utf8))
    exit(2)
}
let inputURL = URL(fileURLWithPath: arguments[1])
let outputURL = URL(fileURLWithPath: arguments[2])

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("오류: \(message)\n".utf8))
    exit(1)
}

// MARK: - 소스 열기

guard let source = CGImageSourceCreateWithURL(inputURL as CFURL, nil) else {
    fail("이미지를 열 수 없다: \(inputURL.path)")
}
guard let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any] else {
    fail("이미지 속성을 읽을 수 없다")
}

// 게인맵을 SDR 베이스에 곱할 때 쓰는 최대 배율. 이 값이 없으면 HDR 사진이 아니다.
guard let headroom = properties["Headroom"] as? Double else {
    fail("Headroom이 없다 — HDR 게인맵이 담긴 사진이 아니다")
}

// 회전 메타데이터는 픽셀에 반영하지 않는다. 데모 애셋은 정방향 사진만 받는다.
if let orientation = properties["Orientation"] as? Int, orientation != 1 {
    FileHandle.standardError.write(Data(
        "경고: Orientation=\(orientation) 이지만 회전을 적용하지 않는다. 정방향 사진을 쓰는 것이 좋다.\n".utf8
    ))
}

// MARK: - 베이스 이미지

guard let baseImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
    fail("베이스 이미지를 디코딩할 수 없다")
}

let baseWidth = baseTargetWidth
let baseHeight = Int((Double(baseTargetWidth) * Double(baseImage.height) / Double(baseImage.width)).rounded())

/// 원본이 Display P3여도 여기서 sRGB로 변환해 둔다.
/// 캔버스 출력이 sRGB 기준이라 P3 값을 그대로 흘리면 색이 틀어지고, 색공간 처리는
/// 이 씬이 검증하려는 축(HDR 재구성)과 별개다.
guard let baseContext = CGContext(
    data: nil,
    width: baseWidth,
    height: baseHeight,
    bitsPerComponent: 8,
    bytesPerRow: baseWidth * 4,
    space: CGColorSpace(name: CGColorSpace.sRGB)!,
    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
) else {
    fail("베이스 비트맵 컨텍스트를 만들 수 없다")
}
baseContext.interpolationQuality = .high
baseContext.draw(baseImage, in: CGRect(x: 0, y: 0, width: baseWidth, height: baseHeight))

guard let basePointer = baseContext.data else { fail("베이스 픽셀을 얻을 수 없다") }
var basePixels = Data(bytes: basePointer, count: baseWidth * baseHeight * 4)

// `noneSkipLast`는 네 번째 바이트를 정의하지 않은 채 남긴다. 셰이더가 알파를 읽어도
// 문제 없도록 255로 채워 둔다.
basePixels.withUnsafeMutableBytes { raw in
    guard let bytes = raw.bindMemory(to: UInt8.self).baseAddress else { return }
    for index in stride(from: 3, to: baseWidth * baseHeight * 4, by: 4) {
        bytes[index] = 255
    }
}

// MARK: - 게인맵

guard let gainInfo = CGImageSourceCopyAuxiliaryDataInfoAtIndex(
    source, 0, kCGImageAuxiliaryDataTypeHDRGainMap
) as? [String: Any],
      let gainData = gainInfo[kCGImageAuxiliaryDataInfoData as String] as? Data,
      let description = gainInfo[kCGImageAuxiliaryDataInfoDataDescription as String] as? [String: Any],
      let sourceGainWidth = description["Width"] as? Int,
      let sourceGainHeight = description["Height"] as? Int,
      let sourceGainBytesPerRow = description["BytesPerRow"] as? Int
else {
    fail("HDR 게인맵 보조 데이터를 찾을 수 없다")
}

// 보조 데이터의 행에는 패딩이 붙어 있다 (예: 2016픽셀인데 BytesPerRow는 2048).
// CGImage에 그 값을 그대로 알려 주면 CoreGraphics가 알아서 건너뛴다.
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
    fail("게인맵을 이미지로 만들 수 없다")
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
    fail("게인맵 비트맵 컨텍스트를 만들 수 없다")
}
gainContext.interpolationQuality = .high
gainContext.draw(gainImage, in: CGRect(x: 0, y: 0, width: gainWidth, height: gainHeight))

guard let gainPointer = gainContext.data else { fail("게인맵 픽셀을 얻을 수 없다") }
let gainPixels = Data(bytes: gainPointer, count: gainWidth * gainHeight)

// MARK: - 패킹

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

precondition(output.count == headerSize, "헤더 크기가 \(headerSize)바이트여야 한다 (지금 \(output.count))")

output.append(basePixels)
output.append(gainPixels)

do {
    try output.write(to: outputURL)
} catch {
    fail("출력을 쓸 수 없다: \(error)")
}

let megabytes = Double(output.count) / 1_048_576
print("""
    \(outputURL.lastPathComponent) 생성됨 — \(String(format: "%.2f", megabytes)) MB
      베이스   \(baseWidth)×\(baseHeight) RGBA8  (\(basePixels.count) B)
      게인맵   \(gainWidth)×\(gainHeight) R8     (\(gainPixels.count) B)
      headroom \(String(format: "%.5f", headroom))
      메타데이터(GPS 포함)는 픽셀만 추출하므로 산출물에 남지 않는다
    """)
