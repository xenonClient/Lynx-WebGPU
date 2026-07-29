// swift-tools-version: 6.0
import PackageDescription

// Swift 5 언어 모드를 유지한다. GPU 객체 그래프(MTLDevice/MTLBuffer/LynxUI …)는
// Sendable이 아닌 참조 타입을 JS 스레드와 메인 스레드가 공유하므로,
// 컴파일러의 격리 검사 대신 `WGPURegistry`의 명시적 락으로 동시성을 보장한다 (docs/ARCHITECTURE.md §7).
let swiftSettings: [SwiftSetting] = [.swiftLanguageMode(.v5)]

let package = Package(
    name: "LynxWebGPU",
    platforms: [
        .iOS(.v17),
        // macOS는 Lynx 없이 엔진/트랜스파일러만 빌드·테스트하는 개발 루프용이다
        // (`swift test` — docs/TESTING.md). Lynx 의존성은 iOS로 조건부 제한된다.
        .macOS(.v14),
    ],
    products: [
        // Lynx 없이도 쓸 수 있는 WebGPU 엔진 (Metal 백엔드 + WGSL 트랜스파일러).
        .library(name: "LynxWebGPU", targets: ["LynxWebGPU"]),
        // Lynx 연동 레이어 — NativeModule + <webgpu-canvas> 엘리먼트. iOS 전용.
        .library(name: "LynxWebGPUBridge", targets: ["LynxWebGPUBridge"]),
    ],
    dependencies: [
        .package(url: "https://github.com/xenonClient/Lynx-XCFramework", exact: "4.0.0"),
    ],
    targets: [
        .target(
            name: "LynxWebGPUCore",
            swiftSettings: swiftSettings
        ),
        .target(
            name: "LynxWebGPUShader",
            dependencies: ["LynxWebGPUCore"],
            swiftSettings: swiftSettings
        ),
        .target(
            name: "LynxWebGPU",
            dependencies: ["LynxWebGPUCore", "LynxWebGPUShader"],
            swiftSettings: swiftSettings
        ),
        .target(
            name: "LynxWebGPUBridge",
            dependencies: [
                "LynxWebGPU",
                .product(name: "Lynx", package: "Lynx-XCFramework", condition: .when(platforms: [.iOS])),
            ],
            swiftSettings: swiftSettings
        ),
        .testTarget(
            name: "LynxWebGPUCoreTests",
            dependencies: ["LynxWebGPUCore"],
            swiftSettings: swiftSettings
        ),
        .testTarget(
            name: "LynxWebGPUShaderTests",
            dependencies: ["LynxWebGPUShader"],
            swiftSettings: swiftSettings
        ),
        .testTarget(
            name: "LynxWebGPUTests",
            dependencies: ["LynxWebGPU"],
            swiftSettings: swiftSettings
        ),
    ]
)
