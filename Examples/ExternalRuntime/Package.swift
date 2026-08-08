// swift-tools-version: 6.0
import PackageDescription

// 저장소 **밖**에서 만든 백엔드가 보는 시야를 흉내 내는 검증 픽스처다.
//
// 일부러 `LynxWebGPUCore`와 `LynxWebGPUConformance`만 링크한다 — Metal 엔진(`LynxWebGPU`)
// 없이 `WebGPURuntime`을 구현하고 적합성 스위트를 돌릴 수 있어야, Dawn 같은 외부 런타임
// 저장소가 정말로 성립한다. 여기서 빌드가 깨지면 (product 누락·internal로 좁힌 타입 등)
// 그것이 곧 외부 구현자가 겪을 문제다.
//
// 검증 명령 (docs/TESTING.md):
//   swift build --package-path Examples/ExternalRuntime
//   swift run --package-path Examples/ExternalRuntime external-runtime-check
let package = Package(
    name: "ExternalRuntime",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "external-runtime-check",
            dependencies: [
                // 경로 의존성의 패키지 식별자는 매니페스트 이름이 아니라 **디렉터리 이름**이다.
                .product(name: "LynxWebGPUCore", package: "Lynx-WebGPU"),
                .product(name: "LynxWebGPUConformance", package: "Lynx-WebGPU"),
            ],
            // 루트와 같은 언어 모드 — 외부 구현자에게 권하는 설정도 이것이다
            // (GPU 객체 그래프는 Sendable이 아니라 명시적 락으로 지킨다).
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
