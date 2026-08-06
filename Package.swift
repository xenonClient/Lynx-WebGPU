// swift-tools-version: 6.0
import PackageDescription

// Swift 5 언어 모드를 유지한다. GPU 객체 그래프(MTLDevice/MTLBuffer/LynxUI …)는
// Sendable이 아닌 참조 타입을 JS 스레드와 메인 스레드가 공유하므로,
// 컴파일러의 격리 검사 대신 `WGPURegistry`의 명시적 락으로 동시성을 보장한다 (docs/ARCHITECTURE.md §7).
let swiftSettings: [SwiftSetting] = [.swiftLanguageMode(.v5)]

// 이 패키지는 **외부 의존성이 없다.**
//
// Lynx 연동 레이어(`Sources/LynxWebGPUBridge/`)는 일부러 SPM 타깃이 아니다. SPM 타깃의
// `canImport(Lynx)`는 **매니페스트가 선언한 의존성만** 보므로, 여기서 Lynx를 선언하지 않으면
// 그 타깃은 무엇을 하든 빈 모듈로 컴파일된다 — 붙여 두면 "제품을 추가했는데 API가 없다"는
// 함정이 될 뿐이다. 반대로 선언하면 Lynx의 버전과 배포처가 이 패키지에 박혀,
// 다른 버전이나 다른 배포 방식(CocoaPods·사내 배포본)을 쓰는 앱이 그대로 쓸 수 없다.
//
// 그래서 브리지는 **앱이 Lynx를 볼 수 있는 자리에서 컴파일하는 소스**로 제공한다.
// 연동 방법은 `docs/LYNX-INTEGRATION.md` §2에 있다.
let package = Package(
    name: "LynxWebGPU",
    platforms: [
        .iOS(.v17),
        // macOS는 Lynx 없이 엔진/트랜스파일러만 빌드·테스트하는 개발 루프용이다
        // (`swift test` — docs/TESTING.md).
        .macOS(.v14),
    ],
    products: [
        // Lynx 없이도 쓸 수 있는 WebGPU 엔진 (Metal 백엔드 + WGSL 트랜스파일러).
        .library(name: "LynxWebGPU", targets: ["LynxWebGPU"]),
        // 커맨드 스트림의 계약만 — `WebGPURuntime` 프로토콜, 디스크립터·커맨드 디코딩, 오류 모양.
        //
        // **다른 백엔드를 만들 때 링크하는 것이 이쪽이다.** Metal 엔진(`LynxWebGPU`) 없이
        // 브리지 + 자체 런타임만으로 앱을 구성할 수 있다 — GPU 코드가 하나도 들어 있지 않다.
        .library(name: "LynxWebGPUCore", targets: ["LynxWebGPUCore"]),
        // 어떤 `WebGPURuntime`이든 계약을 지키는지 재는 적합성 스위트.
        //
        // 테스트 타깃이 아니라 **라이브러리**인 이유: SPM의 테스트 타깃은 다른 패키지가 가져다
        // 쓸 수 없다. 저장소 밖에서 만든 런타임이 같은 스위트로 자신을 증명할 수 있어야 한다.
        .library(name: "LynxWebGPUConformance", targets: ["LynxWebGPUConformance"]),
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
            name: "LynxWebGPUConformance",
            dependencies: ["LynxWebGPUCore"],
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
            dependencies: ["LynxWebGPU", "LynxWebGPUConformance"],
            swiftSettings: swiftSettings
        ),
    ]
)
