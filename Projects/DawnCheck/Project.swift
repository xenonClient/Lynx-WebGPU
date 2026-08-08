import ProjectDescription

// Dawn 백엔드 연동 검증 프로젝트.
//
// `docs/extra/DAWN-BACKEND-REVIEW.md` §3-6이 그리는 "별도 저장소 Lynx-WebGPU-Dawn"의
// 시제품이다 — [Dawn-xcFramework](https://github.com/xenonClient/Dawn-xcFramework)의
// 프리빌트 바이너리 위에 `WebGPURuntime`을 구현하고, 같은 적합성 스위트를 돌려
// Metal 런타임과 계약이 같은지 잰다.
//
// Dawn.xcframework는 iOS 슬라이스만 있으므로(macOS 없음) 검증은 **시뮬레이터의
// 유닛테스트**로 돈다:
//   mise exec -- tuist generate --path Projects/DawnCheck --no-open
//   arch -arm64 xcodebuild -workspace Projects/DawnCheck/DawnCheck.xcworkspace \
//     -scheme DawnCheck -destination 'platform=iOS Simulator,name=iPhone 17,OS=26.2' \
//     -derivedDataPath .derivedData-dawncheck test
let project = Project(
    name: "DawnCheck",
    organizationName: "LynxWebGPU",
    packages: [
        // 루트 SPM 패키지 — 계약(Core)과 증명 수단(Conformance)만 쓴다. Metal 엔진은 링크하지 않는다.
        .local(path: "../.."),
        // Dawn 프리빌트 (제품 "Dawn", 모듈 "WebGPU" — webgpu.h C API)
        .remote(
            url: "https://github.com/xenonClient/Dawn-xcFramework",
            requirement: .exact("20260731.171941.0")
        ),
    ],
    settings: .settings(
        // 이 환경은 서명이 구성돼 있지 않아 팀을 명시해야 시뮬레이터 테스트 번들이 서명된다.
        base: [
            "DEVELOPMENT_TEAM": "TFLQDNW4Z9",
            "CODE_SIGN_STYLE": "Automatic",
        ],
        configurations: [.debug(name: "Debug"), .release(name: "Release")]
    ),
    targets: [
        .target(
            name: "DawnCheckTests",
            destinations: .iOS,
            product: .unitTests,
            bundleId: "org.lynxwebgpu.dawncheck.tests",
            deploymentTargets: .iOS("17.0"),
            sources: ["Sources/**", "Tests/**"],
            dependencies: [
                .package(product: "Dawn"),
                .package(product: "LynxWebGPUCore"),
                .package(product: "LynxWebGPUConformance"),
            ]
        ),
    ],
    schemes: [
        .scheme(
            name: "DawnCheck",
            shared: true,
            buildAction: .buildAction(targets: ["DawnCheckTests"]),
            testAction: .targets(["DawnCheckTests"], configuration: "Debug"),
            runAction: .runAction(configuration: "Debug")
        )
    ]
)
