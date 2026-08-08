import ProjectDescription

// Dawn 백엔드 연동 검증 프로젝트 — **루트 워크스페이스(`LynxWebGPUDemo.xcworkspace`)의
// 별도 프로젝트**다 (`Workspace.swift`). 데모 로직과 완전히 분리돼 있고, 루트에서
// `mise exec -- tuist generate` 한 번이면 스킴 셋(WebGPUDemo · DawnCheck · DawnDemo)이 다 나온다.
//
// `docs/extra/DAWN-BACKEND-REVIEW.md` §3-6이 그리는 "별도 저장소 Lynx-WebGPU-Dawn"의
// 시제품이다 — [Dawn-xcFramework](https://github.com/xenonClient/Dawn-xcFramework)의
// 프리빌트 바이너리 위에 `WebGPURuntime`을 구현하고, 같은 적합성 스위트를 돌려
// Metal 런타임과 계약이 같은지 잰다. Dawn.xcframework는 iOS 슬라이스만 있으므로
// (macOS 없음) 검증은 시뮬레이터에서 돈다:
//   arch -arm64 xcodebuild -workspace LynxWebGPUDemo.xcworkspace -scheme DawnCheck \
//     -destination 'platform=iOS Simulator,name=iPhone 17,OS=26.2' \
//     -derivedDataPath .derivedData-cli test
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
        // Lynx는 **앱 타깃(DawnDemo)이 고른다** — 데모와 같은 규칙 (docs/LYNX-INTEGRATION.md §1).
        .remote(
            url: "https://github.com/xenonClient/Lynx-XCFramework",
            requirement: .exact("4.0.0")
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
        // 오프스크린 적합성 검증 (28검사 — docs/TESTING.md §2-1).
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
        // Lynx 연동 레이어 — 데모와 **같은 브리지 소스**를 컴파일한다. 백엔드 교체 계약의
        // 반쪽이 여기다: 이 타깃이 무수정으로 컴파일되면 브리지는 정말로 백엔드를 모른다.
        .target(
            name: "LynxWebGPUBridge",
            destinations: .iOS,
            product: .staticFramework,
            bundleId: "org.lynxwebgpu.dawncheck.bridge",
            deploymentTargets: .iOS("17.0"),
            sources: ["../../Sources/LynxWebGPUBridge/**"],
            dependencies: [
                .package(product: "LynxWebGPUCore"),
                .package(product: "Lynx"),
            ]
        ),
        // 실제 연동 실증 앱 — 데모의 씬 목록·런처·.lynx.bundle을 **그대로** 쓰되
        // 런타임만 Dawn이다. `LynxWebGPU`(Metal 엔진)는 링크하지 않는다.
        .target(
            name: "DawnDemo",
            destinations: .iOS,
            product: .app,
            bundleId: "org.lynxwebgpu.dawndemo",
            deploymentTargets: .iOS("17.0"),
            infoPlist: .extendingDefault(with: [
                "UILaunchScreen": ["UIColorName": "", "UIImageName": ""],
                "UIApplicationSceneManifest": [
                    "UIApplicationSupportsMultipleScenes": false,
                    "UISceneConfigurations": [
                        "UIWindowSceneSessionRoleApplication": [
                            [
                                "UISceneConfigurationName": "Default Configuration",
                                "UISceneDelegateClassName": "$(PRODUCT_MODULE_NAME).SceneDelegate",
                            ]
                        ]
                    ],
                ],
            ]),
            sources: [
                "App/**",
                "Sources/**",   // DawnWebGPURuntime — 적합성 타깃과 같은 소스
                // 씬 목록과 런처는 데모의 것을 그대로 쓴다 (Foundation/UIKit뿐이다) —
                // DemoViewController만 이 앱의 것(App/)이 대신한다.
                "../WebGPUDemo/Sources/DemoScene.swift",
                "../WebGPUDemo/Sources/LauncherViewController.swift",
            ],
            // 데모 번들(.lynx.bundle)도 그대로 — JS 무변경 계약의 증거다.
            resources: ["../WebGPUDemo/Resources/**"],
            dependencies: [
                .target(name: "LynxWebGPUBridge"),
                .package(product: "Dawn"),
                .package(product: "LynxWebGPUCore"),
                .package(product: "Lynx"),
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
        ),
        .scheme(
            name: "DawnDemo",
            shared: true,
            buildAction: .buildAction(targets: ["DawnDemo"]),
            runAction: .runAction(configuration: "Debug")
        ),
    ]
)
