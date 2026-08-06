import ProjectDescription

// 실기기 설치용 서명. 환경변수로 팀을 주면 자동 서명이 켜진다:
//   TUIST_DEVELOPMENT_TEAM=XXXXXXXXXX mise exec -- tuist generate
// 미지정 시(시뮬레이터 빌드) 서명 설정을 넣지 않는다.
let signingSettings: SettingsDictionary = {
    let team = Environment.developmentTeam.getString(default: "")
    guard !team.isEmpty else { return [:] }
    return ["DEVELOPMENT_TEAM": .string(team), "CODE_SIGN_STYLE": "Automatic"]
}()

let project = Project(
    name: "WebGPUDemo",
    organizationName: "LynxWebGPU",
    // 저장소 루트의 SPM 패키지를 로컬 의존성으로 그대로 쓴다 — 데모가 항상 현재 소스를 링크한다.
    //
    // **Lynx는 이 데모가 직접 고른다.** 라이브러리 패키지에는 Lynx 의존성이 없다 —
    // 버전과 배포처를 앱이 정할 수 있게 하기 위해서다 (`docs/LYNX-INTEGRATION.md` §2).
    // 다른 버전을 시험하려면 아래 `requirement`만 바꾸면 된다.
    packages: [
        .local(path: "../.."),
        .remote(
            url: "https://github.com/xenonClient/Lynx-XCFramework",
            requirement: .exact("4.0.0")
        ),
    ],
    settings: .settings(
        // Lynx 공식 가이드 요구사항 (docs/LYNX-INTEGRATION.md §1)
        base: ["ENABLE_USER_SCRIPT_SANDBOXING": "NO"].merging(signingSettings) { _, new in new },
        configurations: [.debug(name: "Debug"), .release(name: "Release")]
    ),
    targets: [
        // Lynx 연동 레이어 — **저장소의 브리지 소스를 여기서 컴파일한다.**
        //
        // 이 타깃이 Lynx를 의존성으로 들고 있으므로 브리지 안의 `#if canImport(Lynx)`가 켜진다.
        // SPM 패키지 쪽에 두면 매니페스트가 Lynx 버전을 박아 버리므로 일부러 앱 쪽에 둔 것이다.
        // 실제 앱도 이 모양을 그대로 따라 하면 된다 (`docs/LYNX-INTEGRATION.md` §2-2).
        //
        // **의존성이 `LynxWebGPUCore`뿐인 것에 주목할 것** — 브리지는 `WebGPURuntime` 프로토콜만
        // 알고 GPU 백엔드를 모른다. 어느 엔진을 링크할지는 아래 앱 타깃이 정한다.
        .target(
            name: "LynxWebGPUBridge",
            destinations: .iOS,
            product: .staticFramework,
            bundleId: "org.lynxwebgpu.bridge",
            deploymentTargets: .iOS("17.0"),
            sources: ["../../Sources/LynxWebGPUBridge/**"],
            dependencies: [
                .package(product: "LynxWebGPUCore"),
                .package(product: "Lynx"),
            ]
        ),
        .target(
            name: "WebGPUDemo",
            destinations: .iOS,
            product: .app,
            bundleId: "org.lynxwebgpu.demo",
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
            sources: ["Sources/**"],
            // .lynx.bundle 산출물 (DemoSrc의 rspeedy 빌드 결과를 복사한 것)
            resources: ["Resources/**"],
            dependencies: [
                .target(name: "LynxWebGPUBridge"),
                // **런타임(백엔드)을 고르는 자리.** 기본 Metal 엔진을 쓰므로 여기서 링크한다 —
                // 다른 백엔드로 가려면 이 줄을 그 패키지로 바꾸고 `LynxWebGPUHost(runtime:)`에
                // 넘기는 객체만 바꾸면 된다 (브리지와 JS 번들은 그대로다).
                .package(product: "LynxWebGPU"),
                // 앱 코드도 Lynx를 직접 쓴다 (LynxView 호스팅 — AppDelegate/DemoViewController).
                .package(product: "Lynx"),
            ]
        ),
    ],
    schemes: [
        .scheme(
            name: "WebGPUDemo",
            shared: true,
            buildAction: .buildAction(targets: ["WebGPUDemo"]),
            runAction: .runAction(configuration: "Debug")
        )
    ]
)
