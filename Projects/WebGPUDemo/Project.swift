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
    packages: [
        .local(path: "../.."),
    ],
    settings: .settings(
        // Lynx 공식 가이드 요구사항 (docs/LYNX-INTEGRATION.md §1)
        base: ["ENABLE_USER_SCRIPT_SANDBOXING": "NO"].merging(signingSettings) { _, new in new },
        configurations: [.debug(name: "Debug"), .release(name: "Release")]
    ),
    targets: [
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
                .package(product: "LynxWebGPUBridge"),
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
