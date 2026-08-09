import ProjectDescription

// The Dawn backend integration verification project — **a separate project inside the root
// workspace (`LynxWebGPUDemo.xcworkspace`)** (`Workspace.swift`). It is fully separated from the
// demo logic, and one `mise exec -- tuist generate` at the root produces every scheme (WebGPUDemo · DawnCheck · DawnDemo).
//
// It is a prototype of the "separate Lynx-WebGPU-Dawn repository" sketched in
// `docs/extra/DAWN-BACKEND-REVIEW.md` §3-6 — it implements `WebGPURuntime` on top of the prebuilt
// binaries of [Dawn-xcFramework](https://github.com/xenonClient/Dawn-xcFramework) and runs the same
// conformance suite to measure whether its contract matches the Metal runtime's. Dawn.xcframework
// only has iOS slices (no macOS), so verification runs on the simulator:
//   arch -arm64 xcodebuild -workspace LynxWebGPUDemo.xcworkspace -scheme DawnCheck \
//     -destination 'platform=iOS Simulator,name=iPhone 17,OS=26.2' \
//     -derivedDataPath .derivedData-cli test
let project = Project(
    name: "DawnCheck",
    organizationName: "LynxWebGPU",
    packages: [
        // The root SPM package — only the contract (Core) and the means of proof (Conformance) are used. The Metal engine is not linked.
        .local(path: "../.."),
        // The Dawn prebuilt (product "Dawn", module "WebGPU" — the webgpu.h C API). BSD-3-Clause.
        .remote(
            url: "https://github.com/xenonClient/Dawn-xcFramework",
            requirement: .exact("20260731.171941.0")
        ),
        // **The app target (DawnDemo) picks Lynx** — the same rule as the demo (docs/LYNX-INTEGRATION.md §1).
        .remote(
            url: "https://github.com/xenonClient/Lynx-XCFramework",
            requirement: .exact("4.0.0")
        ),
    ],
    settings: .settings(
        // This environment has no signing configuration, so the team must be stated for the simulator test bundle to be signed.
        base: [
            "DEVELOPMENT_TEAM": "TFLQDNW4Z9",
            "CODE_SIGN_STYLE": "Automatic",
        ],
        configurations: [.debug(name: "Debug"), .release(name: "Release")]
    ),
    targets: [
        // Offscreen conformance verification (29 checks — docs/TESTING.md §2-1).
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
        // The real integration demonstration app — it uses the demo's scene list, launcher and
        // .lynx.bundle **unchanged**, with only the runtime being Dawn. `LynxWebGPU` (the Metal engine) is not linked.
        //
        // The bridge is **the demo project's target, used as is** (a `.project` cross reference) — making
        // another target of the same name here would collide on output within one workspace, and proving
        // "the bridge does not know the backend" with **the same build artifact** is more accurate anyway.
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
                "Sources/**",   // DawnWebGPURuntime — the same sources as the conformance target
                // The scene list and launcher are the demo's, used as is (they are only Foundation/UIKit) —
                // only DemoViewController is replaced by this app's own (App/).
                "../WebGPUDemo/Sources/DemoScene.swift",
                "../WebGPUDemo/Sources/LauncherViewController.swift",
            ],
            // The demo bundles (.lynx.bundle) unchanged too — the evidence for the JS-unchanged contract.
            resources: ["../WebGPUDemo/Resources/**"],
            dependencies: [
                .project(target: "LynxWebGPUBridge", path: "../WebGPUDemo"),
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
