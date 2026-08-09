import ProjectDescription

// Signing — this environment has no local signing configuration, so the team must be **stated by
// default** for an Xcode build to work. Override it with an environment variable:
//   TUIST_DEVELOPMENT_TEAM=XXXXXXXXXX mise exec -- tuist generate
let signingSettings: SettingsDictionary = {
    let team = Environment.developmentTeam.getString(default: "TFLQDNW4Z9")
    guard !team.isEmpty else { return [:] }
    return ["DEVELOPMENT_TEAM": .string(team), "CODE_SIGN_STYLE": "Automatic"]
}()

let project = Project(
    name: "WebGPUDemo",
    organizationName: "LynxWebGPU",
    // The repository root's SPM package is used directly as a local dependency — the demo always links current source.
    //
    // **This demo picks Lynx itself.** The library package has no Lynx dependency —
    // so the app can decide the version and distribution (`docs/LYNX-INTEGRATION.md` §2).
    // To try another version, change only the `requirement` below.
    packages: [
        .local(path: "../.."),
        .remote(
            url: "https://github.com/xenonClient/Lynx-XCFramework",
            requirement: .exact("4.0.0")
        ),
    ],
    settings: .settings(
        // A requirement of the official Lynx guide (docs/LYNX-INTEGRATION.md §1)
        base: ["ENABLE_USER_SCRIPT_SANDBOXING": "NO"].merging(signingSettings) { _, new in new },
        configurations: [.debug(name: "Debug"), .release(name: "Release")]
    ),
    targets: [
        // The Lynx integration layer — **the repository's bridge sources are compiled here.**
        //
        // This target carries Lynx as a dependency, which is what turns on `#if canImport(Lynx)` inside the bridge.
        // Putting it in the SPM package would pin the Lynx version in the manifest, so it lives on the app side deliberately.
        // A real app can follow this shape exactly (`docs/LYNX-INTEGRATION.md` §2-2).
        //
        // **Note that its only dependency is `LynxWebGPUCore`** — the bridge knows only the
        // `WebGPURuntime` protocol, not the GPU backend. Which engine to link is decided by the app target below.
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
            // The .lynx.bundle outputs (copies of DemoSrc's rspeedy build results)
            resources: ["Resources/**"],
            dependencies: [
                .target(name: "LynxWebGPUBridge"),
                // **Where the runtime (backend) is chosen.** The default Metal engine is linked here —
                // to move to another backend, change this line to that package and change only the object
                // passed to `LynxWebGPUHost(runtime:)` (the bridge and the JS bundle stay as they are).
                .package(product: "LynxWebGPU"),
                // The app code uses Lynx directly too (LynxView hosting — AppDelegate/DemoViewController).
                .package(product: "Lynx"),
            ]
        ),
        // **iOS verification of the default runtime (Metal)** — runs the same suite as `DawnCheckTests`
        // at the same destination (docs/TESTING.md §2-1).
        //
        // `swift test`'s `ConformanceTests` runs the same 29 checks, but that is **macOS**.
        // Where it actually ships is iOS, and with no place measuring the default backend there,
        // only the experimental backend (Dawn) would get iOS verification — backwards.
        //
        // It links neither Lynx nor the app code — what it measures is **the runtime's contract** alone.
        .target(
            name: "WebGPUDemoTests",
            destinations: .iOS,
            product: .unitTests,
            bundleId: "org.lynxwebgpu.demo.tests",
            deploymentTargets: .iOS("17.0"),
            sources: ["Tests/**"],
            dependencies: [
                .package(product: "LynxWebGPU"),
                .package(product: "LynxWebGPUCore"),
                .package(product: "LynxWebGPUConformance"),
            ]
        ),
    ],
    schemes: [
        .scheme(
            name: "WebGPUDemo",
            shared: true,
            buildAction: .buildAction(targets: ["WebGPUDemo"]),
            // Attached so ⌘U works from the app scheme too — measure straight from Xcode with the demo open.
            testAction: .targets(["WebGPUDemoTests"], configuration: "Debug"),
            runAction: .runAction(configuration: "Debug")
        ),
        // A verification-only scheme — the counterpart of `DawnCheck`. It does not build the app, so it is fast from the CLI.
        .scheme(
            name: "WebGPUCheck",
            shared: true,
            buildAction: .buildAction(targets: ["WebGPUDemoTests"]),
            testAction: .targets(["WebGPUDemoTests"], configuration: "Debug"),
            runAction: .runAction(configuration: "Debug")
        ),
    ]
)
