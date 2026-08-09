// swift-tools-version: 6.0
import PackageDescription

// The Swift 5 language mode is kept. The GPU object graph (MTLDevice/MTLBuffer/LynxUI …) shares
// non-Sendable reference types between the JS thread and the main thread, so concurrency is
// guaranteed by `WGPURegistry`'s explicit locks rather than the compiler's isolation checking (docs/ARCHITECTURE.md §7).
let swiftSettings: [SwiftSetting] = [.swiftLanguageMode(.v5)]

// This package has **no external dependencies.**
//
// The Lynx integration layer (`Sources/LynxWebGPUBridge/`) is deliberately not an SPM target. An SPM
// target's `canImport(Lynx)` only sees **dependencies the manifest declares**, so with Lynx undeclared
// here that target would compile to an empty module no matter what it contained — shipping it would only
// be a trap of the "I added the product but there is no API" kind. Declaring it, conversely, pins Lynx's
// version and distribution into this package, so an app on a different version or a different distribution
// (CocoaPods, an in-house build) could not use it as is.
//
// So the bridge is provided as **sources compiled where the app can see Lynx**.
// The integration method is in `docs/LYNX-INTEGRATION.md` §2.
let package = Package(
    name: "LynxWebGPU",
    platforms: [
        .iOS(.v17),
        // macOS is for the development loop that builds and tests only the engine and transpiler, without Lynx
        // (`swift test` — docs/TESTING.md).
        .macOS(.v14),
    ],
    products: [
        // The WebGPU engine usable without Lynx (the Metal backend plus the WGSL transpiler).
        .library(name: "LynxWebGPU", targets: ["LynxWebGPU"]),
        // The command stream's contract alone — the `WebGPURuntime` protocol, descriptor and command decoding, error shapes.
        //
        // **This is what you link when building another backend.** An app can be assembled from the bridge
        // plus its own runtime, with no Metal engine (`LynxWebGPU`) — there is no GPU code in here at all.
        .library(name: "LynxWebGPUCore", targets: ["LynxWebGPUCore"]),
        // The conformance suite that measures whether any `WebGPURuntime` keeps the contract.
        //
        // Why it is a **library** rather than a test target: SPM test targets cannot be consumed by another
        // package. A runtime built outside this repository has to be able to prove itself with the same suite.
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
