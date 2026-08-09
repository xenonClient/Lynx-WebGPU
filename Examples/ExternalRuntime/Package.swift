// swift-tools-version: 6.0
import PackageDescription

// A verification fixture that mimics the view a backend built **outside** this repository has.
//
// It deliberately links only `LynxWebGPUCore` and `LynxWebGPUConformance` — an external runtime
// repository like Dawn's only really holds together if `WebGPURuntime` can be implemented and the
// conformance suite run without the Metal engine (`LynxWebGPU`). A build break here (a missing product,
// a type narrowed to internal, …) is exactly the problem an external implementer would hit.
//
// Verification commands (docs/TESTING.md):
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
                // A path dependency's package identifier is the **directory name**, not the manifest name.
                .product(name: "LynxWebGPUCore", package: "Lynx-WebGPU"),
                .product(name: "LynxWebGPUConformance", package: "Lynx-WebGPU"),
            ],
            // The same language mode as the root — this is also the setting recommended to external implementers
            // (the GPU object graph is not Sendable and is guarded by explicit locks).
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
