import Foundation

/// A demo scene contained in the bundles. Each is bundled as `Resources/<rawValue>.lynx.bundle`.
///
/// They are matched 1:1 with the features the offscreen harness verifies automatically, so the same
/// things can also be **checked by eye** (see `Tests/LynxWebGPUTests/RenderPipelineTests.swift`).
enum DemoScene: String, CaseIterable {
    case triangle
    case cube
    case particles
    case texture
    case dynamic
    case blending
    case stencil
    case gpudriven
    case bundle
    case query
    case readback
    case constants
    case msl
    case interactive
    case wgsl
    case arraybuffer
    case bench
    case hdr
    case scrollpass
    case fog
    case three
    case spec
    case images
    case contracts
    case threelab

    /// Scenes shown full-screen — **presented modally** rather than in a navigation controller.
    ///
    /// Swipe-back (`interactivePopGestureRecognizer`) intercepts drags starting at the left edge, so a
    /// scene you grab and drag on the canvas competes with that gesture. A modal full-screen
    /// presentation has no such gesture, so touches reach Lynx intact (back is a button we attach).
    var coversFullScreen: Bool { self == .interactive || self == .hdr || self == .scrollpass || self == .fog }

    var title: String {
        switch self {
        case .triangle: return "Rotating triangle"
        case .cube: return "3D cube"
        case .particles: return "4096 particles"
        case .texture: return "Texture and sampler"
        case .dynamic: return "Dynamic texture"
        case .blending: return "Alpha blending"
        case .stencil: return "Stencil mask"
        case .gpudriven: return "GPU-driven rendering"
        case .bundle: return "Render bundles"
        case .query: return "Queries and error scopes"
        case .readback: return "Compute and readback"
        case .constants: return "Pipeline constants"
        case .msl: return "MSL escape hatch"
        case .interactive: return "Holographic card"
        case .wgsl: return "WGSL compatibility"
        case .arraybuffer: return "Binary bridging"
        case .bench: return "Bridge cost measurement"
        case .hdr: return "HDR gain map reconstruction"
        case .scrollpass: return "Scroll passthrough"
        case .fog: return "Condensation wipe"
        case .three: return "three.js renderer"
        case .spec: return "Spec surface checklist"
        case .images: return "Images and compressed textures"
        case .contracts: return "Contract checklist"
        case .threelab: return "three.js advanced combinations"
        }
    }

    /// The WebGPU path this scene actually exercises.
    var subtitle: String {
        switch self {
        case .triangle: return "Vertex buffer + uniform + a writeBuffer every frame"
        case .cube: return "Indexed draw + depth test + backface culling + MVP"
        case .particles: return "Compute shader + storage buffer + instancing + additive blending"
        case .texture: return "createTexture + writeTexture + sampler + textureSample"
        case .dynamic: return "A CPU plasma uploaded by writeTexture every frame — verifies queue-ordered uploads"
        case .blending: return "Premultiplied alpha compositing + overlapping translucent shapes"
        case .stencil: return "The stencil8-only format — the same triangle three times, split by stencil alone"
        case .gpudriven: return "Compute decides the count and indirect dispatch/draw read that buffer"
        case .bundle: return "120 draws recorded once — commands per frame shown in the HUD"
        case .query: return "Occlusion sample counts + timestamps + a pushErrorScope comparison"
        case .readback: return "Compute results read on the CPU with mapAsync and displayed"
        case .constants: return "One shader across several pipelines, changing only override values"
        case .msl: return "language: 'msl' — bypassing the transpiler with an explicit layout"
        case .interactive: return "Grab and tilt to make the foil flow — Lynx touch → 3D pose → shader"
        case .wgsl: return "arrayLength + external textures + untyped constant expressions"
        case .arraybuffer: return "A round trip both ways as ArrayBuffer — verifies Lynx value conversion"
        case .bench: return "base64 string vs ArrayBuffer — comparing encoding and submission cost"
        case .hdr: return "loadAsset + a gain map compute → an rgba16float intermediate → tone mapping"
        case .scrollpass: return "A canvas over a <scroll-view> — verifies passthrough-touches hit testing"
        case .fog: return "A wipeable fog over a GPU-drawn shelf — all inside the spec, so Dawn runs it unchanged"
        case .three: return "A 16-item checklist plus a rotating cube in ASTC textures over a decoded PNG background"
        case .spec: return "14 spec surfaces — debug markers, diagnostics, partial mapping — checked by value"
        case .images: return "ASTC and BC blocks plus PNG decoding, checked by the pixel colors read back"
        case .contracts: return "Buffer copy defaults and ranges, integer vec3 layout, bundle isolation, reverse format mapping"
        case .threelab: return "TSL procedural materials · shadow maps · compute particles · instancing · bloom"
        }
    }
}
