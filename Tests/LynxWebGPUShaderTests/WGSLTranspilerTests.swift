import XCTest
import LynxWebGPUCore
@testable import LynxWebGPUShader

/// The core paths of the WGSL → MSL transpiler.
///
/// Every case checks both (1) that the expected MSL fragment is present and (2) that it **passes the
/// real Metal compiler** — to catch regressions where the string matches but nothing compiles.
final class WGSLTranspilerTests: XCTestCase {
    // MARK: - Helpers

    private func translate(
        _ source: String,
        entryPoints: [String],
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws -> String {
        let module = try WGSLShaderModule(source: source)
        let groups = module.autoBindGroupLayouts(entryPoints: entryPoints)
        let bindings = try WGSLBindingAssigner.assign(groups: groups)
        let msl = try module.translateToMSL(entryPoints: entryPoints, bindings: bindings)
        MetalCompilerHarness.assertCompiles(msl, file: file, line: line)
        return msl
    }

    // MARK: - Triangle (vertex attributes + uniform + helper function)

    private static let triangleShader = """
    struct Uniforms {
        mvp: mat4x4<f32>,
        tint: vec4f,
    };
    @group(0) @binding(0) var<uniform> uniforms: Uniforms;

    struct VertexInput {
        @location(0) position: vec3f,
        @location(1) color: vec3f,
    };

    struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) color: vec3f,
    };

    fn apply_tint(c: vec3f) -> vec3f {
        return c * uniforms.tint.rgb;
    }

    @vertex
    fn vs_main(input: VertexInput) -> VertexOutput {
        var out: VertexOutput;
        out.position = uniforms.mvp * vec4f(input.position, 1.0);
        out.color = apply_tint(input.color);
        return out;
    }

    @fragment
    fn fs_main(input: VertexOutput) -> @location(0) vec4f {
        return vec4f(input.color, 1.0);
    }
    """

    func test_triangleShaderVertexAttributesAndUniformsTranslateToMSL() throws {
        let msl = try translate(Self.triangleShader, entryPoints: ["vs_main", "fs_main"])

        XCTAssertTrue(msl.contains("vertex wgpu_vs_main_out vs_main("), "no vertex entry point wrapper")
        XCTAssertTrue(msl.contains("fragment wgpu_fs_main_out fs_main("), "no fragment entry point wrapper")
        XCTAssertTrue(msl.contains("[[attribute(0)]]"), "no vertex attribute 0")
        XCTAssertTrue(msl.contains("[[attribute(1)]]"), "no vertex attribute 1")
        XCTAssertTrue(msl.contains("[[position]]"), "no position builtin")
        XCTAssertTrue(msl.contains("[[color(0)]]"), "no fragment output target")
        XCTAssertTrue(msl.contains("constant Uniforms& uniforms [[buffer(0)]]"), "no uniform binding")
    }

    func test_aUniformUsedByAHelperIsPassedAsAnArgument() throws {
        let msl = try translate(Self.triangleShader, entryPoints: ["vs_main", "fs_main"])

        // MSL has no mutable globals, so the helper must take the uniform as an argument.
        XCTAssertTrue(
            msl.contains("float3 apply_tint(float3 c, constant Uniforms& uniforms)"),
            "resources were not threaded into the helper function:\n\(msl)"
        )
        XCTAssertTrue(msl.contains("apply_tint(input.color, uniforms)"), "the call site does not pass the resource")
    }

    func test_reflectionReportsEntryPointsAndBindings() throws {
        let module = try WGSLShaderModule(source: Self.triangleShader)

        XCTAssertEqual(module.reflection.entryPoints.map(\.name).sorted(), ["fs_main", "vs_main"])
        XCTAssertEqual(module.reflection.entryPoint(named: "vs_main")?.stage, .vertex)
        XCTAssertEqual(module.reflection.entryPoint(named: "fs_main")?.stage, .fragment)

        let resource = try XCTUnwrap(module.reflection.resource(named: "uniforms"))
        XCTAssertEqual(resource.group, 0)
        XCTAssertEqual(resource.binding, 0)
        XCTAssertEqual(resource.slotKind, .buffer)

        // Only vs_main uses uniforms, so visibility is vertex alone (fs_main reads only interpolated values).
        XCTAssertEqual(module.reflection.visibility(of: "uniforms"), .vertex)
    }

    // MARK: - Struct layout (the vec3 difference between WGSL and MSL)

    func test_aScalarAfterVec3UsesAPackedVectorAndPaddingToMatchTheWGSLLayout() throws {
        let source = """
        struct Light {
            direction: vec3f,
            intensity: f32,
            color: vec3f,
        };
        @group(0) @binding(0) var<uniform> light: Light;

        @fragment
        fn fs_main() -> @location(0) vec4f {
            return vec4f(light.color * light.intensity, light.direction.x);
        }
        """
        let msl = try translate(source, entryPoints: ["fs_main"])

        // intensity (offset 12) follows direction (offset 0, size 12), so float3 (16B) does not line up.
        XCTAssertTrue(msl.contains("packed_float3 direction;"), "a packed vec3 is required:\n\(msl)")
        XCTAssertTrue(msl.contains("float intensity;"))
        // color has nothing after it, so a plain float3 plus trailing padding.
        XCTAssertTrue(msl.contains("float3 color;"))
    }

    /// Integer vec3 gets the same layout correction — `packed_int3`/`packed_uint3` exist in MSL, but
    /// checking only float would leave nobody knowing whether the integer path compiles.
    /// (That `translate` runs the **real Metal compiler** is the point of this test.)
    func test_anIntegerVec3IsAlsoPackedToMatchTheLayout() throws {
        let source = """
        struct Counts {
            offsets: vec3<i32>,
            total: i32,
            sizes: vec3<u32>,
            stride: u32,
        };
        @group(0) @binding(0) var<uniform> counts: Counts;

        @fragment
        fn fs_main() -> @location(0) vec4f {
            let value = f32(counts.offsets.x + counts.total) + f32(counts.sizes.y + counts.stride);
            return vec4f(value, 0.0, 0.0, 1.0);
        }
        """
        let msl = try translate(source, entryPoints: ["fs_main"])

        XCTAssertTrue(msl.contains("packed_int3 offsets;"), "the integer vec3 was not packed:\n\(msl)")
        XCTAssertTrue(msl.contains("packed_uint3 sizes;"), "the unsigned vec3 was not packed:\n\(msl)")
    }

    func test_structLayoutMatchesWGSLOffsets() throws {
        let source = """
        struct S {
            a: vec3f,
            b: f32,
            c: mat4x4<f32>,
        };
        @group(0) @binding(0) var<uniform> s: S;
        @fragment fn fs() -> @location(0) vec4f { return s.c[0] * s.b + vec4f(s.a, 1.0); }
        """
        let module = try WGSLParser.parse(source)
        let structure = try XCTUnwrap(module.structNamed("S"))
        let placement = WGSLLayout.layout(of: structure, module: module, uniform: true)

        XCTAssertEqual(placement.members.map(\.offset), [0, 12, 16])
        XCTAssertEqual(placement.size, 80)
        XCTAssertEqual(placement.align, 16)

        _ = try translate(source, entryPoints: ["fs"])
    }

    // MARK: - Compute / storage buffers

    func test_computeShadersAndStorageBuffersTranslate() throws {
        let source = """
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            let index = id.x;
            output[index] = input[index] * 2.0;
        }
        """
        let msl = try translate(source, entryPoints: ["main"])

        // `main` cannot be a function name in MSL, so it is renamed during emission.
        // The name the runtime passes to makeFunction(name:) comes from mslFunctionName(for:).
        let mslName = WGSLShaderModule.mslFunctionName(for: "main")
        XCTAssertEqual(mslName, "wgpu_fn_main")
        XCTAssertTrue(msl.contains("kernel void \(mslName)("), "the compute entry point did not come out as a kernel:\n\(msl)")
        XCTAssertTrue(msl.contains("const device float* input [[buffer(0)]]"), "read-only storage buffer")
        XCTAssertTrue(msl.contains("device float* output [[buffer(1)]]"), "read-write storage buffer")
        XCTAssertTrue(msl.contains("[[thread_position_in_grid]]"), "the global_invocation_id builtin")

        let module = try WGSLShaderModule(source: source)
        let size = try XCTUnwrap(module.workgroupSize(of: "main"))
        XCTAssertEqual(size.x, 64)
        XCTAssertEqual(size.y, 1)
        XCTAssertEqual(size.z, 1)
    }

    // MARK: - Textures / samplers

    func test_textureSamplingBecomesAnMSLMethodCall() throws {
        let source = """
        @group(0) @binding(0) var tex: texture_2d<f32>;
        @group(0) @binding(1) var samp: sampler;

        @fragment
        fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
            let base = textureSample(tex, samp, uv);
            let lod = textureSampleLevel(tex, samp, uv, 0.0);
            let texel = textureLoad(tex, vec2i(0, 0), 0);
            let size = textureDimensions(tex);
            return base + lod + texel + vec4f(f32(size.x), 0.0, 0.0, 0.0);
        }
        """
        let msl = try translate(source, entryPoints: ["fs_main"])

        XCTAssertTrue(msl.contains("tex.sample(samp, uv)"))
        XCTAssertTrue(msl.contains("tex.sample(samp, uv, level(0.0))"))
        XCTAssertTrue(msl.contains("tex.read(uint2("))
        XCTAssertTrue(msl.contains("tex.get_width("))
        XCTAssertTrue(msl.contains("texture2d<float> tex [[texture(0)]]"))
        XCTAssertTrue(msl.contains("sampler samp [[sampler(0)]]"))
    }

    func test_aStorageTextureWriteBecomesAWriteCall() throws {
        let source = """
        @group(0) @binding(0) var target: texture_storage_2d<rgba8unorm, write>;

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            textureStore(target, vec2i(i32(id.x), i32(id.y)), vec4f(1.0, 0.0, 0.0, 1.0));
        }
        """
        let msl = try translate(source, entryPoints: ["main"])

        XCTAssertTrue(msl.contains("texture2d<float, access::write> target [[texture(0)]]"))
        XCTAssertTrue(msl.contains("target.write("))
    }

    // MARK: - Control flow

    func test_everyControlFlowConstructTranslates() throws {
        let source = """
        fn classify(v: i32) -> i32 {
            switch v {
                case 0, 1: {
                    return 10;
                }
                case 2: {
                    return 20;
                }
                default: {
                    return -1;
                }
            }
        }

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            var total = 0;
            for (var i = 0; i < 8; i = i + 1) {
                if (i % 2 == 0) {
                    total = total + classify(i);
                } else if (i > 5) {
                    total = total - 1;
                } else {
                    continue;
                }
            }
            var n = 0;
            while (n < 4) {
                n++;
            }
            loop {
                n = n - 1;
                if (n <= 0) {
                    break;
                }
            }
        }
        """
        let msl = try translate(source, entryPoints: ["main"])

        XCTAssertTrue(msl.contains("switch ("))
        XCTAssertTrue(msl.contains("break;"))
        XCTAssertTrue(msl.contains("for ("))
        XCTAssertTrue(msl.contains("while (true)"))
        XCTAssertTrue(msl.contains("continue;"))
    }

    func test_workgroupVariablesAreDeclaredAsEntryPointLocalThreadgroup() throws {
        let source = """
        var<workgroup> scratch: array<f32, 64>;
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;

        @compute @workgroup_size(64)
        fn main(@builtin(local_invocation_id) local: vec3u,
                @builtin(global_invocation_id) global_id: vec3u) {
            scratch[local.x] = data[global_id.x];
            workgroupBarrier();
            data[global_id.x] = scratch[63u - local.x];
        }
        """
        let msl = try translate(source, entryPoints: ["main"])

        XCTAssertTrue(msl.contains("threadgroup array<float, 64> scratch;"), "no workgroup variable declaration:\n\(msl)")
        XCTAssertTrue(msl.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"))
    }

    // MARK: - Error handling

    func test_requestingAMissingEntryPointIsAnError() throws {
        let module = try WGSLShaderModule(source: Self.triangleShader)
        XCTAssertThrowsError(try module.requireEntryPoint("missing", stage: .vertex)) { error in
            XCTAssertEqual((error as? WGPUError)?.kind, .validation)
        }
    }

    func test_aDifferingStageIsAnError() throws {
        let module = try WGSLShaderModule(source: Self.triangleShader)
        XCTAssertThrowsError(try module.requireEntryPoint("fs_main", stage: .vertex))
    }

    func test_anUnsupportedBuiltinIsRejectedExplicitly() throws {
        let source = """
        @group(0) @binding(0) var<storage, read> data: array<f32>;
        @compute @workgroup_size(1)
        fn main() {
            let parts = modf(data[0]);
        }
        """
        let module = try WGSLShaderModule(source: source)
        let bindings = try WGSLBindingAssigner.assign(groups: module.autoBindGroupLayouts(entryPoints: ["main"]))
        XCTAssertThrowsError(try module.translateToMSL(entryPoints: ["main"], bindings: bindings)) { error in
            XCTAssertEqual((error as? WGPUError)?.kind, .unsupported)
        }
    }

    func test_aSyntaxErrorIsReportedWithItsLineNumber() throws {
        let source = """
        @vertex
        fn vs() -> @builtin(position) vec4f {
            return vec4f(1.0 1.0, 1.0, 1.0);
        }
        """
        XCTAssertThrowsError(try WGSLShaderModule(source: source)) { error in
            let message = (error as? WGPUError)?.message ?? ""
            XCTAssertTrue(message.contains("line 3"), "no line number: \(message)")
        }
    }

    // MARK: - Name collisions / literals / pipeline constants

    func test_identifiersCollidingWithMSLKeywordsChangeAtDeclarationAndEveryUse() throws {
        // `texture`, `sampler`, `device` and `char` are common names in graphics shaders but MSL keywords.
        let source = """
        struct Glyph {
            device: f32,
            char: f32,
        };
        @group(0) @binding(0) var<uniform> texture: Glyph;

        @fragment
        fn fs_main() -> @location(0) vec4f {
            let sampler = texture.device + texture.char;
            return vec4f(sampler, 0.0, 0.0, 1.0);
        }
        """
        let msl = try translate(source, entryPoints: ["fs_main"])

        XCTAssertTrue(msl.contains("wgpu_id_texture"), "the global name did not change:\n\(msl)")
        XCTAssertTrue(msl.contains("wgpu_id_device"), "the struct member name did not change")
        XCTAssertTrue(msl.contains("wgpu_id_char"))
        XCTAssertTrue(msl.contains("wgpu_id_sampler"), "the local variable name did not change")
        // A mismatch between declaration and use is caught by the Metal compile (the translate helper verifies it).
    }

    func test_floatingPointRemainderIsRoutedToFmod() throws {
        let source = """
        @fragment
        fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
            let f = uv.x % 2.0;
            let i = i32(uv.y) % 3;
            return vec4f(f, f32(i), 0.0, 1.0);
        }
        """
        let msl = try translate(source, entryPoints: ["fs_main"])
        XCTAssertTrue(msl.contains("wgpu_mod("), "% was not routed through the helper:\n\(msl)")
    }

    func test_aVectorConstructorWithNoComponentTypeInfersFromItsArguments() throws {
        let source = """
        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            let inferred = vec2(id.x, 4u);            // with a type in the arguments it infers exactly
            let scaled: vec2u = id.xy * vec2(4, 1);   // an AbstractInt constant expression → freezes at the context (u32)
            let asFloat: vec3f = vec3(1);             // the same constant expression becomes f32 in an f32 context
            let component = vec3(1, 2, 3).y;          // in component-access position it settles on a concrete type
            let combined = f32(scaled.x) + asFloat.y + component;
        }
        """
        let msl = try translate(source, entryPoints: ["main"])
        XCTAssertTrue(msl.contains("wgpu_vec2(id.x, 4u)"), "with a type present, template deduction:\n\(msl)")
        XCTAssertTrue(msl.contains("wgpu_aint2(4, 1)"), "an AbstractInt constant expression becomes a proxy:\n\(msl)")
        XCTAssertTrue(msl.contains("wgpu_aint3(1)"), "a single argument becomes a proxy too")
        // A proxy has no component access — in that position it settles on an f32 vector.
        XCTAssertTrue(msl.contains("float3(1, 2, 3).y"), "a swizzle target settles on a concrete type:\n\(msl)")
    }

    func test_pipelineConstantsArePlantedIntoTheMSL() throws {
        let source = """
        override scale: f32 = 1.0;
        override count: u32;

        @group(0) @binding(0) var<storage, read> data: array<f32, count>;

        @compute @workgroup_size(1)
        fn main() {
            let x = data[0] * scale;
        }
        """
        let module = try WGSLShaderModule(source: source)
        let bindings = try WGSLBindingAssigner.assign(groups: module.autoBindGroupLayouts(entryPoints: ["main"]))

        // Supplying no value fails while telling you what to supply.
        XCTAssertThrowsError(try module.translateToMSL(entryPoints: ["main"], bindings: bindings)) { error in
            let message = (error as? WGPUError)?.message ?? ""
            XCTAssertTrue(message.contains("count"), "it must say which constant is missing: \(message)")
            XCTAssertTrue(message.contains("constants"))
        }

        let msl = try module.translateToMSL(
            entryPoints: ["main"], bindings: bindings, constants: ["count": 8, "scale": 2.5]
        )
        // Even an array length depending on an override resolves.
        XCTAssertTrue(msl.contains("array<float, 8>"), "the override array length was not applied:\n\(msl)")
        XCTAssertTrue(msl.contains("2.5"))
        MetalCompilerHarness.assertCompiles(msl)
    }

    func test_enableDeclarationsAreIgnored() throws {
        let source = """
        enable f16;
        diagnostic(off, derivative_uniformity);

        @fragment
        fn fs_main() -> @location(0) vec4f {
            return vec4f(1.0);
        }
        """
        _ = try translate(source, entryPoints: ["fs_main"])
    }

    // MARK: - Reachability (a shared module several shaders split)

    func test_functionsTheEntryPointNeverCallsAreNotEmitted() throws {
        // Several shaders sharing one `common.wgsl` is a common arrangement (webgpu-samples' cornell).
        // Emitting functions this entry point never uses makes one of them look for a resource absent
        // from the `layout: "auto"` bind group, failing the whole translation — a path that really broke.
        let source = """
        struct Quad { color: vec4f };
        @group(0) @binding(0) var<uniform> uniforms: vec4f;
        @group(0) @binding(1) var<storage> quads: array<Quad>;

        // A helper this entry point does not use — another shader uses it.
        fn unused_helper() -> u32 {
            return arrayLength(&quads);
        }

        @fragment
        fn fs_main() -> @location(0) vec4f {
            return uniforms;
        }
        """
        let msl = try translate(source, entryPoints: ["fs_main"])

        XCTAssertFalse(msl.contains("unused_helper"), "an unused function was emitted:\n\(msl)")
        XCTAssertTrue(msl.contains("fs_main"), "the entry point must be present")
    }

    func test_transitivelyReachableFunctionsAreEmitted() throws {
        // Reachability is **transitive** — it must follow entry point → outer → inner.
        let source = """
        fn inner(x: f32) -> f32 { return x * 2.0; }
        fn outer(x: f32) -> f32 { return inner(x) + 1.0; }
        fn orphan(x: f32) -> f32 { return x; }

        @fragment
        fn fs_main() -> @location(0) vec4f {
            return vec4f(outer(0.5));
        }
        """
        let msl = try translate(source, entryPoints: ["fs_main"])

        XCTAssertTrue(msl.contains("float outer("), "the directly called function is missing:\n\(msl)")
        XCTAssertTrue(msl.contains("float inner("), "the transitively reachable function is missing:\n\(msl)")
        XCTAssertFalse(msl.contains("orphan"), "a function nobody calls was emitted")
    }

    func test_eachEntryPointEmitsOnlyTheFunctionsItNeeds() throws {
        let source = """
        fn only_vertex() -> f32 { return 1.0; }
        fn only_fragment() -> f32 { return 2.0; }

        @vertex
        fn vs_main() -> @builtin(position) vec4f {
            return vec4f(only_vertex());
        }

        @fragment
        fn fs_main() -> @location(0) vec4f {
            return vec4f(only_fragment());
        }
        """
        let vertexOnly = try translate(source, entryPoints: ["vs_main"])
        XCTAssertTrue(vertexOnly.contains("only_vertex"))
        XCTAssertFalse(vertexOnly.contains("only_fragment"), "a function belonging to another entry point came along")

        // Requesting both yields both.
        let both = try translate(source, entryPoints: ["vs_main", "fs_main"])
        XCTAssertTrue(both.contains("only_vertex"))
        XCTAssertTrue(both.contains("only_fragment"))
    }

    // MARK: - Global shadowing (an everyday pattern in machine-generated shaders — Three.js nodeVar0)

    func test_aGlobalShadowedByALocalIsNotInjected() throws {
        // The pattern Three.js's node system produces: the same name created at module scope and in a function.
        // helper uses only the local v, so it must not take the module-scope v as an argument — taking it
        // collides with the `float4 v{}` local declaration as a redefinition.
        let source = """
        var<private> v : vec4<f32>;

        fn helper(c : vec4<f32>) -> vec4<f32> {
            var v : vec4<f32>;
            v = c;
            return v;
        }

        @fragment
        fn main() -> @location(0) vec4<f32> {
            v = vec4<f32>(1.0);
            return helper(v);
        }
        """
        let msl = try translate(source, entryPoints: ["main"])
        XCTAssertTrue(msl.contains("float4 helper(float4 c)"), "a shadowed global was injected:\n\(msl)")
    }

    func test_aGlobalShadowedByAParameterIsNotInjected() throws {
        let source = """
        var<private> tint : vec4<f32>;

        fn helper(tint : vec4<f32>) -> vec4<f32> {
            return tint * 2.0;
        }

        @fragment
        fn main() -> @location(0) vec4<f32> {
            tint = vec4<f32>(0.5);
            return helper(tint);
        }
        """
        let msl = try translate(source, entryPoints: ["main"])
        XCTAssertTrue(msl.contains("float4 helper(float4 tint)"), "a global shadowed by a parameter was injected:\n\(msl)")
    }

    func test_aLocalDeclaredAfterUsingASameNamedGlobalIsRenamed() throws {
        // WGSL has point-of-declaration scope, so v before the declaration is the global and after it the local.
        // With the global injected as an argument, a local of the same name is a C++ redefinition and needs renaming.
        let source = """
        var<private> v : f32;

        fn helper() -> f32 {
            let before = v;
            var v : f32 = 3.0;
            v = v + before;
            return v;
        }

        @fragment
        fn main() -> @location(0) vec4<f32> {
            v = 1.0;
            return vec4<f32>(helper());
        }
        """
        let msl = try translate(source, entryPoints: ["main"])
        XCTAssertTrue(msl.contains("float wgpu_shadow_v = 3.0"), "the local declaration was not renamed:\n\(msl)")
        XCTAssertTrue(msl.contains("const auto before = v"), "a reference before the declaration must see the global (the injected argument):\n\(msl)")
    }

    func test_aGlobalShadowedLocallyIsStillThreadedDownTheCallGraph() throws {
        // outer does not use the global v directly (a local shadows it), but inner does, so
        // outer must **only forward** the global v — passing the local would be silently wrong.
        let source = """
        var<private> v : f32;

        fn inner() -> f32 {
            return v;
        }

        fn outer() -> f32 {
            var v : f32 = 100.0;
            return inner() + v * 0.0;
        }

        @fragment
        fn main() -> @location(0) vec4<f32> {
            v = 1.0;
            return vec4<f32>(outer());
        }
        """
        let msl = try translate(source, entryPoints: ["main"])
        // The local inside outer is renamed, and the inner call passes the original name (the injected argument = the global).
        XCTAssertTrue(msl.contains("float wgpu_shadow_v = 100.0"), "a local colliding with the forwarding injection was not renamed:\n\(msl)")
        XCTAssertTrue(msl.contains("inner(v)"), "inner must receive the global (the injected argument):\n\(msl)")
    }

    func test_shadowingInANestedBlockEndsWithTheBlock() throws {
        // If a rename of a local inside a block leaked out, the assignment to v outside would refer to a
        // name never declared (wgpu_shadow_v) and the compile would break — translate catches it.
        let source = """
        var<private> v : f32;

        fn helper() -> f32 {
            if (v > 0.0) {
                var v : f32 = 2.0;
                v = v * 2.0;
            }
            v = 5.0;
            return v;
        }

        @fragment
        fn main() -> @location(0) vec4<f32> {
            v = 1.0;
            return vec4<f32>(helper());
        }
        """
        _ = try translate(source, entryPoints: ["main"])
    }

    // MARK: - Binding assignment

    func test_bindingAssignmentIsDeterministicInGroupThenBindingOrder() throws {
        let source = """
        @group(0) @binding(0) var<uniform> a: vec4f;
        @group(0) @binding(2) var tex: texture_2d<f32>;
        @group(1) @binding(0) var<uniform> b: vec4f;
        @group(1) @binding(1) var samp: sampler;

        @fragment
        fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
            return a + b + textureSample(tex, samp, uv);
        }
        """
        let module = try WGSLShaderModule(source: source)
        let bindings = try WGSLBindingAssigner.assign(groups: module.autoBindGroupLayouts(entryPoints: ["fs"]))

        // Buffers, textures and samplers use independent index spaces.
        XCTAssertEqual(bindings.index(group: 0, binding: 0), 0)   // buffer 0
        XCTAssertEqual(bindings.index(group: 0, binding: 2), 0)   // texture 0
        XCTAssertEqual(bindings.index(group: 1, binding: 0), 1)   // buffer 1
        XCTAssertEqual(bindings.index(group: 1, binding: 1), 0)   // sampler 0

        _ = try translate(source, entryPoints: ["fs"])
    }

    func test_vertexBufferIndicesAreAssignedDownwardFromTheTopOfTheTable() {
        XCTAssertEqual(WGSLMetalLimits.vertexBufferIndex(slot: 0), 30)
        XCTAssertEqual(WGSLMetalLimits.vertexBufferIndex(slot: 1), 29)
        // It must not collide with the bind group buffer cap.
        XCTAssertLessThan(
            WGSLMetalLimits.maxBindGroupBuffers,
            WGSLMetalLimits.vertexBufferIndex(slot: WGSLMetalLimits.maxVertexBufferSlots - 1) + 1
        )
        // The size table index collides with neither side.
        XCTAssertEqual(WGSLMetalLimits.bufferSizesIndex, 22)
        XCTAssertLessThan(
            WGSLMetalLimits.bufferSizesIndex,
            WGSLMetalLimits.vertexBufferIndex(slot: WGSLMetalLimits.maxVertexBufferSlots - 1)
        )
    }

    // MARK: - arrayLength

    func test_aRuntimeSizedArrayLengthTranslatesIntoASizeTableLookup() throws {
        let source = """
        @group(0) @binding(0) var<storage, read> data: array<f32>;
        @group(0) @binding(1) var<storage, read_write> out: array<u32>;

        @compute @workgroup_size(1)
        fn cs(@builtin(global_invocation_id) id: vec3u) {
            out[0] = arrayLength(&data);
        }
        """
        let msl = try translate(source, entryPoints: ["cs"])

        // The size table arrives at the reserved index.
        XCTAssertTrue(
            msl.contains("constant uint* wgpu_buffer_sizes [[buffer(\(WGSLMetalLimits.bufferSizesIndex))]]"),
            msl
        )
        // length = binding byte count / element size. data is buffer index 0.
        XCTAssertTrue(msl.contains("(wgpu_buffer_sizes[0] / uint(sizeof(float)))"), msl)
    }

    func test_aTrailingRuntimeArraySubtractsThePrecedingMembersSize() throws {
        let source = """
        struct Particles {
            count: u32,
            items: array<vec4f>,
        }
        @group(0) @binding(0) var<storage, read_write> particles: Particles;

        @compute @workgroup_size(1)
        fn cs() {
            particles.count = arrayLength(&particles.items);
        }
        """
        let msl = try translate(source, entryPoints: ["cs"])

        // items is 16-byte aligned, so it starts at offset 16 (count 4B + 12B padding).
        XCTAssertTrue(msl.contains("((wgpu_buffer_sizes[0] - 16u) / uint(sizeof(float4)))"), msl)
    }

    func test_withoutArrayLengthTheSizeTableIsNotPassed() throws {
        let msl = try translate(Self.triangleShader, entryPoints: ["vs_main", "fs_main"])
        XCTAssertFalse(msl.contains("wgpu_buffer_sizes"), msl)
    }

    // MARK: - Compile-time constants

    func test_aConstDeclaredInsideAFunctionCanBeAnArraySize() throws {
        let source = """
        @fragment
        fn fs() -> @location(0) vec4f {
            const maxLayers = 4u;
            var layers: array<vec4f, maxLayers>;
            for (var i = 0u; i < maxLayers; i++) {
                layers[i] = vec4f(f32(i));
            }
            return layers[0];
        }
        """
        let msl = try translate(source, entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("array<float4, 4>"), msl)
    }

    func test_aLocalConstWithDifferingValuesPerFunctionIsNotUsedAsAnArraySize() throws {
        // Settling an array size has no function context, so it must refuse when the value is not unique.
        let source = """
        fn a() -> f32 {
            const n = 4;
            var xs: array<f32, n>;
            return xs[0];
        }
        fn b() -> f32 {
            const n = 8;
            var ys: array<f32, n>;
            return ys[0];
        }
        @fragment fn fs() -> @location(0) vec4f { return vec4f(a() + b()); }
        """
        let module = try WGSLShaderModule(source: source)
        let groups = module.autoBindGroupLayouts(entryPoints: ["fs"])
        let bindings = try WGSLBindingAssigner.assign(groups: groups)
        XCTAssertThrowsError(try module.translateToMSL(entryPoints: ["fs"], bindings: bindings))
    }

    // MARK: - External textures

    func test_anExternalTextureTranslatesToEdgeClampedSampling() throws {
        let source = """
        @group(0) @binding(0) var s: sampler;
        @group(0) @binding(1) var frame: texture_external;

        @fragment
        fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
            return textureSampleBaseClampToEdge(frame, s, uv);
        }
        """
        let msl = try translate(source, entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("wgpu_sample_base_clamp("), msl)
        // texture_external lowers to a sampleable 2D texture.
        XCTAssertTrue(msl.contains("texture2d<float>"), msl)
    }
}
