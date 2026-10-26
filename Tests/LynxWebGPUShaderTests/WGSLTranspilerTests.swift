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

        XCTAssertTrue(msl.contains("vertex wgpu_ep_vs_main_out vs_main("), "no vertex entry point wrapper")
        XCTAssertTrue(msl.contains("fragment wgpu_ep_fs_main_out fs_main("), "no fragment entry point wrapper")
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

    /// A struct value constructor must skip the padding members the layout inserts — positional
    /// aggregate init would push the next argument into `char wgpu_padN[…]` ("cannot initialize an
    /// array element of type 'char'"). Found by the ocean demo's `WaveSample(height, disp)`.
    func test_aStructConstructorSkipsInsertedPaddingMembers() throws {
        let source = """
        struct WaveSample {
            height: f32,
            disp: vec2f,
        };

        fn make(h: f32) -> WaveSample {
            return WaveSample(h, vec2f(h, h * 2.0));
        }

        @fragment
        fn fs_main() -> @location(0) vec4f {
            let s = make(0.25);
            let zero = WaveSample();
            return vec4f(s.height + zero.height, s.disp, 1.0);
        }
        """
        let msl = try translate(source, entryPoints: ["fs_main"])

        XCTAssertTrue(msl.contains("WaveSample{h, {}, "), "the padding slot must be zero-filled:\n\(msl)")
        XCTAssertTrue(msl.contains("WaveSample{}"), "zero-value construction stays a plain empty init:\n\(msl)")
    }

    /// An alias measures as the type it names — through any number of alias hops. The calculator used
    /// to read an alias of a non-struct type (and a two-hop chain) as 4 bytes, so every member after
    /// it sat at the wrong offset and the MSL struct disagreed with the bytes JS wrote.
    func test_aliasesAreLaidOutAsTheTypeTheyName() throws {
        func placement(_ source: String, uniform: Bool = false) throws -> WGSLLayout.StructPlacement {
            let module = try WGSLParser.parse(source)
            return WGSLLayout.layout(of: try XCTUnwrap(module.structNamed("S")), module: module, uniform: uniform)
        }

        let matrix = try placement("alias M = mat4x4f; struct S { m: M, t: f32 }", uniform: true)
        XCTAssertEqual(matrix.members.map(\.offset), [0, 64])
        XCTAssertEqual(matrix.size, 80)

        let chain = try placement("""
        struct Inner { v: vec4f }
        alias I1 = Inner;
        alias I2 = I1;
        struct S { x: f32, i: I2 }
        """)
        XCTAssertEqual(chain.members.map(\.offset), [0, 16])
        XCTAssertEqual(chain.size, 32)

        let array = try placement("alias A4 = array<f32, 4>; struct S { a: A4, b: vec4f }")
        XCTAssertEqual(array.members.map(\.offset), [0, 16])
        XCTAssertEqual(array.size, 32)

        // An aliased vec3 is still a vec3 — packed when the next member starts at byte 12.
        let vector = try placement("alias V = vec3f; struct S { p: V, w: f32 }")
        XCTAssertEqual(vector.members.map(\.offset), [0, 12])
        XCTAssertEqual(vector.size, 16)
        XCTAssertEqual(vector.members.first?.needsPackedVector, true)
    }

    /// The same shaders end to end: the emitted structs match the WGSL offsets (no stray padding),
    /// the buffers' minimum binding sizes count the aliased types, and the MSL compiles.
    func test_aliasedMembersEmitAtTheirWGSLOffsets() throws {
        let source = """
        struct Inner { v: vec4f }
        alias I1 = Inner;
        alias I2 = I1;
        alias M = mat4x4f;
        alias A4 = array<f32, 4>;
        alias V = vec3f;
        struct Params { m: M, t: f32 }
        struct Nested { x: f32, i: I2 }
        struct Packed { a: A4, b: vec4f, p: V, w: f32 }
        @group(0) @binding(0) var<uniform> params: Params;
        @group(0) @binding(1) var<storage, read> nested: Nested;
        @group(0) @binding(2) var<storage, read_write> packed: Packed;
        @group(0) @binding(3) var<uniform> matrix: M;

        @compute @workgroup_size(1)
        fn main() {
            packed.b = params.m[0] * params.t + nested.i.v + matrix[1] + vec4f(nested.x);
            packed.a[3] = packed.p.x + packed.w;
        }
        """
        let module = try WGSLShaderModule(source: source)
        XCTAssertEqual(module.reflection.minimumBindingSizes(usedBy: ["main"]), [0: [0: 80, 1: 32, 2: 48, 3: 64]])

        let msl = try translate(source, entryPoints: ["main"])
        XCTAssertTrue(msl.contains("struct alignas(16) Params {\n    float4x4 m;\n    float t;\n    char wgpu_pad0[12];\n};"), msl)
        XCTAssertTrue(msl.contains("struct alignas(16) Nested {\n    float x;\n    char wgpu_pad0[12];\n    Inner i;\n};"), msl)
        XCTAssertTrue(msl.contains(
            "struct alignas(16) Packed {\n    array<float, 4> a;\n    float4 b;\n    packed_float3 p;\n    float w;\n};"
        ), msl)
    }

    /// An alias may name a struct, but aliases are emitted before the struct definitions (a member may
    /// be spelled with one) — without a forward declaration Metal says "unknown type name 'S'".
    func test_anAliasOfAStructCompilesThroughAForwardDeclaration() throws {
        let msl = try translate("""
        struct S { x: f32 }
        alias T = S;
        alias Pair = array<T, 2>;
        @group(0) @binding(0) var<uniform> s: T;

        @fragment
        fn fs() -> @location(0) vec4f {
            var pair: Pair;
            pair[1] = s;
            return vec4f(pair[1].x);
        }
        """, entryPoints: ["fs"])

        let forward = try XCTUnwrap(msl.range(of: "struct S;"), msl)
        let alias = try XCTUnwrap(msl.range(of: "using T = S;"), msl)
        let definition = try XCTUnwrap(msl.range(of: #"struct alignas\(\d+\) S \{"#, options: .regularExpression), msl)
        XCTAssertTrue(forward.lowerBound < alias.lowerBound && alias.lowerBound < definition.lowerBound, msl)
        XCTAssertTrue(msl.contains("using Pair = array<T, 2>;"), msl)
    }

    // MARK: - Recursive types

    /// WGSL forbids a type that contains itself — it would have infinite size. The layout calculator
    /// walks struct members recursively, so before this check `struct S { a: array<S, 2> }` recursed
    /// until the stack overflowed (signal 11) instead of failing shader module creation.
    func test_recursiveStructsAreRejectedAtModuleCreationWithTheirLine() throws {
        let cases: [(source: String, line: Int, cycle: String)] = [
            ("""
            struct S { a: array<S, 2> }
            """, 1, "S → S"),
            ("""
            struct S { a: S }
            """, 1, "S → S"),
            ("""
            const n = 1;
            struct S { a: array<A, 2> }
            alias A = S;
            """, 2, "S → A → S"),
            ("""
            struct A { b: B }

            struct B {
                x: f32,
                a: array<A, 4>,
            }
            """, 1, "A → B → A"),
            ("""
            alias X = Y;
            alias Y = X;
            """, 1, "X → Y → X"),
        ]
        for (source, line, cycle) in cases {
            XCTAssertThrowsError(try WGSLShaderModule(source: source + """

            @group(0) @binding(0) var<storage, read> data: array<f32>;
            @compute @workgroup_size(1) fn cs() { _ = data[0]; }
            """), source) { error in
                let wgpuError = error as? WGPUError
                XCTAssertEqual(wgpuError?.kind, .validation, source)
                XCTAssertEqual(wgpuError?.line, line, source)
                let message = wgpuError?.message ?? ""
                XCTAssertTrue(message.contains("recursive type") && message.contains(cycle), "\(source)\n→ \(message)")
            }
        }
    }

    /// The layout calculator is bounded even when handed a recursive AST directly (the parser check
    /// keeps one from ever reaching it — this guards the walk itself).
    func test_theLayoutWalkStopsOnARecursiveStructInsteadOfRecursingForever() {
        var module = WGSLModule()
        module.structs = [WGSLStruct(name: "S", members: [
            WGSLStructMember(attributes: [], name: "x", type: .scalar("f32")),
            WGSLStructMember(attributes: [], name: "a", type: .array(element: .named("S"), count: .intLiteral("2"))),
        ])]
        let placement = WGSLLayout.layout(of: module.structs[0], module: module, uniform: false)
        XCTAssertEqual(placement.members.map(\.offset), [0, 4])
        XCTAssertEqual(WGSLLayout.size(of: .named("S"), module: module, uniform: true), 16)
    }

    // MARK: - Parser nesting depth

    /// Runs `body` on a thread with a 512 KB stack — the default for a secondary thread on Apple
    /// platforms, which is where commands run (the JS thread). Overflowing it kills the test process.
    private func onSmallStack(
        file: StaticString = #filePath, line: UInt = #line, _ body: @escaping () -> Void
    ) {
        let finished = DispatchSemaphore(value: 0)
        let thread = Thread {
            body()
            finished.signal()
        }
        thread.stackSize = 512 * 1024
        thread.start()
        XCTAssertEqual(finished.wait(timeout: .now() + 60), .success, "did not finish in time", file: file, line: line)
    }

    /// The parser is recursive descent, so nesting in untrusted source used to recurse once per level
    /// until the stack ran out (signal 11) — 200,000 parentheses or a 200,000-deep `array<array<…>>`
    /// on the main thread, a few dozen levels on a 512 KB stack. Past Tint's depth limit it is now a
    /// validation error with the line. (10,000 levels here: far past any stack, and the debug-build
    /// lexer takes seconds per 100 KB.)
    func test_absurdlyDeepNestingIsAParseErrorNotAStackOverflow() {
        let depth = 10_000
        let sources = [
            "const x = " + String(repeating: "(", count: depth) + "1" + String(repeating: ")", count: depth) + ";",
            "var<private> v: " + String(repeating: "array<", count: depth) + "f32"
                + String(repeating: ", 1>", count: depth) + ";",
            "const y = " + String(repeating: "- ", count: depth) + "1;",
            "fn f() " + String(repeating: "{", count: depth) + String(repeating: "}", count: depth),
            "fn g() {\n" + String(repeating: "if true { ", count: depth) + String(repeating: "}", count: depth) + "\n}",
        ]
        for source in sources {
            var failure: WGPUError?
            onSmallStack {
                do { _ = try WGSLShaderModule(source: source) } catch { failure = error as? WGPUError }
            }
            XCTAssertEqual(failure?.kind, .validation, String(source.prefix(40)))
            XCTAssertNotNil(failure?.line, String(source.prefix(40)))
            XCTAssertTrue(failure?.message.contains("nesting is deeper than") ?? false, "\(String(describing: failure))")
        }
    }

    /// **Whatever the parser accepts also translates**, on a 512 KB caller stack. The parser's limit
    /// alone was not enough: the emitter's frames are much larger than the parser's, so 50 nested
    /// calls or 120 nested blocks parsed and then overflowed `translateToMSL` (signal 11). Parsing and
    /// emission now run on `WGSLDeepStack`. Each shape is checked at the deepest accepted nesting and
    /// one past it (a function body's statement and its expression take two levels).
    func test_everythingTheParserAcceptsTranslatesOnASmallStack() {
        let deepest = WGSLParser.maximumNestingDepth - 2
        func calls(_ n: Int) -> String {
            "let x = " + String(repeating: "abs(", count: n) + "1.0" + String(repeating: ")", count: n) + ";"
        }
        func parens(_ n: Int) -> String {
            "let x = " + String(repeating: "(", count: n) + "1.0" + String(repeating: ")", count: n) + ";"
        }
        func blocks(_ n: Int) -> String {
            "var x = 0.0; " + String(repeating: "{ ", count: n) + "x = 1.0;" + String(repeating: " }", count: n)
        }
        func types(_ n: Int) -> String {
            // The innermost length expression is the last level.
            "var a: " + String(repeating: "array<", count: n) + "f32" + String(repeating: ", 1>", count: n)
                + "; let x = 1.0;"
        }
        for (name, shape) in [("calls", calls), ("parens", parens), ("blocks", blocks), ("types", types)] {
            for depth in [deepest, deepest + 1] {
                let source = """
                @fragment fn fs() -> @location(0) vec4f {
                    \(shape(depth))
                    return vec4f(x);
                }
                """
                var msl: String?
                var failure: WGPUError?
                onSmallStack {
                    do {
                        let module = try WGSLShaderModule(source: source)
                        let bindings = try WGSLBindingAssigner.assign(groups: module.autoBindGroupLayouts(entryPoints: ["fs"]))
                        msl = try module.translateToMSL(entryPoints: ["fs"], bindings: bindings)
                    } catch {
                        failure = error as? WGPUError
                    }
                }
                if depth == deepest {
                    XCTAssertNil(failure, "\(name) at \(depth): \(String(describing: failure))")
                    XCTAssertNotNil(msl, "\(name) at \(depth)")
                } else {
                    XCTAssertTrue(failure?.message.contains("nesting is deeper than") ?? false,
                                  "\(name) at \(depth): \(String(describing: failure))")
                }
            }
        }
        // The deepest accepted shader also passes the Metal compiler.
        _ = try? translate("""
        @fragment fn fs() -> @location(0) vec4f {
            \(calls(deepest))
            return vec4f(x);
        }
        """, entryPoints: ["fs"])
    }

    /// An `else if` chain is not nesting — generated shaders chain hundreds of arms, and Tint (so
    /// Chrome) parses them in a loop. The chain is one flat statement here too: parsed, walked and
    /// emitted iteratively, so its length neither counts toward the depth limit nor recurses.
    func test_aLongElseIfChainIsFlatNotNested() throws {
        func chain(_ arms: Int) -> String {
            var body = "    if i == 0u { r = 0.5; }\n"
            for arm in 1..<arms { body += "    else if i == \(arm)u { r = \(arm).0; }\n" }
            return """
            @group(0) @binding(0) var<uniform> i: u32;
            @fragment fn fs() -> @location(0) vec4f {
                var r = 0.0;
            \(body)    else { r = -1.0; }
                return vec4f(r);
            }
            """
        }

        let msl = try translate(chain(20), entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("else if ((i == 19u))") || msl.contains("else if (i == 19u)"), msl)
        XCTAssertTrue(msl.contains("else\n"), msl)

        let long = chain(1000)
        var output: String?
        var failure: Error?
        onSmallStack {
            do {
                let module = try WGSLShaderModule(source: long)
                let bindings = try WGSLBindingAssigner.assign(groups: module.autoBindGroupLayouts(entryPoints: ["fs"]))
                output = try module.translateToMSL(entryPoints: ["fs"], bindings: bindings)
            } catch { failure = error }
        }
        XCTAssertNil(failure)
        XCTAssertEqual(output?.components(separatedBy: "else if").count, 1000)
    }

    /// A cyclic constant used as an array length gives up with an error instead of recursing.
    func test_aCyclicArrayLengthConstantIsAnErrorNotARecursion() throws {
        let module = try WGSLShaderModule(source: """
        const A = B;
        const B = A;
        @fragment fn fs() -> @location(0) vec4f {
            var v: array<f32, A>;
            return vec4f(v[0]);
        }
        """)
        XCTAssertThrowsError(try module.translateToMSL(entryPoints: ["fs"], bindings: WGSLBindingAssignment()))
    }

    /// Lengths and `@size` values that overflow `Int` are clamped rather than trapping.
    func test_hugeArrayLengthsAndSizeAttributesDoNotOverflowTheLayout() throws {
        let module = try WGSLParser.parse("""
        struct S {
            a: array<array<f32, 4000000000>, 4000000000>,
            @size(4294967295) b: f32,
            @align(2147483648) c: f32,
        }
        """)
        let structure = try XCTUnwrap(module.structNamed("S"))
        let placement = WGSLLayout.layout(of: structure, module: module, uniform: false)
        XCTAssertEqual(placement.members.count, 3)
        XCTAssertNil(WGSLLayout.constantValue(
            .binary(op: "*", .intLiteral("9223372036854775807"), .intLiteral("2")), module: module
        ))
    }

    /// Long acyclic chains are fine up to the composite nesting limit and rejected past it — the
    /// limit is what bounds every later recursive walk over these types.
    func test_deepStructChainsCompileUpToTheNestingLimitAndAreRejectedPastIt() throws {
        func chain(_ count: Int) -> String {
            var source = "struct S0 { v: f32 }\n"
            for index in 1..<count {
                source += "struct S\(index) { inner: S\(index - 1), pad: f32 }\n"
            }
            return source + """
            @group(0) @binding(0) var<storage, read_write> data: S\(count - 1);
            @compute @workgroup_size(1) fn cs() { data.pad = 1.0; }
            """
        }
        let msl = try translate(chain(40), entryPoints: ["cs"])
        XCTAssertTrue(msl.contains("struct alignas(4) S39 {"), msl)

        XCTAssertThrowsError(try WGSLShaderModule(source: chain(WGSLParser.maximumCompositeNesting + 2))) { error in
            let message = (error as? WGPUError)?.message ?? ""
            XCTAssertTrue(message.contains("levels deep"), message)
        }

        // An alias chain adds no nesting but is walked the same way, so its length is bounded too.
        var aliases = ""
        for index in 0..<300 { aliases += "alias X\(index) = X\(index + 1);\n" }
        aliases += "alias X300 = f32;\n"
        XCTAssertThrowsError(try WGSLShaderModule(source: aliases)) { error in
            let message = (error as? WGPUError)?.message ?? ""
            XCTAssertTrue(message.contains("levels deep"), message)
        }
    }

    // MARK: - Minimum buffer binding size

    /// The spec's "minimum buffer binding size": `SizeOf` of the store type, a runtime-sized array
    /// counting as one element. The engine checks bound ranges against it, and `layout: "auto"`
    /// puts it in each buffer entry's `minBindingSize`.
    func test_reflectionReportsEachBuffersMinimumBindingSize() throws {
        let source = """
        struct Params { t: f32 }
        struct Light { dir: vec3f, color: vec3f }
        struct Particles { count: u32, items: array<vec4f> }
        @group(0) @binding(0) var<uniform> params: Params;
        @group(0) @binding(1) var<uniform> light: Light;
        @group(0) @binding(2) var<storage, read_write> particles: Particles;
        @group(1) @binding(0) var<storage, read> values: array<f32>;
        @group(1) @binding(1) var<uniform> m: mat3x3f;
        @group(1) @binding(2) var tex: texture_2d<f32>;

        @compute @workgroup_size(1)
        fn main() {
            let size = textureDimensions(tex);
            particles.items[0] = vec4f(light.dir * params.t + light.color + m[0],
                                       values[0] + f32(size.x) + f32(particles.count));
        }
        """
        let module = try WGSLShaderModule(source: source)
        XCTAssertEqual(module.reflection.minimumBindingSizes(usedBy: ["main"]), [
            // `struct { t: f32 }` is 4 bytes even in the uniform address space — WGSL's SizeOf does not
            // depend on it (the uniform rules only constrain offsets), and the spec/Dawn accept 4.
            0: [0: 4, 1: 32, 2: 32],
            // One f32 element; a mat3x3f is three 16-byte columns.
            1: [0: 4, 1: 48],
        ])
        XCTAssertNil(module.reflection.resource(named: "tex")?.minimumBindingSize)

        let auto = module.autoBindGroupLayouts(entryPoints: ["main"])
        XCTAssertEqual(auto[0].first?.layout, .buffer(WGPUBufferBindingLayout(type: .uniform, minBindingSize: 4)))
        XCTAssertEqual(auto[0].last?.layout, .buffer(WGPUBufferBindingLayout(type: .storage, minBindingSize: 32)))

        _ = try translate(source, entryPoints: ["main"])
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

    /// The trailing `offset` argument of the sampling builtins must reach Metal, where it is the last
    /// argument after any LOD option. It used to be dropped silently — the shader compiled and
    /// sampled the texel next to the one asked for.
    func test_sampleOffsetsReachMetalOn2DTextures() throws {
        let msl = try translate("""
        @group(0) @binding(0) var t: texture_2d<f32>;
        @group(0) @binding(1) var s: sampler;
        const shift = vec2i(-2, 3);

        @fragment
        fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
            let a = textureSample(t, s, uv, vec2i(1, -1));
            let b = textureSampleLevel(t, s, uv, 2.0, vec2<i32>(0, 1));
            let c = textureSampleBias(t, s, uv, 0.5, shift);
            let d = textureSampleGrad(t, s, uv, vec2f(0.01), vec2f(0.02), vec2(1, 1));
            let e = textureSample(t, s, uv);
            return a + b + c + d + e;
        }
        """, entryPoints: ["fs"])

        XCTAssertTrue(msl.contains("t.sample(s, uv, int2(int2(1, -1)))"), msl)
        XCTAssertTrue(msl.contains("t.sample(s, uv, level(2.0), int2(int2(0, 1)))"), msl)
        XCTAssertTrue(msl.contains("t.sample(s, uv, bias(0.5), int2(shift))"), msl)
        XCTAssertTrue(msl.contains("t.sample(s, uv, gradient2d(float2(0.01), float2(0.02)), int2(wgpu_aint2(1, 1)))"), msl)
        XCTAssertTrue(msl.contains("t.sample(s, uv);"), "no offset, no extra argument:\n\(msl)")
    }

    func test_sampleOffsetsFollowTheArrayIndexOnArrayTextures() throws {
        let msl = try translate("""
        @group(0) @binding(0) var t: texture_2d_array<f32>;
        @group(0) @binding(1) var s: sampler;

        @fragment
        fn fs(@location(0) uv: vec2f, @location(1) @interpolate(flat) layer: i32) -> @location(0) vec4f {
            let a = textureSample(t, s, uv, layer, vec2i(1, 0));
            let b = textureSampleLevel(t, s, uv, layer, 1.0, vec2i(0, 1));
            let c = textureSampleBias(t, s, uv, 2u, -1.0, vec2i(-1, 0));
            let d = textureSampleGrad(t, s, uv, layer, vec2f(0.0), vec2f(0.0), vec2i(0, -1));
            return a + b + c + d;
        }
        """, entryPoints: ["fs"])

        XCTAssertTrue(msl.contains("t.sample(s, uv, layer, int2(int2(1, 0)))"), msl)
        XCTAssertTrue(msl.contains("t.sample(s, uv, layer, level(1.0), int2(int2(0, 1)))"), msl)
        XCTAssertTrue(msl.contains("t.sample(s, uv, 2u, bias(-1.0), int2(int2(-1, 0)))"), msl)
        XCTAssertTrue(msl.contains("t.sample(s, uv, layer, gradient2d(float2(0.0), float2(0.0)), int2(int2(0, -1)))"), msl)
    }

    /// A 3D texture takes a 3-component offset — and its gradients are `gradient3d`, not `gradient2d`.
    func test_sampleOffsetsOn3DTexturesAreThreeComponents() throws {
        let msl = try translate("""
        @group(0) @binding(0) var t: texture_3d<f32>;
        @group(0) @binding(1) var s: sampler;

        @fragment
        fn fs(@location(0) uvw: vec3f) -> @location(0) vec4f {
            let a = textureSample(t, s, uvw, vec3i(1, 0, -1));
            let b = textureSampleLevel(t, s, uvw, 0.0, vec3i(0, 0, 1));
            let c = textureSampleGrad(t, s, uvw, vec3f(0.1), vec3f(0.1), vec3i(1, 1, 1));
            return a + b + c;
        }
        """, entryPoints: ["fs"])

        XCTAssertTrue(msl.contains("t.sample(s, uvw, int3(int3(1, 0, -1)))"), msl)
        XCTAssertTrue(msl.contains("t.sample(s, uvw, level(0.0), int3(int3(0, 0, 1)))"), msl)
        XCTAssertTrue(msl.contains("t.sample(s, uvw, gradient3d(float3(0.1), float3(0.1)), int3(int3(1, 1, 1)))"), msl)
    }

    /// Depth textures: plain and level sampling, and both comparison forms — `…CompareLevel` is
    /// mip level 0, spelled out so the offset can follow it (and so it works outside fragment shaders).
    func test_depthSamplingAndComparisonKeepTheirOffsets() throws {
        let msl = try translate("""
        @group(0) @binding(0) var shadow: texture_depth_2d;
        @group(0) @binding(1) var shadows: texture_depth_2d_array;
        @group(0) @binding(2) var cmp: sampler_comparison;
        @group(0) @binding(3) var s: sampler;

        @fragment
        fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
            let a = textureSample(shadow, s, uv, vec2i(1, 1));
            let b = textureSampleLevel(shadow, s, uv, 0, vec2i(-1, 0));
            let c = textureSampleCompare(shadow, cmp, uv, 0.5, vec2i(0, 1));
            let d = textureSampleCompareLevel(shadow, cmp, uv, 0.5, vec2i(1, 0));
            let e = textureSampleCompare(shadows, cmp, uv, 3, 0.5, vec2i(-1, -1));
            let f = textureSampleCompareLevel(shadows, cmp, uv, 3, 0.5, vec2i(1, -1));
            let g = textureSampleCompareLevel(shadow, cmp, uv, 0.25);
            return vec4f(a + b + c + d, e + f, g, 1.0);
        }
        """, entryPoints: ["fs"])

        XCTAssertTrue(msl.contains("shadow.sample(s, uv, int2(int2(1, 1)))"), msl)
        XCTAssertTrue(msl.contains("shadow.sample(s, uv, level(0), int2(int2(-1, 0)))"), msl)
        XCTAssertTrue(msl.contains("shadow.sample_compare(cmp, uv, 0.5, int2(int2(0, 1)))"), msl)
        XCTAssertTrue(msl.contains("shadow.sample_compare(cmp, uv, 0.5, level(0), int2(int2(1, 0)))"), msl)
        XCTAssertTrue(msl.contains("shadows.sample_compare(cmp, uv, 3, 0.5, int2(int2(-1, -1)))"), msl)
        XCTAssertTrue(msl.contains("shadows.sample_compare(cmp, uv, 3, 0.5, level(0), int2(int2(1, -1)))"), msl)
        XCTAssertTrue(msl.contains("shadow.sample_compare(cmp, uv, 0.25, level(0))"), msl)
    }

    /// `textureSampleCompareLevel` has no implicit derivatives, so it is legal in a vertex shader —
    /// Metal's option-less `sample_compare` is not, hence the explicit `level(0)`.
    func test_compareLevelSamplingWorksOutsideFragmentShaders() throws {
        let msl = try translate("""
        @group(0) @binding(0) var shadow: texture_depth_2d;
        @group(0) @binding(1) var cmp: sampler_comparison;

        @vertex
        fn vs() -> @builtin(position) vec4f {
            let lit = textureSampleCompareLevel(shadow, cmp, vec2f(0.5), 0.5);
            return vec4f(lit, 0.0, 0.0, 1.0);
        }
        """, entryPoints: ["vs"])
        XCTAssertTrue(msl.contains("shadow.sample_compare(cmp, float2(0.5), 0.5, level(0))"), msl)
    }

    /// Cubes take no offset in WGSL; their gradients are `gradientcube`.
    func test_cubeSamplingRejectsAnOffsetAndUsesCubeGradients() throws {
        let msl = try translate("""
        @group(0) @binding(0) var sky: texture_cube<f32>;
        @group(0) @binding(1) var s: sampler;

        @fragment
        fn fs(@location(0) dir: vec3f) -> @location(0) vec4f {
            return textureSampleGrad(sky, s, dir, vec3f(0.1), vec3f(0.1));
        }
        """, entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("sky.sample(s, dir, gradientcube(float3(0.1), float3(0.1)))"), msl)

        let module = try WGSLShaderModule(source: """
        @group(0) @binding(0) var sky: texture_cube<f32>;
        @group(0) @binding(1) var s: sampler;
        @fragment
        fn fs(@location(0) dir: vec3f) -> @location(0) vec4f {
            return textureSample(sky, s, dir, vec3i(1, 0, 0));
        }
        """)
        let bindings = try WGSLBindingAssigner.assign(groups: module.autoBindGroupLayouts(entryPoints: ["fs"]))
        XCTAssertThrowsError(try module.translateToMSL(entryPoints: ["fs"], bindings: bindings)) { error in
            XCTAssertTrue((error as? WGPUError)?.message.contains("too many arguments") ?? false, "\(error)")
        }
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

    /// WGSL module scope is **one namespace** — a second declaration of any kind is a shader-creation
    /// error. It used to reach the emitter, whose name-keyed global table **trapped** (killing the host
    /// process) on `var<private> a: f32; var<private> a: f32;`.
    func test_aDuplicateModuleScopeDeclarationIsAShaderCreationError() {
        let cases: [(source: String, line: Int)] = [
            ("var<private> a: f32;\nvar<private> a: f32;", 2),
            ("struct S { x: f32 }\nalias S = f32;", 2),
            ("const k = 1;\noverride k: i32;", 2),
            ("var<private> f: f32;\n\nfn f() {}", 3),
            ("fn g() {}\n@fragment\nfn g() -> @location(0) vec4f { return vec4f(1); }", 3),
        ]
        for (source, line) in cases {
            XCTAssertThrowsError(try WGSLShaderModule(source: source), source) { error in
                let error = error as? WGPUError
                XCTAssertEqual(error?.kind, .validation, source)
                XCTAssertEqual(error?.line, line, "the second declaration's line is reported: \(source)")
                XCTAssertTrue(error?.message.contains("redeclaration of") == true, error?.message ?? source)
            }
        }
    }

    /// Defence in depth — even if duplicate globals bypass the parser, building the emitter must not trap.
    func test_theEmitterDoesNotTrapOnDuplicateGlobalsThatBypassTheParser() throws {
        var ast = try WGSLParser.parse("""
        var<private> a: f32;
        @fragment fn fs() -> @location(0) vec4f { return vec4f(a); }
        """)
        ast.globals.append(ast.globals[0])
        var emitter = MSLEmitter(
            module: ast, reflection: WGSLReflectionBuilder.build(ast), bindings: WGSLBindingAssignment()
        )
        // The first declaration wins (`uniquingKeysWith`) — it must emit, not trap.
        let msl = try emitter.emit(entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("float a"), msl)
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

    func test_identifiersNamedLikeMetalNamespaceMembersAreRenamed() throws {
        // `bias`, `level`, `coord`, … are structs and enums in the `metal` namespace. They are not MSL
        // keywords, so they declared fine — but with the prelude's `using namespace metal;` every use of a
        // module-scope `constant int bias` was "reference to 'bias' is ambiguous". Locals hide the
        // namespace and compiled, which is why only an `override`/`const` exposed it.
        let source = """
        override bias: i32 = 2;
        const level = 3u;
        struct Sample { coord: vec2f, component: f32 }
        @group(0) @binding(0) var<uniform> address: Sample;

        fn lod(mesh: f32) -> f32 { return mesh + f32(level); }

        @fragment
        fn fs_main() -> @location(0) vec4f {
            let compare_func = address.coord.x + address.component;
            return vec4f(lod(compare_func) - f32(bias), 0.0, 0.0, 1.0);
        }
        """
        let msl = try translate(source, entryPoints: ["fs_main"])
        for name in ["bias", "level", "coord", "component", "address", "mesh", "compare_func"] {
            XCTAssertTrue(msl.contains("wgpu_id_\(name)"), "'\(name)' was not renamed:\n\(msl)")
        }
        // A builtin with a colliding *spelling* is not a user identifier — its call must stay as written.
        let builtin = try translate("""
        @fragment fn fs_main() -> @location(0) vec4f { return vec4f(length(vec2f(3.0, 4.0)), 0.0, 0.0, 1.0); }
        """, entryPoints: ["fs_main"])
        XCTAssertTrue(builtin.contains("length(float2("), builtin)
        XCTAssertFalse(builtin.contains("wgpu_id_length"), builtin)
    }

    /// The `wgpu_` namespace belongs to the emitter (prelude helpers, padding members, entry point
    /// wrappers, …). User identifiers spelled into it move to `wgpu_id_<name>` at declaration and every
    /// use, the same as MSL reserved words — otherwise they redeclare an internal name or, worse,
    /// silently **overload** a prelude helper.
    func test_userIdentifiersInTheInternalNamespaceAreMovedOutOfIt() throws {
        let source = """
        struct wgpu_fs_out {
            a: f32,
            wgpu_pad0: vec4f,
        }
        alias wgpu_vec = vec4f;
        @group(0) @binding(0) var<uniform> u: wgpu_fs_out;

        fn wgpu_mod(a: f32, b: f32) -> f32 { return a + b; }

        @fragment
        fn id_wgpu_fs() -> @location(0) vec4f {
            let wgpu_out = wgpu_mod(u.a, u.wgpu_pad0.x) + u.a % 2.0;
            return wgpu_vec(wgpu_out, 0.0, 0.0, 1.0);
        }
        """
        let msl = try translate(source, entryPoints: ["id_wgpu_fs"])

        // A user member next to the padding the layout inserts (`char wgpu_pad0[12]`).
        XCTAssertTrue(msl.contains("char wgpu_pad0[12];"), msl)
        XCTAssertTrue(msl.contains("float4 wgpu_id_wgpu_pad0;"), msl)
        // A user function spelled like the `%` helper must not become an overload of it.
        XCTAssertTrue(msl.contains("float wgpu_id_wgpu_mod(float a, float b)"), msl)
        XCTAssertTrue(msl.contains("wgpu_id_wgpu_mod(u.a, u.wgpu_id_wgpu_pad0.x) + wgpu_mod(u.a, 2.0)"), msl)
        // An alias used as a constructor resolves to the renamed declaration.
        XCTAssertTrue(msl.contains("using wgpu_id_wgpu_vec = float4;"), msl)
        XCTAssertTrue(msl.contains("return wgpu_id_wgpu_vec(wgpu_id_wgpu_out, "), msl)
        // The user's `wgpu_fs_out` and the entry point `id_wgpu_fs`'s output struct stay distinct.
        XCTAssertTrue(msl.contains("struct alignas(16) wgpu_id_wgpu_fs_out {"), msl)
        XCTAssertTrue(msl.contains("fragment wgpu_ep_id_wgpu_fs_out id_wgpu_fs("), msl)
        // The runtime looks entry points up by the same mapping.
        XCTAssertEqual(WGSLShaderModule.mslFunctionName(for: "wgpu_fs"), "wgpu_id_wgpu_fs")
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

    // MARK: - Pipeline constant conversion (WebIDL semantics, never a trap)

    private static let overrideTypesShader = """
    enable f16;
    override n: u32 = 1;
    override k: i32 = 1;
    override s: f32 = 1.0;
    override h: f16 = 1.0h;
    override flag: bool = false;
    override untypedInt = 1;
    override untypedFloat = 0.5;

    @group(0) @binding(0) var<storage, read_write> out: array<f32>;

    @compute @workgroup_size(1)
    fn main() {
        var v = f32(n) + f32(k) + s + f32(h) + f32(untypedInt) + untypedFloat;
        if (flag) { v = -v; }
        out[0] = v;
    }
    """

    private func translateOverrides(_ constants: [String: Double]) throws -> String {
        let module = try WGSLShaderModule(source: Self.overrideTypesShader)
        let bindings = try WGSLBindingAssigner.assign(groups: module.autoBindGroupLayouts(entryPoints: ["main"]))
        return try module.translateToMSL(entryPoints: ["main"], bindings: bindings, constants: constants)
    }

    func test_pipelineConstantsAtTheEdgeOfTheirTypeAreAccepted() throws {
        let msl = try translateOverrides([
            "n": 4_294_967_295, "k": -2_147_483_648, "s": 1e30, "h": 65504, "flag": 2,
            "untypedInt": 2_147_483_647, "untypedFloat": 2,
        ])
        XCTAssertTrue(msl.contains("4294967295u"), "u32 max must be planted as an unsigned literal:\n\(msl)")
        // `-2147483648` would be `-(2147483648)` in C++ — not an int literal. Spelled like the prelude.
        XCTAssertTrue(msl.contains("int k = (-2147483647 - 1);"), msl)
        XCTAssertTrue(msl.contains("1e+30"), msl)
        XCTAssertTrue(msl.contains("half h = 65504.0"), msl)
        XCTAssertTrue(msl.contains("bool flag = true"), "a nonzero number is true:\n\(msl)")
        XCTAssertTrue(msl.contains("2147483647"), msl)
        // `override untypedFloat = 0.5;` is f32 — supplying the integral `2` must not turn it into an int.
        XCTAssertTrue(msl.contains("float untypedFloat = 2.0"), "the untyped f32 override lost its type:\n\(msl)")
        MetalCompilerHarness.assertCompiles(msl)
    }

    func test_pipelineConstantFractionsAreTruncatedForIntegerOverrides() throws {
        // [EnforceRange] truncates toward zero rather than rejecting a fraction.
        let msl = try translateOverrides(["n": 7.9, "k": -3.7, "untypedInt": 5.5])
        XCTAssertTrue(msl.contains("uint n = 7u"), msl)
        XCTAssertTrue(msl.contains("int k = -3"), msl)
        XCTAssertTrue(msl.contains("int untypedInt = 5"), msl)
        MetalCompilerHarness.assertCompiles(msl)
    }

    func test_pipelineConstantsTheTypeCannotHoldAreValidationErrors() throws {
        // Each of these used to reach `Int(value)` (or plant `inf`/`nan` text) — a trap kills the host.
        let rejected: [(name: String, value: Double)] = [
            ("n", 1e30), ("n", .infinity), ("n", -1), ("n", 4_294_967_296), ("n", .nan),
            ("k", 2_147_483_648), ("k", -2_147_483_649), ("k", -.infinity), ("k", .nan),
            ("s", .infinity), ("s", .nan), ("s", 1e39),
            // f16 follows Dawn: |value| ≤ 65504 (65510 would still round to 65504, but Dawn rejects it).
            ("h", 65510), ("h", -65504.5), ("h", -1e6),
            // The untyped `override untypedInt = 1;` is i32.
            ("untypedInt", 1e30), ("untypedInt", .infinity),
            ("untypedFloat", .infinity),
        ]
        for (name, value) in rejected {
            XCTAssertThrowsError(try translateOverrides([name: value]), "\(name) = \(value) was accepted") { error in
                let error = error as? WGPUError
                XCTAssertEqual(error?.kind, .validation, "\(name) = \(value): \(String(describing: error))")
                XCTAssertTrue(
                    error?.message.contains("'\(name)'") == true,
                    "it must name the constant: \(String(describing: error))"
                )
            }
        }
    }

    func test_untypedOverridesTakeTheirTypeFromBuiltinsAndShifts() throws {
        // An integral value supplied to an f32 override must stay a float — planted as `4`, MSL would
        // infer `int` and `a / 4` would become integer division.
        let source = """
        override a = max(0.5, 1.0);
        override b = min(1, 2.0);
        override c = abs(-3);
        override d = clamp(1u, 0u, 4u);
        override e = select(0.5, 1.0, true);
        override f = pow(2.0, 3.0);
        override g = sqrt(4);
        override sh = 1 << 2u;
        override unknown = vec2(1.0, 2.0).x;

        @group(0) @binding(0) var<storage, read_write> out: array<f32>;

        @compute @workgroup_size(1)
        fn main() {
            out[0] = a / 4 + b + f32(c) + f32(d) + e + f + g + f32(sh) + unknown / 4;
        }
        """
        let module = try WGSLShaderModule(source: source)
        let bindings = try WGSLBindingAssigner.assign(groups: module.autoBindGroupLayouts(entryPoints: ["main"]))
        let names = ["a", "b", "c", "d", "e", "f", "g", "sh", "unknown"]
        let msl = try module.translateToMSL(
            entryPoints: ["main"], bindings: bindings,
            constants: Dictionary(uniqueKeysWithValues: names.map { ($0, 4.0) })
        )
        for expected in [
            "float a = 4.0", "float b = 4.0", "int c = 4;", "uint d = 4u", "float e = 4.0",
            "float f = 4.0", "float g = 4.0", "int sh = 4;",
            // Member access takes the vector's component type (`vec2(1.0, 2.0)` is a vec2<f32>).
            "float unknown = 4.0",
        ] {
            XCTAssertTrue(msl.contains(expected), "missing `\(expected)`:\n\(msl)")
        }
        MetalCompilerHarness.assertCompiles(msl)

        // The f32 fallback still rejects what f32 cannot hold.
        XCTAssertThrowsError(try module.translateToMSL(
            entryPoints: ["main"], bindings: bindings,
            constants: Dictionary(uniqueKeysWithValues: names.map { ($0, $0 == "unknown" ? 1e39 : 4.0) })
        )) { error in
            XCTAssertEqual((error as? WGPUError)?.kind, .validation)
        }
    }

    func test_untypedIntegerAndBoolOverridesKeepTheirTypeThroughBuiltinsAndAccess() throws {
        // The f32 fallback is only for initializers that cannot be typed — integer and bool builtins,
        // bitcast, and member/index access on typed vectors and arrays must keep their real type.
        let source = """
        alias Count = u32;
        alias Count2 = Count;
        const lanes: vec2<u32> = vec2<u32>(1u, 2u);
        override bits = countOneBits(7u);
        override lead = firstLeadingBit(8);
        override bc = bitcast<u32>(1.0);
        override every = all(vec2(true, true));
        override packed = pack4x8unorm(vec4f(0.0));
        override dp = dot(vec2i(1, 2), vec2i(3, 4));
        override len = length(vec2f(3.0, 4.0));
        override mx = vec2i(1, 2).x;
        override el = array<u32, 2>(1u, 2u)[1];
        override lane = lanes.y;
        override sum = 1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1;
        override chained: Count2 = 1u;

        @group(0) @binding(0) var<storage, read_write> out: array<f32>;

        @compute @workgroup_size(1)
        fn main() {
            var v = f32(bits + bc + packed + el + lane + chained) + f32(lead + dp + mx + sum) + len;
            if (every) { v = -v; }
            out[0] = v;
        }
        """
        let module = try WGSLShaderModule(source: source)
        let bindings = try WGSLBindingAssigner.assign(groups: module.autoBindGroupLayouts(entryPoints: ["main"]))
        let names = ["bits", "lead", "bc", "every", "packed", "dp", "len", "mx", "el", "lane", "sum", "chained"]
        let msl = try module.translateToMSL(
            entryPoints: ["main"], bindings: bindings,
            constants: Dictionary(uniqueKeysWithValues: names.map { ($0, 4.0) })
        )
        for expected in [
            "uint bits = 4u;", "int lead = 4;", "uint bc = 4u;", "bool every = true;", "uint packed = 4u;",
            "int dp = 4;", "float len = 4.0;", "int mx = 4;", "uint el = 4u;", "uint lane = 4u;",
            // A 20-term sum nests 19 binaries deep — still typed, not dropped to the f32 fallback.
            "int sum = 4;",
            "chained = 4u;",
        ] {
            XCTAssertTrue(msl.contains(expected), "missing `\(expected)`:\n\(msl)")
        }
        MetalCompilerHarness.assertCompiles(msl)

        // The alias chain resolves to u32, so the u32 range check applies.
        XCTAssertThrowsError(try module.translateToMSL(
            entryPoints: ["main"], bindings: bindings,
            constants: Dictionary(uniqueKeysWithValues: names.map { ($0, $0 == "chained" ? -1 : 4.0) })
        )) { error in
            let error = error as? WGPUError
            XCTAssertEqual(error?.kind, .validation)
            XCTAssertTrue(error?.message.contains("u32") == true, String(describing: error))
        }
    }

    private func translateWithOverride(_ declarations: String, value: Double = 4) throws -> String {
        let source = """
        \(declarations)

        @group(0) @binding(0) var<storage, read_write> out: array<f32>;

        @compute @workgroup_size(1)
        fn main() {
            out[0] = f32(o);
        }
        """
        let module = try WGSLShaderModule(source: source)
        let bindings = try WGSLBindingAssigner.assign(groups: module.autoBindGroupLayouts(entryPoints: ["main"]))
        return try module.translateToMSL(entryPoints: ["main"], bindings: bindings, constants: ["o": value])
    }

    func test_selfReferentialAliasesAndConstantsDoNotCrashOverrideTyping() {
        // Invalid shaders, but user input: typing the override must terminate (these recursed until the
        // stack overflowed). Either a graceful fallback or a thrown error is fine — not a crash.
        let shapes = [
            "alias A = array<A, 2>;\noverride o = bitcast<A>(1u);",
            "alias A = array<A, 2>;\nconst c: A = A();\noverride o = c[0];",
            "alias A = vec2<A>;\noverride o = bitcast<A>(1u);",
            "alias A = atomic<A>;\noverride o = bitcast<A>(1u);",
            "alias A = array<B, 2>;\nalias B = array<A, 2>;\noverride o = bitcast<A>(1u);",
            "alias A = B;\nalias B = A;\noverride o: A = 1;",
            "const a = a + 1;\noverride o = a;",
            "const a = b;\nconst b = a;\noverride o = a;",
        ]
        for shape in shapes {
            if let msl = try? translateWithOverride(shape, value: 1) {
                XCTAssertTrue(msl.contains(" o = "), "the override must still be planted:\n\(msl)")
            }
        }
    }

    func test_overrideTypingIsLinearInConstantChains() throws {
        // `x(k) = x(k-1) + x(k-1)` — without memoizing each constant's type this walks 2^30 nodes.
        var declarations = ["const x0 = 1u;"]
        for k in 1...30 { declarations.append("const x\(k) = x\(k - 1) + x\(k - 1);") }
        declarations.append("override o = x30;")
        let started = Date()
        let msl = try translateWithOverride(declarations.joined(separator: "\n"))
        XCTAssertLessThan(Date().timeIntervalSince(started), 5, "typing the override must not blow up")
        XCTAssertTrue(msl.contains("uint o = 4u;"), msl)
        MetalCompilerHarness.assertCompiles(msl)
    }

    func test_longAliasChainsDoNotOverflowASmallStack() {
        // A 10k-link alias chain used to recurse once per link — signal 11 on a 512 KB stack (a JS
        // thread's size), and O(n²) on top. Past the hop budget the override falls back gracefully.
        var aliases = ["alias A0 = u32;"]
        for k in 1..<10_000 { aliases.append("alias A\(k) = A\(k - 1);") }
        let chain = aliases.joined(separator: "\n")

        for declaration in ["override o: A9999 = 1u;", "override o = A9999(1u);"] {
            let finished = DispatchSemaphore(value: 0)
            let outcome = ThreadOutcome()
            let source = chain + "\n" + declaration
            let thread = Thread { [self] in
                do { outcome.msl = try translateWithOverride(source, value: 1) } catch { outcome.error = error }
                finished.signal()
            }
            thread.stackSize = 512 * 1024
            thread.start()
            XCTAssertEqual(finished.wait(timeout: .now() + 20), .success, "\(declaration) did not finish in time")
            // Read only after the semaphore — the thread is done writing by then.
            if let msl = outcome.msl {
                XCTAssertTrue(msl.contains(" o = "), "the override must still be planted")
            } else {
                XCTAssertNotNil(outcome.error)
            }
        }
    }

    func test_aDepthCutOffIsNotMemoizedAsTheConstantsType() throws {
        // `a` reaches `c` right at the depth limit, so `c` is typed only partially there. That partial
        // nil must not be cached — `b = c` is a plain u32. Sweep around the limit so it stays meaningful.
        // The depth is built from a chain of constants, not parentheses: the parser rejects nesting past
        // `WGSLParser.maximumNestingDepth` (128), well before the typer's limit, but a chain is not nesting.
        for hops in 250...260 {
            var chain = "const k0 = c;\n"
            for index in 1..<hops { chain += "const k\(index) = k\(index - 1);\n" }
            let source = """
            const c = 1u;
            \(chain)override a = k\(hops - 1);
            override b = c;

            @group(0) @binding(0) var<storage, read_write> out: array<f32>;

            @compute @workgroup_size(1)
            fn main() {
                out[0] = f32(a) + f32(b);
            }
            """
            let module = try WGSLShaderModule(source: source)
            let bindings = try WGSLBindingAssigner.assign(groups: module.autoBindGroupLayouts(entryPoints: ["main"]))
            let msl = try module.translateToMSL(
                entryPoints: ["main"], bindings: bindings, constants: ["a": 3, "b": 3]
            )
            XCTAssertTrue(msl.contains("uint b = 3u;"), "\(hops) hops poisoned b's type:\n\(msl)")
            if hops == 255 { MetalCompilerHarness.assertCompiles(msl) }
        }
    }

    func test_aStructNamedLikeAMatrixIsNotAMatrixConstructor() throws {
        // `matter` is six letters starting with "mat" — it is a struct, not `mat2x2`-like. Its member is
        // f32 even though the argument is an AbstractInt, so the override must not be pinned to int.
        let msl = try translateWithOverride("struct matter { a: f32 };\noverride o = matter(1).a;")
        XCTAssertTrue(msl.contains("float o = 4.0;"), msl)
        MetalCompilerHarness.assertCompiles(msl)

        let matrix = try translateWithOverride("override o = mat2x2(1, 2, 3, 4)[0][1];")
        XCTAssertTrue(matrix.contains("int o = 4;"), "a real matrix constructor still types from its arguments:\n\(matrix)")
    }

    func test_pipelineBoolConstantFollowsIDLBoolean() throws {
        // IDL `boolean`: 0, -0 and NaN are false.
        for value in [0, -0.0, Double.nan] {
            let msl = try translateOverrides(["flag": value])
            XCTAssertTrue(msl.contains("bool flag = false"), "\(value):\n\(msl)")
            MetalCompilerHarness.assertCompiles(msl)
        }
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
        // The subtraction saturates — a binding smaller than the offset gives 0, not a wrapped ~4e9.
        XCTAssertTrue(
            msl.contains("((wgpu_buffer_sizes[0] > 16u ? wgpu_buffer_sizes[0] - 16u : 0u) / uint(sizeof(float4)))"), msl
        )
    }

    func test_withoutArrayLengthTheSizeTableIsNotPassed() throws {
        let msl = try translate(Self.triangleShader, entryPoints: ["vs_main", "fs_main"])
        XCTAssertFalse(msl.contains("wgpu_buffer_sizes"), msl)
    }

    // MARK: - Compile-time constants

    /// The zero-value constructor `array<T, N>()` has no arguments to count, so its length can only
    /// come from the written type. It used to come out as `array<uint, 0>{}` — a zero-length array
    /// whose every index clamps to nothing.
    func test_aZeroArgumentArrayConstructorKeepsItsDeclaredLength() throws {
        let msl = try translate("""
        struct Item { a: vec3f, b: f32 }
        const N = 8u;
        const zeros = array<f32, 4>();

        @fragment
        fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
            var data = array<u32, 4>();
            data[1] = 3u;
            var grid = array<array<f32, 2>, 3>();
            grid[2][1] = 1.0;
            var items = array<Item, 2>();
            items[1].b = 2.0;
            var colors = array<vec3f, 5>();
            var counts = array<i32, N>();
            counts[7] = 1;
            let lut = array<f32, 3>(1.0, 2.0, 3.0);
            let inferred = array(vec2f(1.0), vec2f(2.0));
            return vec4f(f32(data[1]) + grid[2][1] + items[1].b + zeros[3] + lut[2] + inferred[1].x,
                         colors[4].x, f32(counts[7]), 1.0);
        }
        """, entryPoints: ["fs"])

        XCTAssertTrue(msl.contains("auto data = array<uint, 4>{};"), msl)
        XCTAssertTrue(msl.contains("auto grid = array<array<float, 2>, 3>{};"), msl)
        XCTAssertTrue(msl.contains("auto items = array<Item, 2>{};"), msl)
        XCTAssertTrue(msl.contains("auto colors = array<float3, 5>{};"), msl)
        XCTAssertTrue(msl.contains("auto counts = array<int, 8>{};"), msl)
        XCTAssertTrue(msl.contains("constant array<float, 4> zeros = array<float, 4>{};"), msl)
        XCTAssertTrue(msl.contains("array<float, 3>{1.0, 2.0, 3.0}"), msl)
        XCTAssertTrue(msl.contains("array<float2, 2>{float2(1.0), float2(2.0)}"), msl)
        XCTAssertFalse(msl.contains(", 0>{}"), "a zero-length array leaked through:\n\(msl)")
    }

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

    // MARK: - Integer attribute arguments / array lengths

    func test_integerLiteralsFollowTheWGSLGrammar() {
        XCTAssertEqual(WGSLIntegerLiteral("0")?.value, 0)
        XCTAssertEqual(WGSLIntegerLiteral("42")?.value, 42)
        XCTAssertEqual(WGSLIntegerLiteral("0x10")?.value, 16)
        XCTAssertEqual(WGSLIntegerLiteral("0XfF")?.value, 255)
        XCTAssertEqual(WGSLIntegerLiteral("7")?.suffix, WGSLIntegerLiteral.Suffix.none)
        XCTAssertEqual(WGSLIntegerLiteral("2u")?.value, 2)
        XCTAssertEqual(WGSLIntegerLiteral("2u")?.suffix, .u)
        XCTAssertEqual(WGSLIntegerLiteral("0x10i")?.value, 16)
        XCTAssertEqual(WGSLIntegerLiteral("0x10i")?.suffix, .i)
        XCTAssertEqual(WGSLIntegerLiteral("0u")?.value, 0)
        XCTAssertEqual(WGSLIntegerLiteral("2147483647i")?.value, Int(Int32.max))
        XCTAssertEqual(WGSLIntegerLiteral("4294967295u")?.value, Int(UInt32.max))
        XCTAssertEqual(WGSLIntegerLiteral("9223372036854775807")?.value, Int.max)

        // Malformed, or out of range for the suffix's type (AbstractInt is i64).
        for text in ["", "0x", "0xg", "08", "1_0", "1.5", "-1", "+1", "u", "1uu", "٣",
                     "2147483648i", "4294967296u", "9223372036854775808", "0x8000000000000000"] {
            XCTAssertNil(WGSLIntegerLiteral(text), "'\(text)' must be rejected")
        }
    }

    func test_hexAndSuffixedAttributeArgumentsAreReadAsWGSLLiterals() throws {
        // Stripping non-digits used to read 0x10 as 10 and (1 + 1) as 11.
        let source = """
        struct VertexInput {
            @location(0x1) position: vec2f,
            @location(0u) weight: f32,
        };
        @group(0) @binding(0x10) var<storage, read> data: array<f32>;
        @group(0u) @binding(1 + 1) var<uniform> tint: vec4f;

        @vertex
        fn vs(input: VertexInput) -> @builtin(position) vec4f {
            return vec4f(input.position * input.weight * data[0], 0.0, 1.0) * tint;
        }
        """
        let module = try WGSLShaderModule(source: source)
        XCTAssertEqual(module.reflection.resource(named: "data")?.binding, 16)
        XCTAssertEqual(module.reflection.resource(named: "tint")?.binding, 2)

        let msl = try translate(source, entryPoints: ["vs"])
        XCTAssertTrue(msl.contains("[[attribute(1)]]"), msl)
        XCTAssertTrue(msl.contains("[[attribute(0)]]"), msl)
    }

    func test_aHexArrayLengthSetsTheStructLayout() throws {
        let source = """
        struct Lights {
            colors: array<vec4f, 0x10>,
            @align(0x20) count: u32,
            @size(0x10u) scale: f32,
        };
        @group(0) @binding(0) var<uniform> lights: Lights;

        @fragment
        fn fs() -> @location(0) vec4f {
            return lights.colors[lights.count] * lights.scale;
        }
        """
        let module = try WGSLParser.parse(source)
        let structure = try XCTUnwrap(module.structNamed("Lights"))
        let placement = WGSLLayout.layout(of: structure, module: module, uniform: true)
        // 16 × vec4f = 256 bytes — 0x10 read as 10 used to put `count` at 160.
        XCTAssertEqual(placement.members.map(\.offset), [0, 256, 260])
        XCTAssertEqual(placement.members.map(\.size), [256, 4, 16])
        XCTAssertEqual(placement.size, 288)

        let msl = try translate(source, entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("array<float4, 16> colors;"), msl)
    }

    func test_constNamedAttributeArgumentsResolveThroughModuleConstants() throws {
        // A module-scope const may be declared after its use.
        let source = """
        @group(GROUP) @binding(LIGHTS) var<uniform> lights: array<vec4f, COUNT>;
        @group(GROUP) @binding(BASE) var<uniform> tint: vec4f;

        @fragment
        fn fs(@location(BASE - 2) uv: vec2f) -> @location(0) vec4f {
            return lights[1] * tint * uv.x;
        }

        const GROUP = 0;
        const BASE = 2u;
        const LIGHTS = (BASE + 0x1) * 2;
        const COUNT = BASE * 2;
        """
        let module = try WGSLShaderModule(source: source)
        XCTAssertEqual(module.reflection.resource(named: "lights")?.binding, 6)
        XCTAssertEqual(module.reflection.resource(named: "lights")?.group, 0)
        XCTAssertEqual(module.reflection.resource(named: "tint")?.binding, 2)

        let msl = try translate(source, entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("[[user(locn0)]]"), msl)
        XCTAssertTrue(msl.contains("array<float4, 4>"), msl)
    }

    func test_anAttributeArgumentThatIsNotAUsableIntegerIsAShaderCreationError() {
        // The attribute text as the message quotes it (argument tokens joined).
        let cases: [(attribute: String, prelude: String, reason: String)] = [
            ("@binding(0x)", "", "not a valid integer literal"),
            ("@binding(08)", "", "not a valid integer literal"),
            ("@binding(4294967296u)", "", "not a valid integer literal"),
            ("@binding(4294967296)", "", "does not fit in u32"),
            ("@binding(1.5)", "", "not an integer"),
            ("@binding(-1)", "", "must be non-negative"),
            ("@binding(missing)", "", "not a module-scope const"),
            ("@binding(SLOT)", "override SLOT: u32 = 1;", "is an override"),
            ("@binding(A)", "const A = B; const B = A;", "defined in terms of itself"),
            ("@binding(4/(2-2))", "", "division by zero"),
            ("@binding(1<<2)", "", "not supported"),
            ("@binding(f(1))", "", "only integer literals"),
            ("@binding()", "", "exactly one argument"),
            ("@binding(1, 2)", "", "exactly one argument"),
        ]
        for testCase in cases {
            let source = """
            \(testCase.prelude)
            @group(0) \(testCase.attribute) var<uniform> u: vec4f;
            @fragment fn fs() -> @location(0) vec4f { return u; }
            """
            XCTAssertThrowsError(try WGSLShaderModule(source: source), testCase.attribute) { error in
                let error = error as? WGPUError
                let message = error?.message ?? ""
                XCTAssertTrue(message.contains(testCase.attribute), "names the attribute: \(message)")
                XCTAssertTrue(message.contains(testCase.reason), "says why (\(testCase.reason)): \(message)")
                XCTAssertEqual(error?.line, 2, message)
            }
        }

        // The other integer attributes go through the same check.
        for (source, reason) in [
            ("struct S { @align(3) a: f32 };", "positive power of two"),
            ("struct S { @size(0) a: f32 };", "must be positive"),
            ("@fragment fn fs() -> @location(oops) vec4f { return vec4f(); }", "not a module-scope const"),
        ] {
            XCTAssertThrowsError(try WGSLShaderModule(source: source), source) { error in
                let message = (error as? WGPUError)?.message ?? ""
                XCTAssertTrue(message.contains(reason), "says why (\(reason)): \(message)")
            }
        }
    }

    func test_aSelfReferentialArrayLengthIsAnErrorNotAHang() throws {
        for source in [
            """
            const A = B + 1;
            const B = A;
            var<private> xs: array<f32, A>;
            @fragment fn fs() -> @location(0) vec4f { return vec4f(xs[0]); }
            """,
            """
            @fragment fn fs() -> @location(0) vec4f {
                const n = n + 1;
                var xs: array<f32, n>;
                return vec4f(xs[0]);
            }
            """,
        ] {
            let module = try WGSLShaderModule(source: source)
            let bindings = try WGSLBindingAssigner.assign(groups: module.autoBindGroupLayouts(entryPoints: ["fs"]))
            XCTAssertThrowsError(try module.translateToMSL(entryPoints: ["fs"], bindings: bindings)) { error in
                let message = (error as? WGPUError)?.message ?? ""
                XCTAssertTrue(message.contains("defined in terms of itself"), message)
            }
        }
    }

    func test_aDeeplyNestedConstantExpressionIsAnErrorNotAStackOverflow() throws {
        // The JS thread's stack is small — evaluation depth is bounded instead of trusting the shader.
        let terms = Array(repeating: "1", count: 1000).joined(separator: "+")
        let chain = (1..<1000).map { "const C\($0) = C\($0 - 1);" }.joined(separator: "\n")
        for source in [
            "@group(0) @binding(\(terms)) var<uniform> u: vec4f;",
            "const C0 = 0;\n\(chain)\n@group(0) @binding(C999) var<uniform> u: vec4f;",
        ] {
            XCTAssertThrowsError(try WGSLShaderModule(source: source)) { error in
                let message = (error as? WGPUError)?.message ?? ""
                XCTAssertTrue(message.contains("nested too deeply"), String(message.suffix(200)))
            }
        }

        // The same, as an array length (an override-expression).
        let module = try WGSLShaderModule(source: """
        var<private> xs: array<f32, \(terms)>;
        @fragment fn fs() -> @location(0) vec4f { return vec4f(xs[0]); }
        """)
        let bindings = try WGSLBindingAssigner.assign(groups: module.autoBindGroupLayouts(entryPoints: ["fs"]))
        XCTAssertThrowsError(try module.translateToMSL(entryPoints: ["fs"], bindings: bindings)) { error in
            let message = (error as? WGPUError)?.message ?? ""
            XCTAssertTrue(message.contains("nested too deeply"), String(message.suffix(200)))
        }

        // A realistic depth still resolves.
        let shallow = Array(repeating: "1", count: 100).joined(separator: "+")
        let shallowModule = try WGSLShaderModule(source: """
        @group(0) @binding(\(shallow)) var<uniform> u: vec4f;
        @fragment fn fs() -> @location(0) vec4f { return u; }
        """)
        XCTAssertEqual(shallowModule.reflection.resource(named: "u")?.binding, 100)
    }

    func test_integerConversionsAreConstExpressions() throws {
        let source = """
        const SLOT = u32(3);
        @group(0) @binding(SLOT) var<uniform> tint: vec4f;
        @group(i32(0)) @binding(i32(SLOT) + 1) var<storage, read_write> out: array<f32>;
        var<private> weights: array<f32, u32(4)>;

        @compute @workgroup_size(1)
        fn main() {
            weights[0] = tint.x;
            out[0] = weights[0];
        }
        """
        let module = try WGSLShaderModule(source: source)
        XCTAssertEqual(module.reflection.resource(named: "tint")?.binding, 3)
        XCTAssertEqual(module.reflection.resource(named: "out")?.binding, 4)

        let msl = try translate(source, entryPoints: ["main"])
        XCTAssertTrue(msl.contains("array<float, 4>"), msl)

        // The conversion keeps its range check.
        for (attribute, reason) in [("@binding(u32(-1))", "does not fit in u32"),
                                    ("@binding(i32(0x80000000))", "does not fit in i32"),
                                    ("@binding(u32(1, 2))", "only integer literals")] {
            let broken = "@group(0) \(attribute) var<uniform> u: vec4f;"
            XCTAssertThrowsError(try WGSLShaderModule(source: broken), attribute) { error in
                let message = (error as? WGPUError)?.message ?? ""
                XCTAssertTrue(message.contains(reason), "says why (\(reason)): \(message)")
            }
        }
    }

    func test_aNonPositiveOverrideArrayLengthIsRejected() throws {
        // A negative pipeline constant used to be read digit by digit — -3 became `array<float, 3>`.
        let source = """
        override N: u32 = 64;
        override bias: i32 = 0;
        @group(0) @binding(0) var<storage, read_write> out: array<f32>;
        var<workgroup> buf: array<f32, N>;

        @compute @workgroup_size(1)
        fn main() {
            buf[0] = f32(10 - bias);
            out[0] = buf[0];
        }
        """
        let module = try WGSLShaderModule(source: source)
        let bindings = try WGSLBindingAssigner.assign(groups: module.autoBindGroupLayouts(entryPoints: ["main"]))
        // -3 does not reach the length check: a u32 override rejects it on conversion ([EnforceRange]).
        for (value, reason) in [(-3.0, "cannot be converted to u32"), (0.0, "an array length must be positive")] {
            XCTAssertThrowsError(
                try module.translateToMSL(entryPoints: ["main"], bindings: bindings, constants: ["N": value]),
                "N = \(value)"
            ) { error in
                let message = (error as? WGPUError)?.message ?? ""
                XCTAssertTrue(message.contains(reason), message)
            }
        }

        // A negative constant elsewhere is planted as a negation (WGSL has no negative literals).
        let msl = try module.translateToMSL(
            entryPoints: ["main"], bindings: bindings, constants: ["N": 8, "bias": -3]
        )
        XCTAssertTrue(msl.contains("array<float, 8>"), msl)
        XCTAssertTrue(msl.contains("-3"), msl)
        MetalCompilerHarness.assertCompiles(msl)
    }

    func test_aConstReferencedManyTimesIsEvaluatedOnce() throws {
        // `const xN = (x(N-1) + x(N-1)) / 2u` references the previous name twice, so evaluated naively the
        // walk doubles per line: 26 lines are 2^26 visits (a minute or more), memoized they are 26. The
        // chain is far below the depth limit, so only the cost tells the two apart — hence the clock.
        // Twenty such lines hung `createShaderModule` on the JS thread.
        var chain = "const x0 = 1u;\n"
        for index in 1...26 { chain += "const x\(index) = (x\(index - 1) + x\(index - 1)) / 2u;\n" }
        let source = chain + """
        @group(0) @binding(x26) var<storage, read_write> out: array<u32, x26>;
        @compute @workgroup_size(x26) fn main() { out[0] = x26; }
        """
        let started = Date()
        let module = try WGSLShaderModule(source: source)
        let auto = module.autoBindGroupLayouts(entryPoints: ["main"])
        let size = try XCTUnwrap(module.workgroupSize(of: "main"))
        let bindings = try WGSLBindingAssigner.assign(groups: auto)
        let msl = try module.translateToMSL(entryPoints: ["main"], bindings: bindings)
        XCTAssertLessThan(Date().timeIntervalSince(started), 3, "the const chain was re-evaluated per reference")
        XCTAssertEqual(auto[0].first?.binding, 1)
        XCTAssertEqual([size.x, size.y, size.z], [1, 1, 1])
        // The MSL is deliberately not pushed through the Metal compiler here: an untyped module-scope
        // `const` is emitted as a `#define`, so this chain expands exponentially in Metal's preprocessor
        // (18 levels take 2.4 s, 26 levels about 25 minutes) — a separate problem from the evaluator's.
        XCTAssertTrue(msl.contains("#define x26 (wgpu_div((x25 + x25), 2u))"), msl)
    }

    // MARK: - @workgroup_size

    func test_workgroupSizeArgumentsAreEvaluatedAsExpressions() throws {
        // Each argument used to be read as `Int(text) ?? 1` — every one of these dispatched 1 thread.
        let source = """
        const BLOCK = 8u;
        @group(0) @binding(0) var<storage, read_write> out: array<u32>;

        @compute @workgroup_size(64u)
        fn suffixed(@builtin(global_invocation_id) id: vec3u) { out[id.x] = 1u; }

        @compute @workgroup_size(0x10, BLOCK, 2)
        fn hexAndConst(@builtin(global_invocation_id) id: vec3u) { out[id.x] = 2u; }

        @compute @workgroup_size(BLOCK * 2, (1i), u32(4))
        fn expressions(@builtin(global_invocation_id) id: vec3u) { out[id.x] = 3u; }
        """
        let module = try WGSLShaderModule(source: source)
        let expected = ["suffixed": [64, 1, 1], "hexAndConst": [16, 8, 2], "expressions": [16, 1, 4]]
        for (entry, size) in expected.sorted(by: { $0.key < $1.key }) {
            let evaluated = try XCTUnwrap(module.workgroupSize(of: entry), entry)
            XCTAssertEqual([evaluated.x, evaluated.y, evaluated.z], size, entry)
            let reflected = try XCTUnwrap(module.reflection.entryPoint(named: entry)?.workgroupSize, entry)
            XCTAssertEqual([reflected.x, reflected.y, reflected.z], size, entry)
            _ = try translate(source, entryPoints: [entry])
        }
    }

    func test_anOverrideWorkgroupSizeTakesThePipelineConstants() throws {
        // webgpu-samples' gameOfLife: `@workgroup_size(blockSize, blockSize)`.
        let source = """
        override blockSize: u32 = 8u;
        override depth: u32;
        @group(0) @binding(0) var<storage, read_write> out: array<u32>;

        @compute @workgroup_size(blockSize, blockSize)
        fn main(@builtin(global_invocation_id) id: vec3u) { out[id.x] = blockSize; }

        @compute @workgroup_size(4, 1, depth)
        fn deep(@builtin(global_invocation_id) id: vec3u) { out[id.x] = depth; }
        """
        let module = try WGSLShaderModule(source: source)

        // The default, then a supplied value.
        let byDefault = try XCTUnwrap(module.workgroupSize(of: "main"))
        XCTAssertEqual([byDefault.x, byDefault.y, byDefault.z], [8, 8, 1])
        XCTAssertEqual(module.reflection.entryPoint(named: "main")?.workgroupSize?.x, 8)
        let supplied = try XCTUnwrap(module.workgroupSize(of: "main", constants: ["blockSize": 4, "depth": 1]))
        XCTAssertEqual([supplied.x, supplied.y, supplied.z], [4, 4, 1])

        // No default: unknown until the pipeline supplies it — a clear error, never 1.
        XCTAssertNil(module.reflection.entryPoint(named: "deep")?.workgroupSize)
        XCTAssertThrowsError(try module.workgroupSize(of: "deep")) { error in
            let message = (error as? WGPUError)?.message ?? ""
            XCTAssertTrue(message.contains("override 'depth' has no value"), message)
            XCTAssertTrue(message.contains("@workgroup_size(4, 1, depth)"), message)
        }
        let deep = try XCTUnwrap(module.workgroupSize(of: "deep", constants: ["depth": 16]))
        XCTAssertEqual([deep.x, deep.y, deep.z], [4, 1, 16])

        let bindings = try WGSLBindingAssigner.assign(groups: module.autoBindGroupLayouts(entryPoints: ["main"]))
        let msl = try module.translateToMSL(
            entryPoints: ["main"], bindings: bindings, constants: ["blockSize": 4, "depth": 1]
        )
        MetalCompilerHarness.assertCompiles(msl)
    }

    func test_workgroupMemoryZeroInitStridesByTheEvaluatedSize() throws {
        // The zero-init loop strides by the thread count. Too large a stride leaves slots never zeroed,
        // so it must be the size the pipeline dispatches — constants applied.
        let source = """
        override blockSize: u32 = 8u;
        const ROWS = 0x2;
        @group(0) @binding(0) var<storage, read_write> out: array<f32>;
        var<workgroup> tile: array<f32, 256>;

        @compute @workgroup_size(blockSize, ROWS)
        fn main(@builtin(local_invocation_index) index: u32) {
            tile[index] = f32(index);
            workgroupBarrier();
            out[index] = tile[255u - index];
        }
        """
        let module = try WGSLShaderModule(source: source)
        let bindings = try WGSLBindingAssigner.assign(groups: module.autoBindGroupLayouts(entryPoints: ["main"]))
        for (constants, stride) in [([String: Double](), 16), (["blockSize": 32], 64), (["blockSize": 1], 2)] {
            let msl = try module.translateToMSL(entryPoints: ["main"], bindings: bindings, constants: constants)
            XCTAssertTrue(msl.contains("wgpu_zi += \(stride)u"), "\(constants) → stride \(stride):\n\(msl)")
            let size = try XCTUnwrap(module.workgroupSize(of: "main", constants: constants))
            XCTAssertEqual(size.x * size.y * size.z, stride, "the dispatch and the zero-init agree")
            MetalCompilerHarness.assertCompiles(msl)
        }
    }

    func test_aWorkgroupSizeThatIsNotAPositiveIntegerIsAShaderCreationError() {
        let cases: [(attribute: String, prelude: String, reason: String)] = [
            ("@workgroup_size(0)", "", "must be at least 1"),
            ("@workgroup_size(8, -1)", "", "must be at least 1"),
            ("@workgroup_size(4, 4, ZERO)", "const ZERO = 0u;", "must be at least 1"),
            ("@workgroup_size(1.5)", "", "not an integer"),
            ("@workgroup_size(0x)", "", "not a valid integer literal"),
            ("@workgroup_size(missing)", "", "not a module-scope const"),
            ("@workgroup_size()", "", "one to three arguments"),
            ("@workgroup_size(1, 1, 1, 1)", "", "one to three arguments"),
            ("@workgroup_size(1<<2)", "", "not supported"),
        ]
        for testCase in cases {
            let source = """
            \(testCase.prelude)
            @compute \(testCase.attribute) fn main() {}
            """
            XCTAssertThrowsError(try WGSLShaderModule(source: source), testCase.attribute) { error in
                let error = error as? WGPUError
                let message = error?.message ?? ""
                XCTAssertTrue(message.contains(testCase.attribute), "names the attribute: \(message)")
                XCTAssertTrue(message.contains(testCase.reason), "says why (\(testCase.reason)): \(message)")
                XCTAssertEqual(error?.line, 2, message)
            }
        }

        // A compute entry point must declare one — dispatching 1 thread silently is what this replaces.
        XCTAssertThrowsError(try WGSLShaderModule(source: "@compute fn main() {}")) { error in
            let message = (error as? WGPUError)?.message ?? ""
            XCTAssertTrue(message.contains("@compute — compute entry point 'main' has no @workgroup_size"), message)
        }

        // An override argument is left to pipeline creation, even when its default is 0.
        XCTAssertNoThrow(try WGSLShaderModule(source: """
        override size: u32 = 0u;
        @compute @workgroup_size(size) fn main() {}
        """))
    }

    func test_aWorkgroupSizeOutsideTheLimitsIsAPipelineCreationError() throws {
        let module = try WGSLShaderModule(source: """
        override wx: i32 = 64;
        override wy: u32 = 1u;
        @compute @workgroup_size(wx, wy) fn byOverride() {}
        @compute @workgroup_size(512) fn wide() {}
        @compute @workgroup_size(1, 1, 65) fn deep() {}
        @compute @workgroup_size(16, 32) fn many() {}
        @compute @workgroup_size(1024) fn large() {}
        """)
        func assertRejected(
            _ entry: String, constants: [String: Double] = [:], limits: WGSLWorkgroupLimits = .specDefaults,
            _ reason: String, line: UInt = #line
        ) {
            XCTAssertThrowsError(
                try module.workgroupSize(of: entry, constants: constants, limits: limits), entry, line: line
            ) { error in
                let error = error as? WGPUError
                XCTAssertEqual(error?.kind, .validation, line: line)
                XCTAssertTrue(error?.message.contains(reason) == true, "\(reason): \(error?.message ?? "")", line: line)
            }
        }

        // The spec's default limits: 256 × 256 × 64, 256 invocations.
        assertRejected("wide", "512 exceeds maxComputeWorkgroupSizeX (256)")
        assertRejected("deep", "65 exceeds maxComputeWorkgroupSizeZ (64)")
        assertRejected("many", "512 invocations exceed maxComputeInvocationsPerWorkgroup (256)")
        assertRejected("byOverride", constants: ["wy": 300], "300 exceeds maxComputeWorkgroupSizeY (256)")
        assertRejected("byOverride", constants: ["wx": 0], "must be at least 1")
        assertRejected("byOverride", constants: ["wx": -2], "must be at least 1")
        // `wx` is an i32, so 2.5 converts to 2 ([EnforceRange] truncates toward zero) — not an error.
        let truncated = try XCTUnwrap(module.workgroupSize(of: "byOverride", constants: ["wx": 2.5]))
        XCTAssertEqual([truncated.x, truncated.y, truncated.z], [2, 1, 1])

        // A device that advertises more lets more through — the backend passes its own limits.
        let generous = WGSLWorkgroupLimits(maxSizeX: 1024, maxSizeY: 1024, maxSizeZ: 1024, maxInvocations: 1024)
        let large = try XCTUnwrap(module.workgroupSize(of: "large", limits: generous))
        XCTAssertEqual([large.x, large.y, large.z], [1024, 1, 1])
        assertRejected("large", "1024 exceeds maxComputeWorkgroupSizeX (256)")
        let fits = try XCTUnwrap(module.workgroupSize(of: "byOverride", constants: ["wx": 16, "wy": 16]))
        XCTAssertEqual([fits.x, fits.y, fits.z], [16, 16, 1])
    }

    func test_anOverflowingWorkgroupSizeIsAnErrorInTranslationToo() throws {
        // `translateToMSL` applies no device limits — the zero-init stride must not trap on overflow.
        let module = try WGSLShaderModule(source: """
        var<workgroup> tile: array<u32, 4>;
        @group(0) @binding(0) var<storage, read_write> out: array<u32>;
        @compute @workgroup_size(4294967295, 4294967295, 4294967295)
        fn huge(@builtin(local_invocation_index) index: u32) { out[index] = tile[index % 4u]; }
        @compute @workgroup_size(65536, 65536)
        fn wide(@builtin(local_invocation_index) index: u32) { out[index] = tile[index % 4u]; }
        """)
        for entry in ["huge", "wide"] {
            let bindings = try WGSLBindingAssigner.assign(groups: module.autoBindGroupLayouts(entryPoints: [entry]))
            XCTAssertThrowsError(try module.translateToMSL(entryPoints: [entry], bindings: bindings), entry) { error in
                let error = error as? WGPUError
                XCTAssertEqual(error?.kind, .validation, entry)
                XCTAssertTrue(error?.message.contains("does not fit in u32") == true, error?.message ?? "")
            }
            XCTAssertThrowsError(try module.workgroupSize(of: entry), entry)
        }
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

    // MARK: - Expression height

    /// A flat chain is not nesting — the parser builds it in a loop — but the tree it builds is as tall
    /// as the chain is long, and the emitter walks it recursively. A hundred-thousand-term chain
    /// overflowed that walk (a SIGSEGV from shader text). So the finished tree's **height** is capped,
    /// for every chain shape, with a line-numbered validation error.
    func test_aFlatChainTallerThanTheHeightLimitIsAnErrorNotAStackOverflow() {
        let terms = WGSLParser.maximumExpressionHeight + 8
        let sources = [
            "fn f(a: f32) -> f32 {\n return " + Array(repeating: "a", count: terms).joined(separator: " + ") + ";\n}",
            "fn g(v: vec4f) -> f32 {\n return v" + String(repeating: ".x", count: terms) + ";\n}",
            "fn h(m: array<array<f32, 2>, 2>) -> f32 {\n return m" + String(repeating: "[0]", count: terms) + ";\n}",
            // An assignment target is an expression too.
            "fn k(v: vec4f) {\n var w = v; w" + String(repeating: ".x", count: terms) + " = 1.0;\n}",
        ]
        for source in sources {
            var failure: WGPUError?
            onSmallStack {
                do { _ = try WGSLShaderModule(source: source) } catch { failure = error as? WGPUError }
            }
            XCTAssertEqual(failure?.kind, .validation, String(source.prefix(40)))
            XCTAssertEqual(failure?.line, 2, String(source.prefix(40)))
            XCTAssertTrue(failure?.message.contains("deeper than") ?? false, "\(String(describing: failure))")
        }
    }

    /// Inside the limit, a long chain keeps translating — the cap is far above any real shader, and
    /// the translation runs on `WGSLDeepStack` regardless of the caller's stack.
    func test_aLongFlatChainInsideTheHeightLimitTranslates() {
        let terms = WGSLParser.maximumExpressionHeight - 8
        let source = """
        @fragment
        fn fs(@location(0) a: f32) -> @location(0) vec4f {
            let s = \(Array(repeating: "a", count: terms).joined(separator: " + "));
            return vec4f(s);
        }
        """
        let outcome = ThreadOutcome()
        onSmallStack {
            do {
                let module = try WGSLShaderModule(source: source)
                let bindings = try WGSLBindingAssigner.assign(groups: module.autoBindGroupLayouts(entryPoints: ["fs"]))
                outcome.msl = try module.translateToMSL(entryPoints: ["fs"], bindings: bindings)
            } catch {
                outcome.error = error
            }
        }
        XCTAssertNil(outcome.error, "\(String(describing: outcome.error))")
        XCTAssertNotNil(outcome.msl)
    }
}

/// What a worker thread produced — handed back through a semaphore, so the access is ordered.
private final class ThreadOutcome: @unchecked Sendable {
    var msl: String?
    var error: Error?
}
