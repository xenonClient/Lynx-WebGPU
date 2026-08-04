import XCTest
import LynxWebGPUCore
@testable import LynxWebGPUShader

/// WGSL → MSL 트랜스파일러의 핵심 경로.
///
/// 모든 케이스는 (1) 기대한 MSL 조각이 들어 있는지와 (2) **실제 Metal 컴파일러를 통과하는지**를
/// 함께 본다 — 문자열만 맞고 컴파일이 안 되는 회귀를 잡기 위해서다.
final class WGSLTranspilerTests: XCTestCase {
    // MARK: - 헬퍼

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

    // MARK: - 삼각형 (정점 속성 + 유니폼 + 헬퍼 함수)

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

    func test_삼각형셰이더_정점속성과_유니폼이_MSL로_번역된다() throws {
        let msl = try translate(Self.triangleShader, entryPoints: ["vs_main", "fs_main"])

        XCTAssertTrue(msl.contains("vertex wgpu_vs_main_out vs_main("), "vertex 진입점 래퍼가 없다")
        XCTAssertTrue(msl.contains("fragment wgpu_fs_main_out fs_main("), "fragment 진입점 래퍼가 없다")
        XCTAssertTrue(msl.contains("[[attribute(0)]]"), "정점 속성 0이 없다")
        XCTAssertTrue(msl.contains("[[attribute(1)]]"), "정점 속성 1이 없다")
        XCTAssertTrue(msl.contains("[[position]]"), "position 빌트인이 없다")
        XCTAssertTrue(msl.contains("[[color(0)]]"), "프래그먼트 출력 타깃이 없다")
        XCTAssertTrue(msl.contains("constant Uniforms& uniforms [[buffer(0)]]"), "유니폼 바인딩이 없다")
    }

    func test_헬퍼함수가_쓰는_유니폼은_인자로_전달된다() throws {
        let msl = try translate(Self.triangleShader, entryPoints: ["vs_main", "fs_main"])

        // MSL에는 가변 전역이 없으므로 헬퍼가 유니폼을 인자로 받아야 한다.
        XCTAssertTrue(
            msl.contains("float3 apply_tint(float3 c, constant Uniforms& uniforms)"),
            "헬퍼 함수에 리소스가 스레딩되지 않았다:\n\(msl)"
        )
        XCTAssertTrue(msl.contains("apply_tint(input.color, uniforms)"), "호출 측이 리소스를 넘기지 않았다")
    }

    func test_리플렉션이_진입점과_바인딩을_보고한다() throws {
        let module = try WGSLShaderModule(source: Self.triangleShader)

        XCTAssertEqual(module.reflection.entryPoints.map(\.name).sorted(), ["fs_main", "vs_main"])
        XCTAssertEqual(module.reflection.entryPoint(named: "vs_main")?.stage, .vertex)
        XCTAssertEqual(module.reflection.entryPoint(named: "fs_main")?.stage, .fragment)

        let resource = try XCTUnwrap(module.reflection.resource(named: "uniforms"))
        XCTAssertEqual(resource.group, 0)
        XCTAssertEqual(resource.binding, 0)
        XCTAssertEqual(resource.slotKind, .buffer)

        // vs_main만 uniforms를 쓰므로 visibility는 vertex뿐이다 (fs_main은 보간값만 읽는다).
        XCTAssertEqual(module.reflection.visibility(of: "uniforms"), .vertex)
    }

    // MARK: - 구조체 배치 (WGSL vs MSL의 vec3 차이)

    func test_vec3뒤_스칼라는_packed벡터와_패딩으로_WGSL배치를_맞춘다() throws {
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

        // direction(offset 0, 크기 12) 뒤에 intensity(offset 12)가 붙으므로 float3(16B)로는 자리가 안 맞는다.
        XCTAssertTrue(msl.contains("packed_float3 direction;"), "패킹된 vec3가 필요하다:\n\(msl)")
        XCTAssertTrue(msl.contains("float intensity;"))
        // color는 뒤가 비어 있으므로 일반 float3 + 꼬리 패딩.
        XCTAssertTrue(msl.contains("float3 color;"))
    }

    func test_구조체_배치가_WGSL_오프셋과_일치한다() throws {
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

    // MARK: - 컴퓨트 / 스토리지 버퍼

    func test_컴퓨트셰이더와_스토리지버퍼가_번역된다() throws {
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

        // `main`은 MSL에서 함수 이름으로 쓸 수 없어 방출 시 이름이 바뀐다.
        // 런타임이 makeFunction(name:)에 넘길 이름은 mslFunctionName(for:)이 알려준다.
        let mslName = WGSLShaderModule.mslFunctionName(for: "main")
        XCTAssertEqual(mslName, "wgpu_fn_main")
        XCTAssertTrue(msl.contains("kernel void \(mslName)("), "컴퓨트 진입점이 kernel로 나오지 않았다:\n\(msl)")
        XCTAssertTrue(msl.contains("const device float* input [[buffer(0)]]"), "읽기 전용 스토리지 버퍼")
        XCTAssertTrue(msl.contains("device float* output [[buffer(1)]]"), "읽기·쓰기 스토리지 버퍼")
        XCTAssertTrue(msl.contains("[[thread_position_in_grid]]"), "global_invocation_id 빌트인")

        let module = try WGSLShaderModule(source: source)
        let size = try XCTUnwrap(module.workgroupSize(of: "main"))
        XCTAssertEqual(size.x, 64)
        XCTAssertEqual(size.y, 1)
        XCTAssertEqual(size.z, 1)
    }

    // MARK: - 텍스처 / 샘플러

    func test_텍스처샘플링이_MSL_메서드호출로_바뀐다() throws {
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

    func test_스토리지텍스처_쓰기가_write호출로_바뀐다() throws {
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

    // MARK: - 제어 흐름

    func test_제어흐름_구문이_모두_번역된다() throws {
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

    func test_workgroup변수는_진입점_지역_threadgroup로_선언된다() throws {
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

        XCTAssertTrue(msl.contains("threadgroup array<float, 64> scratch;"), "workgroup 변수 선언이 없다:\n\(msl)")
        XCTAssertTrue(msl.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"))
    }

    // MARK: - 오류 처리

    func test_없는_진입점을_요청하면_오류다() throws {
        let module = try WGSLShaderModule(source: Self.triangleShader)
        XCTAssertThrowsError(try module.requireEntryPoint("missing", stage: .vertex)) { error in
            XCTAssertEqual((error as? WGPUError)?.kind, .validation)
        }
    }

    func test_스테이지가_다르면_오류다() throws {
        let module = try WGSLShaderModule(source: Self.triangleShader)
        XCTAssertThrowsError(try module.requireEntryPoint("fs_main", stage: .vertex))
    }

    func test_지원하지_않는_내장함수는_명시적으로_거부한다() throws {
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

    func test_문법오류는_줄번호와_함께_보고된다() throws {
        let source = """
        @vertex
        fn vs() -> @builtin(position) vec4f {
            return vec4f(1.0 1.0, 1.0, 1.0);
        }
        """
        XCTAssertThrowsError(try WGSLShaderModule(source: source)) { error in
            let message = (error as? WGPUError)?.message ?? ""
            XCTAssertTrue(message.contains("line 3"), "줄 번호가 없다: \(message)")
        }
    }

    // MARK: - 이름 충돌 / 리터럴 / 파이프라인 상수

    func test_MSL_예약어와_겹치는_식별자는_선언과_사용처가_함께_바뀐다() throws {
        // `texture` `sampler` `device` `char` 는 그래픽스 셰이더에서 흔한 이름이지만 MSL 예약어다.
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

        XCTAssertTrue(msl.contains("wgpu_id_texture"), "전역 이름이 바뀌지 않았다:\n\(msl)")
        XCTAssertTrue(msl.contains("wgpu_id_device"), "구조체 멤버 이름이 바뀌지 않았다")
        XCTAssertTrue(msl.contains("wgpu_id_char"))
        XCTAssertTrue(msl.contains("wgpu_id_sampler"), "지역 변수 이름이 바뀌지 않았다")
        // 선언과 사용처가 어긋나면 Metal 컴파일에서 잡힌다 (translate 헬퍼가 검증한다).
    }

    func test_부동소수_나머지연산은_fmod로_보내진다() throws {
        let source = """
        @fragment
        fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
            let f = uv.x % 2.0;
            let i = i32(uv.y) % 3;
            return vec4f(f, f32(i), 0.0, 1.0);
        }
        """
        let msl = try translate(source, entryPoints: ["fs_main"])
        XCTAssertTrue(msl.contains("wgpu_mod("), "% 가 헬퍼로 우회되지 않았다:\n\(msl)")
    }

    func test_성분타입이_생략된_벡터생성자는_인자에서_추론한다() throws {
        let source = """
        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
            let inferred = vec2(id.x, 4u);            // 인자에 타입이 있으면 정확히 추론
            let scaled: vec2u = id.xy * vec2(4, 1);   // AbstractInt 상수식 → 문맥(u32)에서 굳는다
            let asFloat: vec3f = vec3(1);             // 같은 상수식이 f32 문맥에서는 f32로
            let component = vec3(1, 2, 3).y;          // 성분 접근 자리에서는 구체 타입으로
            let combined = f32(scaled.x) + asFloat.y + component;
        }
        """
        let msl = try translate(source, entryPoints: ["main"])
        XCTAssertTrue(msl.contains("wgpu_vec2(id.x, 4u)"), "타입이 있으면 템플릿 추론:\n\(msl)")
        XCTAssertTrue(msl.contains("wgpu_aint2(4, 1)"), "AbstractInt 상수식은 프록시로:\n\(msl)")
        XCTAssertTrue(msl.contains("wgpu_aint3(1)"), "인자 하나여도 프록시로")
        // 프록시에는 성분 접근이 없다 — 그 자리에서는 f32 벡터로 확정한다.
        XCTAssertTrue(msl.contains("float3(1, 2, 3).y"), "스위즐 대상은 구체 타입으로:\n\(msl)")
    }

    func test_파이프라인_상수가_MSL에_박힌다() throws {
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

        // 값을 주지 않으면 무엇을 줘야 하는지 알려 주며 실패한다.
        XCTAssertThrowsError(try module.translateToMSL(entryPoints: ["main"], bindings: bindings)) { error in
            let message = (error as? WGPUError)?.message ?? ""
            XCTAssertTrue(message.contains("count"), "어떤 상수가 빠졌는지 알려 줘야 한다: \(message)")
            XCTAssertTrue(message.contains("constants"))
        }

        let msl = try module.translateToMSL(
            entryPoints: ["main"], bindings: bindings, constants: ["count": 8, "scale": 2.5]
        )
        // 배열 길이가 override에 걸린 경우까지 풀린다.
        XCTAssertTrue(msl.contains("array<float, 8>"), "override 배열 길이가 반영되지 않았다:\n\(msl)")
        XCTAssertTrue(msl.contains("2.5"))
        MetalCompilerHarness.assertCompiles(msl)
    }

    func test_확장선언은_무시된다() throws {
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

    // MARK: - 도달 가능성 (여럿이 나눠 쓰는 공용 모듈)

    func test_진입점이_부르지_않는_함수는_방출되지_않는다() throws {
        // `common.wgsl` 하나를 여러 셰이더가 나눠 쓰는 구성이 흔하다 (webgpu-samples의 cornell).
        // 그때 이 진입점이 안 쓰는 함수까지 내보내면, `layout: "auto"`의 바인드 그룹에 없는
        // 리소스를 그 함수가 찾다가 번역이 통째로 실패한다 — 실제로 깨졌던 경로다.
        let source = """
        struct Quad { color: vec4f };
        @group(0) @binding(0) var<uniform> uniforms: vec4f;
        @group(0) @binding(1) var<storage> quads: array<Quad>;

        // 이 진입점이 쓰지 않는 헬퍼 — 다른 셰이더가 쓰는 것이다.
        fn unused_helper() -> u32 {
            return arrayLength(&quads);
        }

        @fragment
        fn fs_main() -> @location(0) vec4f {
            return uniforms;
        }
        """
        let msl = try translate(source, entryPoints: ["fs_main"])

        XCTAssertFalse(msl.contains("unused_helper"), "안 쓰는 함수가 방출됐다:\n\(msl)")
        XCTAssertTrue(msl.contains("fs_main"), "진입점은 있어야 한다")
    }

    func test_전이적으로_닿는_함수는_방출된다() throws {
        // 도달 가능성은 **전이적**이다 — 진입점 → outer → inner 를 따라가야 한다.
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

        XCTAssertTrue(msl.contains("float outer("), "직접 호출한 함수가 없다:\n\(msl)")
        XCTAssertTrue(msl.contains("float inner("), "전이적으로 닿는 함수가 없다:\n\(msl)")
        XCTAssertFalse(msl.contains("orphan"), "아무도 안 부르는 함수가 방출됐다")
    }

    func test_진입점마다_필요한_함수가_다르면_각각_그것만_방출한다() throws {
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
        XCTAssertFalse(vertexOnly.contains("only_fragment"), "다른 진입점 전용 함수가 딸려 왔다")

        // 둘 다 요청하면 둘 다 나온다.
        let both = try translate(source, entryPoints: ["vs_main", "fs_main"])
        XCTAssertTrue(both.contains("only_vertex"))
        XCTAssertTrue(both.contains("only_fragment"))
    }

    // MARK: - 전역 섀도잉 (기계 생성 셰이더의 일상 패턴 — Three.js nodeVar0)

    func test_지역변수에_가려진_전역은_주입되지_않는다() throws {
        // Three.js 노드 시스템이 만드는 패턴: 같은 이름을 모듈 스코프와 함수 지역에 동시 생성한다.
        // helper는 지역 v만 쓰므로 모듈 스코프 v를 인자로 받으면 안 된다 — 받으면
        // `float4 v{}` 지역 선언과 재정의 충돌이 난다.
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
        XCTAssertTrue(msl.contains("float4 helper(float4 c)"), "가려진 전역이 주입됐다:\n\(msl)")
    }

    func test_매개변수에_가려진_전역은_주입되지_않는다() throws {
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
        XCTAssertTrue(msl.contains("float4 helper(float4 tint)"), "매개변수에 가려진 전역이 주입됐다:\n\(msl)")
    }

    func test_전역을_쓴_뒤_같은_이름의_지역을_선언하면_지역이_리네임된다() throws {
        // WGSL은 point-of-declaration 스코프라 선언 앞의 v는 전역, 뒤의 v는 지역이다.
        // 전역이 인자로 주입된 상태에서 같은 이름의 지역 선언은 C++ 재정의라 리네임이 필요하다.
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
        XCTAssertTrue(msl.contains("float wgpu_shadow_v = 3.0"), "지역 선언이 리네임되지 않았다:\n\(msl)")
        XCTAssertTrue(msl.contains("const auto before = v"), "선언 앞의 참조가 전역(주입 인자)을 봐야 한다:\n\(msl)")
    }

    func test_지역이_가린_전역도_호출_그래프를_따라_전달된다() throws {
        // outer는 전역 v를 직접 쓰지 않지만(지역이 가린다) inner가 쓰므로,
        // outer는 전역 v를 **전달만** 해야 한다 — 지역을 넘기면 조용히 틀린다.
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
        // outer 안의 지역은 리네임되고, inner 호출은 원래 이름(주입 인자 = 전역)을 넘긴다.
        XCTAssertTrue(msl.contains("float wgpu_shadow_v = 100.0"), "전달용 주입과 겹친 지역이 리네임되지 않았다:\n\(msl)")
        XCTAssertTrue(msl.contains("inner(v)"), "inner에는 전역(주입 인자)이 넘어가야 한다:\n\(msl)")
    }

    func test_중첩블록의_섀도잉은_블록을_벗어나면_풀린다() throws {
        // 블록 안 지역 선언의 리네임이 블록 밖으로 새면, 밖의 v 대입이 선언된 적 없는
        // 이름(wgpu_shadow_v)을 참조해 컴파일이 깨진다 — translate가 잡는다.
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

    // MARK: - 바인딩 배정

    func test_바인딩배정은_그룹_바인딩_순으로_결정적이다() throws {
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

        // 버퍼/텍스처/샘플러가 각각 독립된 인덱스 공간을 쓴다.
        XCTAssertEqual(bindings.index(group: 0, binding: 0), 0)   // buffer 0
        XCTAssertEqual(bindings.index(group: 0, binding: 2), 0)   // texture 0
        XCTAssertEqual(bindings.index(group: 1, binding: 0), 1)   // buffer 1
        XCTAssertEqual(bindings.index(group: 1, binding: 1), 0)   // sampler 0

        _ = try translate(source, entryPoints: ["fs"])
    }

    func test_정점버퍼_인덱스는_테이블_위쪽부터_역순배정된다() {
        XCTAssertEqual(WGSLMetalLimits.vertexBufferIndex(slot: 0), 30)
        XCTAssertEqual(WGSLMetalLimits.vertexBufferIndex(slot: 1), 29)
        // 바인드 그룹 버퍼 상한과 겹치지 않아야 한다.
        XCTAssertLessThan(
            WGSLMetalLimits.maxBindGroupBuffers,
            WGSLMetalLimits.vertexBufferIndex(slot: WGSLMetalLimits.maxVertexBufferSlots - 1) + 1
        )
        // 크기 표 인덱스는 양쪽 어디와도 겹치지 않는다.
        XCTAssertEqual(WGSLMetalLimits.bufferSizesIndex, 22)
        XCTAssertLessThan(
            WGSLMetalLimits.bufferSizesIndex,
            WGSLMetalLimits.vertexBufferIndex(slot: WGSLMetalLimits.maxVertexBufferSlots - 1)
        )
    }

    // MARK: - arrayLength

    func test_런타임_크기_배열의_길이는_크기표_조회로_번역된다() throws {
        let source = """
        @group(0) @binding(0) var<storage, read> data: array<f32>;
        @group(0) @binding(1) var<storage, read_write> out: array<u32>;

        @compute @workgroup_size(1)
        fn cs(@builtin(global_invocation_id) id: vec3u) {
            out[0] = arrayLength(&data);
        }
        """
        let msl = try translate(source, entryPoints: ["cs"])

        // 예약 인덱스로 크기 표가 들어온다.
        XCTAssertTrue(
            msl.contains("constant uint* wgpu_buffer_sizes [[buffer(\(WGSLMetalLimits.bufferSizesIndex))]]"),
            msl
        )
        // 길이 = 바인딩 바이트 수 / 원소 크기. data는 버퍼 인덱스 0.
        XCTAssertTrue(msl.contains("(wgpu_buffer_sizes[0] / uint(sizeof(float)))"), msl)
    }

    func test_구조체_말미의_런타임_배열은_앞쪽_멤버_크기를_빼고_센다() throws {
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

        // items는 16바이트 정렬이므로 오프셋 16에서 시작한다 (count 4B + 패딩 12B).
        XCTAssertTrue(msl.contains("((wgpu_buffer_sizes[0] - 16u) / uint(sizeof(float4)))"), msl)
    }

    func test_arrayLength를_쓰지_않으면_크기표를_넘기지_않는다() throws {
        let msl = try translate(Self.triangleShader, entryPoints: ["vs_main", "fs_main"])
        XCTAssertFalse(msl.contains("wgpu_buffer_sizes"), msl)
    }

    // MARK: - 컴파일 타임 상수

    func test_함수_안에서_선언한_const도_배열_크기로_쓸_수_있다() throws {
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

    func test_같은_이름의_지역_const가_함수마다_다르면_배열_크기로_쓰지_않는다() throws {
        // 배열 크기를 정할 때는 함수 문맥이 없으므로, 값이 하나로 정해지지 않으면 거부해야 한다.
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

    // MARK: - 외부 텍스처

    func test_외부_텍스처는_가장자리_클램프_샘플링으로_번역된다() throws {
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
        // texture_external은 샘플링 가능한 2D 텍스처로 내려간다.
        XCTAssertTrue(msl.contains("texture2d<float>"), msl)
    }
}
