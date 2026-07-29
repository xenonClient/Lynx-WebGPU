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
            let n = arrayLength(&data);
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
    }
}
