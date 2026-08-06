import XCTest
import LynxWebGPUCore
@testable import LynxWebGPUShader

/// **WGSL이 정의하는데 MSL(C++)에서는 정의되지 않은 자리**를 메우는 변환들.
///
/// 전부 명세가 요구하는 동작이고, 없으면 드라이버가 무엇을 하든 이상하지 않다 —
/// 인접 메모리를 읽거나 덮어쓰고, 최적화가 "일어날 수 없는 일"로 보고 주변 코드를 지우기도 한다.
/// Tint(Dawn의 WGSL 컴파일러)가 같은 이름의 변환들로 하는 일이다.
///
/// 각 케이스는 (1) 기대한 MSL 조각과 (2) **실제 Metal 컴파일러 통과**를 함께 본다.
final class ShaderSafetyTests: XCTestCase {
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

    // MARK: - 인덱싱 범위 (robustness)

    /// 고정 크기 배열은 **타입이 크기를 안다** — C++ 템플릿이 상한을 뽑는다.
    func test_고정크기_배열_인덱싱이_범위로_잘린다() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<storage, read> data: array<f32, 8>;
        @group(0) @binding(1) var<uniform> idx: u32;
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(data[idx], 0, 0, 1);
        }
        """, entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("wgpu_at(data, idx)"), "인덱싱이 클램프를 지나지 않는다\n\(msl)")
    }

    /// 런타임 크기 배열은 타입에 크기가 없다 — **버퍼 크기 표**로 상한을 구한다.
    /// 인덱스는 유니폼에서 오므로 결국 번들(JS)이 정하는 값이다.
    func test_런타임크기_배열_인덱싱이_버퍼_크기로_잘린다() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;
        @group(0) @binding(1) var<uniform> idx: u32;
        @compute @workgroup_size(1) fn cs() {
            data[idx] = 1.0;
        }
        """, entryPoints: ["cs"])
        XCTAssertTrue(msl.contains("wgpu_store_n(data, idx"), "런타임 배열 쓰기가 클램프를 지나지 않는다\n\(msl)")
        XCTAssertTrue(msl.contains("wgpu_buffer_sizes"), "상한을 구할 크기 표가 없다\n\(msl)")
    }

    /// 크기 표를 요구했으면 **파이프라인도 같은 답을 봐야 한다** — 어긋나면 셰이더가
    /// 바인딩되지 않은 버퍼를 읽는다.
    func test_런타임배열_인덱싱만_해도_크기표가_필요하다고_보고한다() throws {
        let module = try WGSLShaderModule(source: """
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;
        @compute @workgroup_size(1) fn cs() { data[0] = 1.0; }
        """)
        XCTAssertTrue(
            module.usesArrayLength(entryPoints: ["cs"]),
            "arrayLength()를 안 써도 인덱싱하면 표가 필요하다"
        )
    }

    /// 벡터 성분은 참조로 묶을 수 없어(MSL 제약) 읽기는 값, 쓰기는 store 헬퍼로 간다.
    func test_벡터_성분_인덱싱도_읽기_쓰기_모두_잘린다() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<uniform> idx: u32;
        @fragment fn fs() -> @location(0) vec4f {
            var v = vec4f(1, 2, 3, 4);
            v[idx] = 9.0;
            return vec4f(v[idx], 0, 0, 1);
        }
        """, entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("wgpu_store(v, idx"), "벡터 쓰기가 클램프를 지나지 않는다\n\(msl)")
        XCTAssertTrue(msl.contains("wgpu_at(v, idx)"), "벡터 읽기가 클램프를 지나지 않는다\n\(msl)")
    }

    func test_행렬_열_인덱싱과_중첩_인덱싱이_컴파일된다() throws {
        _ = try translate("""
        @group(0) @binding(0) var<uniform> idx: u32;
        @fragment fn fs() -> @location(0) vec4f {
            var m = mat4x4f();
            m[idx][idx] = 1.0;
            var grid = array<array<f32, 4>, 4>();
            grid[idx][idx] = 2.0;
            return vec4f(m[idx][idx] + grid[idx][idx], 0, 0, 1);
        }
        """, entryPoints: ["fs"])
    }

    /// 복합 대입(`+=`)도 같은 경로를 지나야 한다 — 한쪽만 막으면 그 경로로 샌다.
    func test_복합_대입도_클램프를_지난다() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<storage, read_write> data: array<f32, 4>;
        @group(0) @binding(1) var<uniform> idx: u32;
        @compute @workgroup_size(1) fn cs() {
            data[idx] += 1.0;
        }
        """, entryPoints: ["cs"])
        XCTAssertTrue(msl.contains("wgpu_store(data, idx"), "복합 대입이 클램프를 지나지 않는다\n\(msl)")
    }

    // MARK: - 정수 나눗셈 · 시프트 · 변환

    /// WGSL은 `x / 0 == x`, `x % 0 == 0`으로 정의한다. C++에서는 UB다.
    func test_정수_나눗셈과_나머지가_0_분모를_거른다() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<uniform> d: i32;
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(f32(100 / d), f32(100 % d), 0, 1);
        }
        """, entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("wgpu_div(100, d)"), "나눗셈에 가드가 없다\n\(msl)")
        XCTAssertTrue(msl.contains("wgpu_mod(100, d)"), "나머지에 가드가 없다\n\(msl)")
    }

    /// 폭 이상의 시프트는 C++에서 UB다 — 하위 5비트만 남긴다 (Tint와 같은 규칙).
    func test_시프트_폭이_마스킹된다() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<uniform> s: u32;
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(f32(1u << s), f32(256u >> s), 0, 1);
        }
        """, entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("wgpu_shl(1u, s)"), "왼쪽 시프트가 마스킹되지 않는다\n\(msl)")
        XCTAssertTrue(msl.contains("wgpu_shr(256u, s)"), "오른쪽 시프트가 마스킹되지 않는다\n\(msl)")
    }

    /// WGSL은 범위 밖 float → int 변환을 **포화**로 정의한다. C++ 캐스트는 UB다.
    func test_float에서_정수_변환이_포화한다() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<uniform> f: f32;
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(f32(i32(f)), f32(u32(f)), f32(vec2i(f).x), 1);
        }
        """, entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("wgpu_ftoi(f)"), "i32() 변환이 포화하지 않는다\n\(msl)")
        XCTAssertTrue(msl.contains("wgpu_ftou(f)"), "u32() 변환이 포화하지 않는다\n\(msl)")
        XCTAssertTrue(msl.contains("wgpu_ftoi_n<2>(f)"), "벡터 변환이 포화하지 않는다\n\(msl)")
    }

    /// 정수 벡터를 다른 정수 벡터로 옮기는 것은 포화와 무관하다 — 좁히려다 깨지면 안 된다.
    func test_정수_벡터_변환은_그대로_지나간다() throws {
        _ = try translate("""
        @group(0) @binding(0) var<uniform> v: vec2u;
        @fragment fn fs() -> @location(0) vec4f {
            let signed = vec2i(v);
            return vec4f(f32(signed.x), 0, 0, 1);
        }
        """, entryPoints: ["fs"])
    }

    // MARK: - workgroup 0 초기화

    /// WGSL은 `var<workgroup>`의 0 초기화를 보장한다. MSL의 threadgroup 저장소는 아니다 —
    /// 없으면 **이전 디스패치의 잔여 값**을 읽는다.
    func test_workgroup_변수가_0으로_깔리고_배리어가_붙는다() throws {
        let msl = try translate("""
        var<workgroup> tile: array<f32, 64>;
        var<workgroup> total: f32;
        @group(0) @binding(0) var<storage, read_write> out: array<f32>;
        @compute @workgroup_size(64)
        fn cs(@builtin(local_invocation_id) lid: vec3u) {
            out[lid.x] = tile[lid.x] + total;
        }
        """, entryPoints: ["cs"])
        XCTAssertTrue(msl.contains("wgpu_zi"), "배열이 0으로 깔리지 않는다\n\(msl)")
        XCTAssertTrue(msl.contains("total = float{}"), "스칼라가 0으로 깔리지 않는다\n\(msl)")
        XCTAssertTrue(
            msl.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"),
            "배리어가 없으면 다른 스레드가 아직 안 깔린 자리를 읽는다\n\(msl)"
        )
    }

    /// 이미 `local_invocation_index`를 받는 셰이더에 **같은 속성이 두 번** 붙으면 안 된다.
    func test_이미_받고_있는_인덱스_빌트인은_중복되지_않는다() throws {
        let msl = try translate("""
        var<workgroup> tile: array<f32, 8>;
        @group(0) @binding(0) var<storage, read_write> out: array<f32>;
        @compute @workgroup_size(8)
        fn cs(@builtin(local_invocation_index) i: u32) {
            out[i] = tile[i];
        }
        """, entryPoints: ["cs"])
        let occurrences = msl.components(separatedBy: "[[thread_index_in_threadgroup]]").count - 1
        XCTAssertEqual(occurrences, 1, "같은 빌트인이 두 번 붙었다\n\(msl)")
    }

    func test_workgroup_원자변수는_store로_0이_된다() throws {
        let msl = try translate("""
        var<workgroup> counter: atomic<u32>;
        @group(0) @binding(0) var<storage, read_write> out: array<u32>;
        @compute @workgroup_size(4)
        fn cs(@builtin(local_invocation_id) lid: vec3u) {
            atomicAdd(&counter, 1u);
            workgroupBarrier();
            out[lid.x] = atomicLoad(&counter);
        }
        """, entryPoints: ["cs"])
        XCTAssertTrue(
            msl.contains("atomic_store_explicit(&counter, 0, memory_order_relaxed)"),
            "원자 변수는 대입이 아니라 store로만 0이 된다\n\(msl)"
        )
    }

    // MARK: - discard 이후 쓰기

    /// MSL의 `discard_fragment()`는 즉시 종료가 아니다 — 뒤의 코드가 계속 돌아
    /// **버려진 프래그먼트가 스토리지를 오염시킨다.**
    func test_discard_뒤의_스토리지_쓰기가_가려진다() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<storage, read_write> out: array<f32>;
        @group(0) @binding(1) var<uniform> cutoff: f32;
        @fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
            if (p.x > cutoff) { discard; }
            out[0] = 1.0;
            return vec4f(1);
        }
        """, entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("wgpu_discarded = true"), "discard가 플래그로 바뀌지 않았다\n\(msl)")
        XCTAssertTrue(msl.contains("if (!wgpu_discarded)"), "쓰기가 가려지지 않았다\n\(msl)")
        XCTAssertTrue(
            msl.contains("if (wgpu_discarded) { discard_fragment(); }"),
            "실제 폐기가 진입점 끝에서 일어나야 한다 (같은 쿼드의 미분값이 살아 있어야 하므로)\n\(msl)"
        )
    }

    /// `discard`가 없는 셰이더에는 **아무 비용도 붙지 않아야 한다.**
    func test_discard가_없으면_플래그를_만들지_않는다() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<storage, read_write> out: array<f32>;
        @fragment fn fs() -> @location(0) vec4f {
            out[0] = 1.0;
            return vec4f(1);
        }
        """, entryPoints: ["fs"])
        XCTAssertFalse(msl.contains("wgpu_discarded"), "discard가 없는데 플래그가 생겼다\n\(msl)")
    }

    /// 헬퍼 함수 안의 `discard`도 같은 플래그를 봐야 한다 — 플래그가 호출 그래프를 따라 내려간다.
    func test_헬퍼_함수의_discard도_같은_플래그를_쓴다() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<storage, read_write> out: array<f32>;
        fn maybeDiscard(x: f32) { if (x > 0.5) { discard; } }
        @fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
            maybeDiscard(p.x);
            out[0] = 1.0;
            return vec4f(1);
        }
        """, entryPoints: ["fs"])
        XCTAssertTrue(
            msl.contains("thread bool& wgpu_discarded"),
            "플래그가 헬퍼로 내려가지 않는다\n\(msl)"
        )
    }

    // MARK: - 무한 루프

    /// C++에서 무한 루프는 UB다 — 컴파일러가 "끝난다"고 가정하고 주변 코드를 지울 수 있다.
    func test_무한_루프에_탈출_조건이_하나_더_붙는다() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<uniform> n: u32;
        @fragment fn fs() -> @location(0) vec4f {
            var i = 0u;
            loop {
                if (i >= n) { break; }
                i = i + 1u;
            }
            return vec4f(f32(i), 0, 0, 1);
        }
        """, entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("wgpu_loop_guard_1"), "무한 루프에 가드가 없다\n\(msl)")
        XCTAssertTrue(msl.contains("break;"), "가드가 탈출로 이어지지 않는다\n\(msl)")
    }

    /// 조건이 있는 `for`/`while`은 컴파일러가 끝난다고 볼 근거가 있다 — 가드를 붙이지 않는다.
    func test_조건이_있는_루프에는_가드가_붙지_않는다() throws {
        let msl = try translate("""
        @fragment fn fs() -> @location(0) vec4f {
            var total = 0.0;
            for (var i = 0u; i < 4u; i = i + 1u) { total = total + 1.0; }
            return vec4f(total, 0, 0, 1);
        }
        """, entryPoints: ["fs"])
        XCTAssertFalse(msl.contains("wgpu_loop_guard"), "조건이 있는 루프에 불필요한 가드가 붙었다\n\(msl)")
    }

    func test_중첩된_무한_루프의_가드_이름이_겹치지_않는다() throws {
        let msl = try translate("""
        @group(0) @binding(0) var<uniform> n: u32;
        @fragment fn fs() -> @location(0) vec4f {
            var i = 0u;
            loop {
                if (i >= n) { break; }
                var j = 0u;
                loop {
                    if (j >= n) { break; }
                    j = j + 1u;
                }
                i = i + 1u;
            }
            return vec4f(f32(i), 0, 0, 1);
        }
        """, entryPoints: ["fs"])
        XCTAssertTrue(msl.contains("wgpu_loop_guard_1"), "\(msl)")
        XCTAssertTrue(msl.contains("wgpu_loop_guard_2"), "중첩 루프의 가드 이름이 겹친다\n\(msl)")
    }
}
