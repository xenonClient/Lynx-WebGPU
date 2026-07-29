import Foundation

/// 생성된 MSL 맨 앞에 붙는 헬퍼 모음.
///
/// **왜 필요한가** — 이 트랜스파일러는 타입 추론기가 아니라 구문 번역기다. 그래서 WGSL이
/// 타입으로 결정하는 것들(성분 타입이 생략된 `vec2(a, b)`, `max(x, 0)`의 리터럴 승격,
/// 부동소수 `%`)을 그대로 옮기면 MSL에서 타입 오류가 나거나 **조용히 틀린 타입**이 된다.
///
/// C++ 템플릿의 `decltype`에 그 판단을 넘기면 타입 추론 없이도 WGSL과 같은 결과가 나온다.
/// 인스턴스화되지 않은 템플릿은 코드를 만들지 않으므로 비용도 없다.
enum MSLPrelude {
    static let source = """
    // ─── Lynx-WebGPU 셰이더 프렐류드 ───────────────────────────────────────
    // WGSL과 MSL의 의미가 어긋나는 지점만 메운다. 자세한 배경은 docs/WGSL.md §1.

    // WGSL의 `%`는 부동소수에도 정의되지만(fmod), MSL의 `%`는 정수 전용이다.
    // 먼저 두 인자의 공통 타입으로 맞춘 뒤(정수/부동소수 혼합 대응) 종류별 구현으로 보낸다.
    inline int wgpu_mod_same(int a, int b) { return a % b; }
    inline uint wgpu_mod_same(uint a, uint b) { return a % b; }
    inline int2 wgpu_mod_same(int2 a, int2 b) { return a % b; }
    inline int3 wgpu_mod_same(int3 a, int3 b) { return a % b; }
    inline int4 wgpu_mod_same(int4 a, int4 b) { return a % b; }
    inline uint2 wgpu_mod_same(uint2 a, uint2 b) { return a % b; }
    inline uint3 wgpu_mod_same(uint3 a, uint3 b) { return a % b; }
    inline uint4 wgpu_mod_same(uint4 a, uint4 b) { return a % b; }
    template<typename T> inline T wgpu_mod_same(T a, T b) { return fmod(a, b); }
    template<typename A, typename B> inline auto wgpu_mod(A a, B b) {
        using T = decltype(a + b);
        return wgpu_mod_same(T(a), T(b));
    }

    // MSL에는 radians/degrees가 없다.
    template<typename T> inline T wgpu_radians(T d) { return d * T(0.017453292519943295); }
    template<typename T> inline T wgpu_degrees(T r) { return r * T(57.29577951308232); }

    // 성분 타입이 생략된 벡터 생성자 — 인자에서 타입을 추론한다 (`vec2(u, 4)` → uint2).
    template<typename T> inline vec<T, 2> wgpu_vec2(vec<T, 2> a) { return a; }
    template<typename A> inline vec<A, 2> wgpu_vec2(A a) { return vec<A, 2>(a); }
    template<typename A, typename B> inline auto wgpu_vec2(A a, B b) {
        return vec<decltype(a + b), 2>(a, b);
    }
    template<typename T> inline vec<T, 3> wgpu_vec3(vec<T, 3> a) { return a; }
    template<typename A> inline vec<A, 3> wgpu_vec3(A a) { return vec<A, 3>(a); }
    template<typename T, typename B> inline vec<T, 3> wgpu_vec3(vec<T, 2> a, B b) {
        return vec<T, 3>(a, T(b));
    }
    template<typename T, typename A> inline vec<T, 3> wgpu_vec3(A a, vec<T, 2> b) {
        return vec<T, 3>(T(a), b);
    }
    template<typename A, typename B, typename C> inline auto wgpu_vec3(A a, B b, C c) {
        return vec<decltype(a + b + c), 3>(a, b, c);
    }
    template<typename T> inline vec<T, 4> wgpu_vec4(vec<T, 4> a) { return a; }
    template<typename A> inline vec<A, 4> wgpu_vec4(A a) { return vec<A, 4>(a); }
    template<typename T, typename B> inline vec<T, 4> wgpu_vec4(vec<T, 3> a, B b) {
        return vec<T, 4>(a, T(b));
    }
    template<typename T, typename U> inline vec<T, 4> wgpu_vec4(vec<T, 2> a, vec<U, 2> b) {
        return vec<T, 4>(a, vec<T, 2>(b));
    }
    template<typename T, typename A> inline vec<T, 4> wgpu_vec4(A a, vec<T, 3> b) {
        return vec<T, 4>(T(a), b);
    }
    template<typename T, typename B, typename C> inline vec<T, 4> wgpu_vec4(vec<T, 2> a, B b, C c) {
        return vec<T, 4>(a, T(b), T(c));
    }
    template<typename T, typename A, typename C> inline vec<T, 4> wgpu_vec4(A a, vec<T, 2> b, C c) {
        return vec<T, 4>(T(a), b, T(c));
    }
    template<typename T, typename A, typename B> inline vec<T, 4> wgpu_vec4(A a, B b, vec<T, 2> c) {
        return vec<T, 4>(T(a), T(b), c);
    }
    template<typename A, typename B, typename C, typename D> inline auto wgpu_vec4(A a, B b, C c, D d) {
        return vec<decltype(a + b + c + d), 4>(a, b, c, d);
    }

    // WGSL은 리터럴을 문맥 타입으로 승격한다(`max(x, 0)`). MSL은 그 자리에서 오버로드가 갈린다.
    template<typename A, typename B> inline auto wgpu_max(A a, B b) {
        using T = decltype(a + b);
        return max(T(a), T(b));
    }
    template<typename A, typename B> inline auto wgpu_min(A a, B b) {
        using T = decltype(a + b);
        return min(T(a), T(b));
    }
    template<typename A, typename B, typename C> inline auto wgpu_clamp(A a, B b, C c) {
        using T = decltype(a + b + c);
        return clamp(T(a), T(b), T(c));
    }
    template<typename A, typename B, typename C> inline auto wgpu_mix(A a, B b, C c) {
        using T = decltype(a + b);
        return mix(T(a), T(b), c);
    }
    template<typename A, typename B> inline auto wgpu_pow(A a, B b) {
        using T = decltype(a + b);
        return pow(T(a), T(b));
    }
    template<typename A, typename B> inline auto wgpu_step(A a, B b) {
        using T = decltype(a + b);
        return step(T(a), T(b));
    }
    template<typename A, typename B, typename C> inline auto wgpu_smoothstep(A a, B b, C c) {
        using T = decltype(a + b + c);
        return smoothstep(T(a), T(b), T(c));
    }
    // ──────────────────────────────────────────────────────────────────────
    """

    /// 프렐류드 헬퍼로 우회시키는 내장 함수 (WGSL 이름 → MSL 헬퍼 이름).
    static let redirectedBuiltins: [String: String] = [
        "max": "wgpu_max",
        "min": "wgpu_min",
        "clamp": "wgpu_clamp",
        "mix": "wgpu_mix",
        "pow": "wgpu_pow",
        "step": "wgpu_step",
        "smoothstep": "wgpu_smoothstep",
        "radians": "wgpu_radians",
        "degrees": "wgpu_degrees",
    ]

    /// 성분 타입이 생략된 벡터 생성자.
    static let inferredVectorConstructors: Set<String> = ["vec2", "vec3", "vec4"]
}
