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

    // 정수 나눗셈·나머지의 **정의되지 않은 자리**를 메운다.
    //
    // WGSL은 `x / 0 == x`, `x % 0 == 0`으로 정의하고, `INT_MIN / -1 == INT_MIN`,
    // `INT_MIN % -1 == 0`도 정의한다. C++에서는 셋 다 **정의되지 않은 동작**이라
    // 그대로 두면 GPU 드라이버가 무엇을 하든 이상하지 않다.
    //
    // 두 경우 모두 **나누는 수를 1로 바꾸면** WGSL이 정한 값이 그대로 나온다
    // (`a / 1 == a`, `a % 1 == 0`). 그래서 분모만 고른다.
    template<typename A, typename B> inline B wgpu_denom(A, B b) { return b; }   // 부동소수·추상 정수는 그대로
    inline int wgpu_denom(int a, int b) {
        return (b == 0 || (b == -1 && a == (-2147483647 - 1))) ? 1 : b;
    }
    inline uint wgpu_denom(uint, uint b) { return b == 0u ? 1u : b; }
    template<int N> inline vec<int, N> wgpu_denom(vec<int, N> a, vec<int, N> b) {
        return select(b, vec<int, N>(1), (b == 0) || ((b == -1) && (a == (-2147483647 - 1))));
    }
    template<int N> inline vec<uint, N> wgpu_denom(vec<uint, N>, vec<uint, N> b) {
        return select(b, vec<uint, N>(1u), b == 0u);
    }

    template<typename A, typename B> inline auto wgpu_mod(A a, B b) {
        using T = decltype(a + b);
        return wgpu_mod_same(T(a), wgpu_denom(T(a), T(b)));
    }
    template<typename A, typename B> inline auto wgpu_div(A a, B b) {
        using T = decltype(a + b);
        return T(a) / wgpu_denom(T(a), T(b));
    }

    // 시프트 폭 마스킹 — WGSL은 폭 이상의 시프트를 정의하지 않고, C++에서는 UB다.
    // Tint와 같은 규칙으로 하위 5비트만 남긴다 (i32·u32 모두 32비트다).
    inline uint wgpu_shift_amount(uint b) { return b & 31u; }
    inline int wgpu_shift_amount(int b) { return b & 31; }
    template<int N> inline vec<uint, N> wgpu_shift_amount(vec<uint, N> b) { return b & 31u; }
    template<int N> inline vec<int, N> wgpu_shift_amount(vec<int, N> b) { return b & 31; }
    template<typename T> inline T wgpu_shift_amount(T b) { return b; }
    template<typename A, typename B> inline auto wgpu_shl(A a, B b) { return a << wgpu_shift_amount(b); }
    template<typename A, typename B> inline auto wgpu_shr(A a, B b) { return a >> wgpu_shift_amount(b); }

    // f32 → i32/u32 변환은 WGSL에서 **포화**한다 (범위를 벗어나면 끝값으로 자른다).
    // C++의 캐스트는 범위 밖이면 UB다. 비교가 전부 거짓이 되는 NaN은 상한으로 간다 — Tint와 같다.
    inline int wgpu_ftoi(float v) {
        return select(2147483647, select(int(v), (-2147483647 - 1), (v < -2147483648.0f)), (v < 2147483520.0f));
    }
    inline uint wgpu_ftou(float v) {
        return select(4294967295u, select(uint(v), 0u, (v < 0.0f)), (v < 4294967040.0f));
    }
    template<int N> inline vec<int, N> wgpu_ftoi(vec<float, N> v) {
        return select(vec<int, N>(2147483647),
                      select(vec<int, N>(v), vec<int, N>(-2147483647 - 1), (v < -2147483648.0f)),
                      (v < 2147483520.0f));
    }
    template<int N> inline vec<uint, N> wgpu_ftou(vec<float, N> v) {
        return select(vec<uint, N>(4294967295u),
                      select(vec<uint, N>(v), vec<uint, N>(0u), (v < 0.0f)),
                      (v < 4294967040.0f));
    }
    // f32가 아닌 것(정수·bool·f16·추상 정수)은 그냥 변환한다 — 범위를 벗어날 수 없다.
    template<typename T> inline int wgpu_ftoi(T v) { return int(v); }
    template<typename T> inline uint wgpu_ftou(T v) { return uint(v); }
    // 벡터 자리의 변환 — 크기를 명시해 스칼라 브로드캐스트(`vec2i(1.5)`)와 성분 변환을 함께 받는다.
    // 마지막 대역은 **그냥 생성**한다: f32가 아닌 벡터(`vec2i(someU32Vec)`)까지 성분 변환을
    // 거치게 하면 벡터를 스칼라로 좁히려다 컴파일이 깨진다.
    template<int N> inline vec<int, N> wgpu_ftoi_n(vec<float, N> v) { return wgpu_ftoi(v); }
    template<int N> inline vec<int, N> wgpu_ftoi_n(float v) { return vec<int, N>(wgpu_ftoi(v)); }
    template<int N, typename T> inline vec<int, N> wgpu_ftoi_n(T v) { return vec<int, N>(v); }
    template<int N> inline vec<uint, N> wgpu_ftou_n(vec<float, N> v) { return wgpu_ftou(v); }
    template<int N> inline vec<uint, N> wgpu_ftou_n(float v) { return vec<uint, N>(wgpu_ftou(v)); }
    template<int N, typename T> inline vec<uint, N> wgpu_ftou_n(T v) { return vec<uint, N>(v); }

    // 인덱싱 범위 클램프 — WebGPU 명세가 요구하는 robustness.
    //
    // 범위를 벗어난 접근은 같은 프로세스의 **인접 GPU 메모리를 읽거나 덮어쓰고**, 크게 벗어나면
    // 페이지 폴트로 커맨드 버퍼가 죽는다. 인덱스는 보통 유니폼·스토리지에서 오므로 결국
    // **번들(JS)이 정하는 값**이다 — 서버에서 내려받는 번들을 전제로 하는 환경에서 특히 중요하다.
    //
    // 크기는 C++이 안다 (`array<T,N>`·`vec<T,N>`). 주소 공간마다 참조 타입이 갈려 오버로드가 는다.
    // 참조를 돌려주므로 읽기·쓰기·`&a[i]` 모두 원래대로 쓸 수 있고, 인덱스는 한 번만 평가된다.
    template<typename T, size_t N, typename I> inline thread T& wgpu_at(thread array<T, N>& a, I i) { return a[min(uint(i), uint(N - 1))]; }
    template<typename T, size_t N, typename I> inline const thread T& wgpu_at(const thread array<T, N>& a, I i) { return a[min(uint(i), uint(N - 1))]; }
    template<typename T, size_t N, typename I> inline device T& wgpu_at(device array<T, N>& a, I i) { return a[min(uint(i), uint(N - 1))]; }
    template<typename T, size_t N, typename I> inline const device T& wgpu_at(const device array<T, N>& a, I i) { return a[min(uint(i), uint(N - 1))]; }
    template<typename T, size_t N, typename I> inline constant T& wgpu_at(constant array<T, N>& a, I i) { return a[min(uint(i), uint(N - 1))]; }
    template<typename T, size_t N, typename I> inline threadgroup T& wgpu_at(threadgroup array<T, N>& a, I i) { return a[min(uint(i), uint(N - 1))]; }
    template<typename T, size_t N, typename I> inline const threadgroup T& wgpu_at(const threadgroup array<T, N>& a, I i) { return a[min(uint(i), uint(N - 1))]; }

    // 벡터 성분만은 **값으로** 돌려준다. MSL에서 `v[i]`는 참조로 묶을 수 없는 자리라
    // (`non-const reference cannot bind to vector element`) 참조를 돌려주면 컴파일이 깨진다.
    // 쓰기는 아래 `wgpu_store`가 맡는다.
    // `thread` 벡터는 **const 참조 하나만** 둔다 — 값 받기와 const 참조를 함께 두면
    // 비-const 좌변값에서 둘 다 똑같이 맞아 호출이 모호해진다.
    template<typename T, int N, typename I> inline T wgpu_at(const thread vec<T, N>& v, I i) { return v[min(uint(i), uint(N - 1))]; }
    template<typename T, int N, typename I> inline T wgpu_at(const device vec<T, N>& v, I i) { return v[min(uint(i), uint(N - 1))]; }
    template<typename T, int N, typename I> inline T wgpu_at(device vec<T, N>& v, I i) { return v[min(uint(i), uint(N - 1))]; }
    template<typename T, int N, typename I> inline T wgpu_at(constant vec<T, N>& v, I i) { return v[min(uint(i), uint(N - 1))]; }
    template<typename T, int N, typename I> inline T wgpu_at(threadgroup vec<T, N>& v, I i) { return v[min(uint(i), uint(N - 1))]; }

    template<typename T, int C, int R, typename I> inline thread vec<T, R>& wgpu_at(thread matrix<T, C, R>& m, I i) { return m[min(uint(i), uint(C - 1))]; }
    template<typename T, int C, int R, typename I> inline const thread vec<T, R>& wgpu_at(const thread matrix<T, C, R>& m, I i) { return m[min(uint(i), uint(C - 1))]; }
    template<typename T, int C, int R, typename I> inline device vec<T, R>& wgpu_at(device matrix<T, C, R>& m, I i) { return m[min(uint(i), uint(C - 1))]; }
    template<typename T, int C, int R, typename I> inline const device vec<T, R>& wgpu_at(const device matrix<T, C, R>& m, I i) { return m[min(uint(i), uint(C - 1))]; }
    template<typename T, int C, int R, typename I> inline constant vec<T, R>& wgpu_at(constant matrix<T, C, R>& m, I i) { return m[min(uint(i), uint(C - 1))]; }
    template<typename T, int C, int R, typename I> inline threadgroup vec<T, R>& wgpu_at(threadgroup matrix<T, C, R>& m, I i) { return m[min(uint(i), uint(C - 1))]; }

    // 런타임 크기 배열(`array<T>`)은 포인터로 내려와 크기가 타입에 없다 — 그 자리는 방출기가
    // 버퍼 크기 표로 상한을 계산해 넘긴다 (`wgpu_at_n`). 표가 없으면 여기로 떨어진다.
    template<typename T, typename I> inline device T& wgpu_at(device T* p, I i) { return p[uint(i)]; }
    template<typename T, typename I> inline const device T& wgpu_at(const device T* p, I i) { return p[uint(i)]; }
    template<typename T, typename I> inline constant T& wgpu_at(constant T* p, I i) { return p[uint(i)]; }

    // 상한을 밖에서 받는 형태 — 런타임 크기 배열용. `count`가 0이면 0번 자리로 접는다
    // (그 버퍼에는 읽을 것이 없지만, 주소만은 유효하다).
    template<typename T, typename I> inline device T& wgpu_at_n(device T* p, I i, uint count) { return p[count == 0u ? 0u : min(uint(i), count - 1u)]; }
    template<typename T, typename I> inline const device T& wgpu_at_n(const device T* p, I i, uint count) { return p[count == 0u ? 0u : min(uint(i), count - 1u)]; }

    // 인덱스 자리로 **쓰는** 경로. 벡터 성분은 참조로 못 돌려주므로 대입을 함수 안에서 끝낸다.
    // 배열·행렬·포인터도 같은 이름으로 받아, 방출기가 대상 종류를 따지지 않아도 되게 한다.
    template<typename C, typename I, typename V> inline void wgpu_store(thread C& c, I i, V value) { wgpu_at(c, i) = value; }
    template<typename C, typename I, typename V> inline void wgpu_store(device C& c, I i, V value) { wgpu_at(c, i) = value; }
    template<typename C, typename I, typename V> inline void wgpu_store(threadgroup C& c, I i, V value) { wgpu_at(c, i) = value; }
    template<typename T, int N, typename I, typename V> inline void wgpu_store(thread vec<T, N>& v, I i, V value) { v[min(uint(i), uint(N - 1))] = value; }
    template<typename T, int N, typename I, typename V> inline void wgpu_store(device vec<T, N>& v, I i, V value) { v[min(uint(i), uint(N - 1))] = value; }
    template<typename T, int N, typename I, typename V> inline void wgpu_store(threadgroup vec<T, N>& v, I i, V value) { v[min(uint(i), uint(N - 1))] = value; }
    template<typename T, typename I, typename V> inline void wgpu_store(device T* p, I i, V value) { p[uint(i)] = value; }
    template<typename T, typename I, typename V> inline void wgpu_store_n(device T* p, I i, uint count, V value) { p[count == 0u ? 0u : min(uint(i), count - 1u)] = value; }

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
    // WGSL의 정수 상수식(AbstractInt)은 **문맥 타입으로 굳는다** — `vec2(4, 1)`이 uint2 자리에
    // 오면 uint2, float2 자리에 오면 float2다. 타입 추론 없이 그 결정을 C++ 변환 연산자에 넘긴다.
    template<int N> struct wgpu_aint {
        vec<int, N> value;
        template<typename T> operator vec<T, N>() const { return vec<T, N>(value); }
    };
    inline wgpu_aint<2> wgpu_aint2(int x) { return wgpu_aint<2>{ int2(x) }; }
    inline wgpu_aint<2> wgpu_aint2(int x, int y) { return wgpu_aint<2>{ int2(x, y) }; }
    inline wgpu_aint<3> wgpu_aint3(int x) { return wgpu_aint<3>{ int3(x) }; }
    inline wgpu_aint<3> wgpu_aint3(int x, int y, int z) { return wgpu_aint<3>{ int3(x, y, z) }; }
    inline wgpu_aint<4> wgpu_aint4(int x) { return wgpu_aint<4>{ int4(x) }; }
    inline wgpu_aint<4> wgpu_aint4(int x, int y, int z, int w) {
        return wgpu_aint<4>{ int4(x, y, z, w) };
    }
    // 연산자는 다섯 모양이 필요하다. C++는 벡터 피연산자 하나만 보고 변환 연산자의 T를 추론하지
    // 못하므로(그래서 그냥 "invalid operands"가 난다) 각 자리를 명시해 준다.
    // 벡터 쪽이 `vec<T,N>`로 더 특수화되어 있어 스칼라 템플릿 S와 겹쳐도 모호하지 않다.
    // aint ⊗ aint는 정수 그대로 두어 **상수식이 계속 추상 상태로** 남게 한다.
    template<int N, typename T> inline vec<T,N> operator*(vec<T,N> a, wgpu_aint<N> b) { return a * vec<T,N>(b.value); }
    template<int N, typename T> inline vec<T,N> operator*(wgpu_aint<N> a, vec<T,N> b) { return vec<T,N>(a.value) * b; }
    template<int N, typename S> inline vec<S,N> operator*(S a, wgpu_aint<N> b) { return vec<S,N>(a) * vec<S,N>(b.value); }
    template<int N, typename S> inline vec<S,N> operator*(wgpu_aint<N> a, S b) { return vec<S,N>(a.value) * vec<S,N>(b); }
    template<int N> inline wgpu_aint<N> operator*(wgpu_aint<N> a, wgpu_aint<N> b) { return wgpu_aint<N>{ a.value * b.value }; }
    template<int N, typename T> inline vec<T,N> operator+(vec<T,N> a, wgpu_aint<N> b) { return a + vec<T,N>(b.value); }
    template<int N, typename T> inline vec<T,N> operator+(wgpu_aint<N> a, vec<T,N> b) { return vec<T,N>(a.value) + b; }
    template<int N, typename S> inline vec<S,N> operator+(S a, wgpu_aint<N> b) { return vec<S,N>(a) + vec<S,N>(b.value); }
    template<int N, typename S> inline vec<S,N> operator+(wgpu_aint<N> a, S b) { return vec<S,N>(a.value) + vec<S,N>(b); }
    template<int N> inline wgpu_aint<N> operator+(wgpu_aint<N> a, wgpu_aint<N> b) { return wgpu_aint<N>{ a.value + b.value }; }
    template<int N, typename T> inline vec<T,N> operator-(vec<T,N> a, wgpu_aint<N> b) { return a - vec<T,N>(b.value); }
    template<int N, typename T> inline vec<T,N> operator-(wgpu_aint<N> a, vec<T,N> b) { return vec<T,N>(a.value) - b; }
    template<int N, typename S> inline vec<S,N> operator-(S a, wgpu_aint<N> b) { return vec<S,N>(a) - vec<S,N>(b.value); }
    template<int N, typename S> inline vec<S,N> operator-(wgpu_aint<N> a, S b) { return vec<S,N>(a.value) - vec<S,N>(b); }
    template<int N> inline wgpu_aint<N> operator-(wgpu_aint<N> a, wgpu_aint<N> b) { return wgpu_aint<N>{ a.value - b.value }; }
    template<int N> inline wgpu_aint<N> operator-(wgpu_aint<N> a) { return wgpu_aint<N>{ -a.value }; }
    template<int N, typename T> inline vec<T,N> operator/(vec<T,N> a, wgpu_aint<N> b) { return a / vec<T,N>(b.value); }
    template<int N, typename T> inline vec<T,N> operator/(wgpu_aint<N> a, vec<T,N> b) { return vec<T,N>(a.value) / b; }
    template<int N, typename S> inline vec<S,N> operator/(S a, wgpu_aint<N> b) { return vec<S,N>(a) / vec<S,N>(b.value); }
    template<int N, typename S> inline vec<S,N> operator/(wgpu_aint<N> a, S b) { return vec<S,N>(a.value) / vec<S,N>(b); }
    template<int N> inline wgpu_aint<N> operator/(wgpu_aint<N> a, wgpu_aint<N> b) { return wgpu_aint<N>{ a.value / b.value }; }

    // `textureSampleBaseClampToEdge` — 밉 0에서, 좌표를 텍셀 절반만큼 안쪽으로 물려 샘플한다.
    template<typename T> inline vec<T,4> wgpu_sample_base_clamp(texture2d<T> tex, sampler smp, float2 coord) {
        float2 size = float2(tex.get_width(), tex.get_height());
        float2 halfTexel = 0.5 / size;
        return tex.sample(smp, clamp(coord, halfTexel, float2(1.0) - halfTexel), level(0.0));
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
