import Foundation

/// The helper set prepended to every generated MSL source.
///
/// **Why it is needed** — this transpiler is a syntax translator, not a type inferencer. So the
/// things WGSL settles by type (a `vec2(a, b)` with the component type omitted, literal promotion in
/// `max(x, 0)`, floating-point `%`) would become type errors in MSL, or **silently the wrong type**,
/// if carried across as-is.
///
/// Handing that judgement to C++ templates' `decltype` reproduces WGSL's result without any type
/// inference. An uninstantiated template generates no code, so it costs nothing.
enum MSLPrelude {
    static let source = """
    // ─── Lynx-WebGPU shader prelude ───────────────────────────────────────
    // Fills only the points where WGSL and MSL semantics diverge. Background: docs/WGSL.md §1.

    // WGSL's `%` is defined for floating point too (fmod); MSL's `%` is integer-only.
    // Unify both arguments to a common type first (handling mixed int/float), then dispatch by kind.
    inline int wgpu_mod_same(int a, int b) { return a % b; }
    inline uint wgpu_mod_same(uint a, uint b) { return a % b; }
    inline int2 wgpu_mod_same(int2 a, int2 b) { return a % b; }
    inline int3 wgpu_mod_same(int3 a, int3 b) { return a % b; }
    inline int4 wgpu_mod_same(int4 a, int4 b) { return a % b; }
    inline uint2 wgpu_mod_same(uint2 a, uint2 b) { return a % b; }
    inline uint3 wgpu_mod_same(uint3 a, uint3 b) { return a % b; }
    inline uint4 wgpu_mod_same(uint4 a, uint4 b) { return a % b; }
    template<typename T> inline T wgpu_mod_same(T a, T b) { return fmod(a, b); }

    // Fills the **undefined places** of integer division and remainder.
    //
    // WGSL defines `x / 0 == x` and `x % 0 == 0`, and also `INT_MIN / -1 == INT_MIN` and
    // `INT_MIN % -1 == 0`. All three are **undefined behaviour** in C++, so leaving them alone lets a
    // GPU driver do anything at all.
    //
    // In every case, **replacing the divisor with 1** yields exactly the value WGSL specifies
    // (`a / 1 == a`, `a % 1 == 0`). So only the denominator is chosen.
    template<typename A, typename B> inline B wgpu_denom(A, B b) { return b; }   // floats and abstract ints pass through
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

    // Shift-amount masking — WGSL leaves shifts at or beyond the width undefined, and C++ makes it UB.
    // Keep the low 5 bits, the same rule as Tint (i32 and u32 are both 32-bit).
    inline uint wgpu_shift_amount(uint b) { return b & 31u; }
    inline int wgpu_shift_amount(int b) { return b & 31; }
    template<int N> inline vec<uint, N> wgpu_shift_amount(vec<uint, N> b) { return b & 31u; }
    template<int N> inline vec<int, N> wgpu_shift_amount(vec<int, N> b) { return b & 31; }
    template<typename T> inline T wgpu_shift_amount(T b) { return b; }
    template<typename A, typename B> inline auto wgpu_shl(A a, B b) { return a << wgpu_shift_amount(b); }
    template<typename A, typename B> inline auto wgpu_shr(A a, B b) { return a >> wgpu_shift_amount(b); }

    // f32 → i32/u32 conversion **saturates** in WGSL (out-of-range values clamp to the ends).
    // A C++ cast is UB out of range. NaN, where every comparison is false, goes to the upper bound — as in Tint.
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
    // Anything that is not f32 (ints, bool, f16, abstract ints) converts plainly — it cannot go out of range.
    template<typename T> inline int wgpu_ftoi(T v) { return int(v); }
    template<typename T> inline uint wgpu_ftou(T v) { return uint(v); }
    // Conversion in vector position — stating the size accepts scalar broadcast (`vec2i(1.5)`) and
    // component conversion alike. The last overload **just constructs**: routing non-f32 vectors
    // (`vec2i(someU32Vec)`) through component conversion would break the compile trying to narrow a vector to a scalar.
    template<int N> inline vec<int, N> wgpu_ftoi_n(vec<float, N> v) { return wgpu_ftoi(v); }
    template<int N> inline vec<int, N> wgpu_ftoi_n(float v) { return vec<int, N>(wgpu_ftoi(v)); }
    template<int N, typename T> inline vec<int, N> wgpu_ftoi_n(T v) { return vec<int, N>(v); }
    template<int N> inline vec<uint, N> wgpu_ftou_n(vec<float, N> v) { return wgpu_ftou(v); }
    template<int N> inline vec<uint, N> wgpu_ftou_n(float v) { return vec<uint, N>(wgpu_ftou(v)); }
    template<int N, typename T> inline vec<uint, N> wgpu_ftou_n(T v) { return vec<uint, N>(v); }

    // Index range clamping — the robustness the WebGPU spec requires.
    //
    // An out-of-range access **reads or overwrites adjacent GPU memory** in the same process, and a
    // large excursion kills the command buffer with a page fault. Indices usually come from uniforms
    // or storage, so they are ultimately **values the bundle (JS) decides** — which matters most in an
    // environment that assumes bundles downloaded from a server.
    //
    // C++ knows the size (`array<T,N>`, `vec<T,N>`). Reference types differ per address space, hence the overloads.
    // Returning a reference keeps reads, writes and `&a[i]` working as before, and the index is evaluated once.
    template<typename T, size_t N, typename I> inline thread T& wgpu_at(thread array<T, N>& a, I i) { return a[min(uint(i), uint(N - 1))]; }
    template<typename T, size_t N, typename I> inline const thread T& wgpu_at(const thread array<T, N>& a, I i) { return a[min(uint(i), uint(N - 1))]; }
    template<typename T, size_t N, typename I> inline device T& wgpu_at(device array<T, N>& a, I i) { return a[min(uint(i), uint(N - 1))]; }
    template<typename T, size_t N, typename I> inline const device T& wgpu_at(const device array<T, N>& a, I i) { return a[min(uint(i), uint(N - 1))]; }
    template<typename T, size_t N, typename I> inline constant T& wgpu_at(constant array<T, N>& a, I i) { return a[min(uint(i), uint(N - 1))]; }
    template<typename T, size_t N, typename I> inline threadgroup T& wgpu_at(threadgroup array<T, N>& a, I i) { return a[min(uint(i), uint(N - 1))]; }
    template<typename T, size_t N, typename I> inline const threadgroup T& wgpu_at(const threadgroup array<T, N>& a, I i) { return a[min(uint(i), uint(N - 1))]; }

    // Vector components alone come back **by value**. In MSL `v[i]` is not a place a reference can
    // bind to (`non-const reference cannot bind to vector element`), so returning one breaks the compile.
    // Writing is handled by `wgpu_store` below.
    // A `thread` vector gets **only a const reference** — offering by-value and const-reference
    // together makes both match equally on a non-const lvalue, so the call becomes ambiguous.
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

    // A runtime-sized array (`array<T>`) arrives as a pointer with no size in its type — there the
    // emitter computes the bound from the buffer size table and passes it (`wgpu_at_n`). With no table it falls through to here.
    template<typename T, typename I> inline device T& wgpu_at(device T* p, I i) { return p[uint(i)]; }
    template<typename T, typename I> inline const device T& wgpu_at(const device T* p, I i) { return p[uint(i)]; }
    template<typename T, typename I> inline constant T& wgpu_at(constant T* p, I i) { return p[uint(i)]; }

    // The form taking the bound from outside — for runtime-sized arrays. A `count` of 0 folds to index 0
    // (there is nothing to read in that buffer, but the address alone stays valid).
    template<typename T, typename I> inline device T& wgpu_at_n(device T* p, I i, uint count) { return p[count == 0u ? 0u : min(uint(i), count - 1u)]; }
    template<typename T, typename I> inline const device T& wgpu_at_n(const device T* p, I i, uint count) { return p[count == 0u ? 0u : min(uint(i), count - 1u)]; }

    // The **write** path in index position. Vector components cannot come back by reference, so the
    // assignment finishes inside the function. Arrays, matrices and pointers take the same name so the
    // emitter need not distinguish the target kind.
    template<typename C, typename I, typename V> inline void wgpu_store(thread C& c, I i, V value) { wgpu_at(c, i) = value; }
    template<typename C, typename I, typename V> inline void wgpu_store(device C& c, I i, V value) { wgpu_at(c, i) = value; }
    template<typename C, typename I, typename V> inline void wgpu_store(threadgroup C& c, I i, V value) { wgpu_at(c, i) = value; }
    template<typename T, int N, typename I, typename V> inline void wgpu_store(thread vec<T, N>& v, I i, V value) { v[min(uint(i), uint(N - 1))] = value; }
    template<typename T, int N, typename I, typename V> inline void wgpu_store(device vec<T, N>& v, I i, V value) { v[min(uint(i), uint(N - 1))] = value; }
    template<typename T, int N, typename I, typename V> inline void wgpu_store(threadgroup vec<T, N>& v, I i, V value) { v[min(uint(i), uint(N - 1))] = value; }
    template<typename T, typename I, typename V> inline void wgpu_store(device T* p, I i, V value) { p[uint(i)] = value; }
    template<typename T, typename I, typename V> inline void wgpu_store_n(device T* p, I i, uint count, V value) { p[count == 0u ? 0u : min(uint(i), count - 1u)] = value; }

    // MSL has no radians/degrees.
    template<typename T> inline T wgpu_radians(T d) { return d * T(0.017453292519943295); }
    template<typename T> inline T wgpu_degrees(T r) { return r * T(57.29577951308232); }

    // Vector constructors with the component type omitted — the type is inferred from the arguments (`vec2(u, 4)` → uint2).
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

    // WGSL promotes a literal to the context type (`max(x, 0)`). MSL splits the overload right there.
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
    // WGSL's integer constant expressions (AbstractInt) **freeze into the context type** — `vec2(4, 1)`
    // is uint2 in a uint2 position and float2 in a float2 one. That decision is handed to C++ conversion
    // operators, with no type inference.
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
    // The operators need five shapes. C++ cannot infer the conversion operator's T from a single vector
    // operand (it just reports "invalid operands"), so each position is spelled out.
    // The vector side is more specialized as `vec<T,N>`, so it stays unambiguous against the scalar template S.
    // aint ⊗ aint stays integral so that **a constant expression remains abstract**.
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

    // `textureSampleBaseClampToEdge` — samples at mip 0 with the coordinate pulled half a texel inward.
    template<typename T> inline vec<T,4> wgpu_sample_base_clamp(texture2d<T> tex, sampler smp, float2 coord) {
        float2 size = float2(tex.get_width(), tex.get_height());
        float2 halfTexel = 0.5 / size;
        return tex.sample(smp, clamp(coord, halfTexel, float2(1.0) - halfTexel), level(0.0));
    }
    // ──────────────────────────────────────────────────────────────────────
    """

    /// Builtins routed through a prelude helper (WGSL name → MSL helper name).
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

    /// Vector constructors with the component type omitted.
    static let inferredVectorConstructors: Set<String> = ["vec2", "vec3", "vec4"]
}
