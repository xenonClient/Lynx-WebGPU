# WGSL 서브셋

`LynxWebGPUShader`가 WGSL을 Metal Shading Language로 옮기는 범위와 규칙.

## 1. 지원 문법

### 선언

| WGSL | MSL |
|---|---|
| `struct S { … }` | `struct alignas(N) S { … }` (WGSL 배치에 맞춘 패딩 삽입 — §3) |
| `alias T = …;` | `using T = …;` |
| `const X = …;` / `override X = …;` | `constant … X = …;` (override는 기본값만) |
| `fn f(a: T) -> R { … }` | `R f(T a, /* 스레딩된 리소스 */) { … }` |
| `@vertex` / `@fragment` / `@compute` | `vertex` / `fragment` / `kernel` 래퍼 + `…_inner` |
| `var<uniform> u: T` | `constant T& u [[buffer(n)]]` |
| `var<storage, read> s: T` | `const device T& s [[buffer(n)]]` |
| `var<storage, read_write> s: T` | `device T& s [[buffer(n)]]` |
| `var<storage, …> s: array<T>` | `device T* s [[buffer(n)]]` |
| `var<workgroup> w: T` | 진입점 안의 `threadgroup T w;` + `threadgroup T&` 인자 |
| `var<private> p: T` | 진입점 안의 지역 변수 + `thread T&` 인자 |
| `var t: texture_2d<f32>` | `texture2d<float> t [[texture(n)]]` |
| `var s: sampler` | `sampler s [[sampler(n)]]` |

### 타입

`f32` `i32` `u32` `bool` `f16` · `vecN<T>` / `vecNf` `vecNi` `vecNu` `vecNh` ·
`matCxR<f32>` / `matCxRf` · `array<T, N>` · `array<T>`(storage 전용) · `atomic<T>` · `ptr<space, T>` ·
`texture_1d/2d/2d_array/3d/cube/cube_array<T>` · `texture_multisampled_2d<T>` ·
`texture_depth_2d/2d_array/cube/cube_array` · `texture_storage_1d/2d/2d_array/3d<format, access>` ·
`sampler` · `sampler_comparison` · 사용자 구조체/별칭

### 문장 · 표현식

`let` `var` `const` · 대입/복합 대입 · `++` `--` · `if`/`else if`/`else` · `for` · `while` ·
`loop`(+ `continuing`) · `switch`(fallthrough 없음) · `return` `break` `continue` `discard` ·
모든 이항/단항 연산자 · 스위즐(`.xyz` `.rgb`) · 인덱싱 · 멤버 접근 · `&`/`*` (포인터)

### 빌트인

| WGSL `@builtin` | MSL |
|---|---|
| `vertex_index` / `instance_index` | `[[vertex_id]]` / `[[instance_id]]` |
| `position` | `[[position]]` |
| `front_facing` / `frag_depth` | `[[front_facing]]` / `[[depth(any)]]` |
| `sample_index` / `sample_mask` | `[[sample_id]]` / `[[sample_mask]]` |
| `local_invocation_id` / `local_invocation_index` | `[[thread_position_in_threadgroup]]` / `[[thread_index_in_threadgroup]]` |
| `global_invocation_id` | `[[thread_position_in_grid]]` |
| `workgroup_id` / `num_workgroups` | `[[threadgroup_position_in_grid]]` / `[[threadgroups_per_grid]]` |

`@location(n)`은 정점 입력에서 `[[attribute(n)]]`, 정점→프래그먼트 보간에서 `[[user(locnN)]]`,
프래그먼트 출력에서 `[[color(n)]]`이 된다. `@interpolate(flat)` → `[[flat]]`.

### 내장 함수

이름이 같은 것은 그대로 통과한다 (`min` `max` `clamp` `mix` `dot` `cross` `normalize` `length`
`pow` `sin` `cos` `atan2` `floor` `fract` `smoothstep` `select` `transpose` `determinant` …).

이름이 다른 것들:

| WGSL | MSL |
|---|---|
| `inverseSqrt` | `rsqrt` |
| `dpdx` / `dpdy` (+ Coarse/Fine) | `dfdx` / `dfdy` |
| `faceForward` | `faceforward` |
| `countLeadingZeros` / `countTrailingZeros` / `countOneBits` | `clz` / `ctz` / `popcount` |
| `reverseBits` / `extractBits` / `insertBits` | `reverse_bits` / `extract_bits` / `insert_bits` |
| `pack4x8unorm` 계열 | `pack_float_to_unorm4x8` 계열 |
| `bitcast<T>(x)` | `as_type<T>(x)` |
| `quantizeToF16(x)` | `float(half(x))` |

메서드 호출로 바뀌는 것들 (선언된 텍스처 차원을 보고 좌표를 맞춘다):

| WGSL | MSL |
|---|---|
| `textureSample(t, s, uv)` | `t.sample(s, uv)` |
| `textureSampleLevel(t, s, uv, lod)` | `t.sample(s, uv, level(lod))` |
| `textureSampleBias(t, s, uv, b)` | `t.sample(s, uv, bias(b))` |
| `textureSampleGrad(t, s, uv, dx, dy)` | `t.sample(s, uv, gradient2d(dx, dy))` |
| `textureSampleCompare(t, s, uv, ref)` | `t.sample_compare(s, uv, ref)` |
| `textureLoad(t, coord, level)` | `t.read(uint2(coord), level)` |
| `textureStore(t, coord, v)` | `t.write(v, uint2(coord))` |
| `textureDimensions(t)` | `uint2(t.get_width(), t.get_height())` |
| `textureNumLayers/Levels/Samples(t)` | `t.get_array_size()/get_num_mip_levels()/get_num_samples()` |

동기화/아토믹:

| WGSL | MSL |
|---|---|
| `workgroupBarrier()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| `storageBarrier()` | `threadgroup_barrier(mem_flags::mem_device)` |
| `atomicLoad/Store/Exchange` | `atomic_*_explicit(…, memory_order_relaxed)` |
| `atomicAdd/Sub/Max/Min/And/Or/Xor` | `atomic_fetch_*_explicit(…, memory_order_relaxed)` |

## 2. 이름 규칙

`main`처럼 MSL이 함수 이름으로 거부하는 이름은 방출 시 `wgpu_fn_main`으로 바뀐다.
JS에서는 원래 이름(`entryPoint: 'main'`)을 그대로 쓰면 되고, 런타임이
`WGSLShaderModule.mslFunctionName(for:)`으로 변환해 `MTLLibrary`에서 찾는다.

## 3. 구조체 배치

WGSL과 MSL은 `vec3`에서 갈린다 — WGSL `vec3<f32>`는 정렬 16 / **크기 12**, MSL `float3`는 크기 **16**이다.

```wgsl
struct Light {
    direction: vec3f,   // offset 0,  size 12
    intensity: f32,     // offset 12
    color: vec3f,       // offset 16
};                      // size 32
```
```metal
struct alignas(16) Light {
    packed_float3 direction;   // ← 뒤 멤버가 12번지를 쓰므로 packed
    float intensity;
    float3 color;              // ← 뒤가 비었으므로 일반 float3
};
```

규칙:
- 멤버 오프셋은 WGSL 명세(`AlignOf`/`SizeOf`/`RoundUp`)대로 계산한다. `@align(n)`/`@size(n)`을 존중한다.
- uniform 주소 공간에서 쓰이는 구조체는 정렬/배열 스트라이드가 16으로 올라간다.
- vec3 멤버 뒤 12~15바이트 구간을 다른 멤버가 쓰면 `packed_floatN`, 아니면 `floatN` + `char` 패딩.

**따라서 JS는 WGSL 규칙대로 버퍼를 채우면 된다.** 브라우저에서 쓰던 유니폼 패킹 코드가 그대로 동작한다.

## 4. 지원하지 않는 것

명시적으로 **거부**한다 (조용히 틀리게 번역하지 않는다):

| 기능 | 이유 / 대안 |
|---|---|
| `arrayLength(&buf)` | 셰이더가 버퍼 크기를 알아야 한다. 길이를 유니폼으로 넘길 것 |
| `atomicCompareExchangeWeak` | 반환 구조체를 옮기지 못한다 |
| `modf` / `frexp` | 반환 구조체를 옮기지 못한다. `floor`/`fract`로 나눠 쓸 것 |
| `workgroupUniformLoad` | 미지원 |
| `break if cond;` | `continuing` 블록 전용 구문. `if (cond) { break; }`로 바꿔 쓸 것 |
| `continuing` + `continue` 조합 | `continue`가 `continuing` 블록을 건너뛰게 되어 의미가 달라진다. `for`로 바꿀 것 |
| 런타임 크기 배열이 **구조체 멤버**인 경우 | 길이 1 배열로 방출된다. 저장 타입을 `array<T>` 자체로 선언할 것 |

암묵적 제약 (Metal 컴파일러가 잡는다):
- **부동소수 `%`** — WGSL은 f32에 `%`를 허용하지만 MSL은 정수만 받는다. `fmod(a, b)`를 쓸 것.
- 성분 타입을 생략한 `vec3(…)` 생성자는 **f32로 가정**한다. `vec3u(…)`처럼 명시할 것.
- 여기 표에 없는 내장 함수는 **이름 그대로 통과**한다. MSL에 같은 이름이 없으면 파이프라인 생성 시
  "MSL 컴파일 실패" 오류와 함께 생성된 MSL 전문이 나온다.

## 5. MSL 탈출구

트랜스파일러가 감당 못 하는 셰이더는 MSL을 직접 넣을 수 있다:

```js
const module = device.createShaderModule({ code: MSL_SOURCE, language: 'msl' })
```

이때는
- 바인딩 인덱스(`[[buffer(n)]]` 등)를 **직접** 써야 한다. 배정 규칙은 `docs/ARCHITECTURE.md` §4 참고.
- `layout: 'auto'`를 쓸 수 없다 — `GPUPipelineLayout`을 명시할 것.
- 진입점 이름은 MSL 함수 이름 그대로 쓴다 (이름 변환 없음).
- 정점 버퍼는 Metal 인덱스 `30 - slot`에 온다.

## 6. 트랜스파일러 확장

문법이나 내장 함수를 더할 때의 절차는 `.claude/skills/wgsl-feature/SKILL.md`에 있다.
**모든 변경은 `MetalCompilerHarness.assertCompiles`가 붙은 테스트를 동반해야 한다** —
문자열만 맞고 실제로는 컴파일되지 않는 MSL을 막기 위한 장치다.
