# WGSL 서브셋

`LynxWebGPUShader`가 WGSL을 Metal Shading Language로 옮기는 범위와 규칙.

## 1. 지원 문법

### 선언

| WGSL | MSL |
|---|---|
| `struct S { … }` | `struct alignas(N) S { … }` (WGSL 배치에 맞춘 패딩 삽입 — §3) |
| `alias T = …;` | `using T = …;` |
| `const X = …;` / `override X = …;` | `constant … X = …;` (override는 파이프라인 `constants`로 대체 — §2-2) |
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
| `enable f16;` / `requires …;` / `diagnostic(…);` | 무시 (MSL로 옮길 것이 없다) |

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

WGSL에서는 평범한 식별자가 MSL(C++14 + Metal 확장)에서는 예약어인 경우가 있다 —
그래픽스 셰이더에서 흔한 `texture` `sampler` `device` `char` `vertex` 같은 이름이 대표적이다.
방출기가 **선언과 모든 사용처를 함께** 바꾸므로 셰이더를 고칠 필요는 없다:

| 대상 | 변환 |
|---|---|
| 진입점·함수 이름 중 `main` | `wgpu_fn_main` |
| 그 외 예약어 충돌 (변수·매개변수·구조체·멤버·상수) | `wgpu_id_<이름>` |

JS에서는 원래 이름(`entryPoint: 'main'`)을 그대로 쓴다. 런타임이
`WGSLShaderModule.mslFunctionName(for:)`으로 변환해 `MTLLibrary`에서 찾는다.

## 2-1. 셰이더 프렐류드

생성된 MSL 맨 앞에 헬퍼 템플릿 묶음이 붙는다 (`MSLPrelude`). 이 트랜스파일러는 타입 추론기가
아니라 **구문 번역기**라서, WGSL이 타입으로 결정하는 것들을 C++ 템플릿의 `decltype`에 넘긴다:

| WGSL | 프렐류드가 하는 일 |
|---|---|
| `a % b` (부동소수) | `wgpu_mod` — 정수는 `%`, 부동소수는 `fmod`로 갈라 보낸다 |
| `vec2(x, 4)` (성분 타입 생략) | `wgpu_vec2` — 인자 타입에서 성분 타입을 추론한다 |
| `max(x, 0)` (리터럴 승격) | `wgpu_max` 등 — 두 인자의 공통 타입으로 맞춘 뒤 호출한다 |
| `radians(d)` / `degrees(r)` | MSL에 없어 직접 정의한다 |
| `vec3(1)` (인자가 전부 정수 상수식) | `wgpu_aint<N>` — 문맥 타입으로 굳는 **추상 정수 벡터** 대역. 변환 연산자와 산술 연산자 오버로드로 f32/u32/i32 어느 자리에나 들어간다 |
| `textureSampleBaseClampToEdge(t, s, uv)` | `wgpu_sample_base_clamp` — 밉 0에서 좌표를 텍셀 절반만큼 물려 샘플한다 |

인스턴스화되지 않은 템플릿은 코드를 만들지 않으므로 런타임 비용은 없다.

## 2-2. 파이프라인 상수 (`override`)

```wgsl
override blockSize: u32 = 16;
override intensity: f32;          // 기본값 없음 — 호스트가 반드시 줘야 한다
@group(0) @binding(0) var<storage, read> data: array<f32, blockSize>;
```

```js
device.createComputePipeline({
  layout: 'auto',
  compute: { module, entryPoint: 'main', constants: { blockSize: 32, intensity: 0.8 } },
})
```

값은 **MSL 방출 시점에 상수로 박힌다**. 배열 길이가 `override`에 걸린 경우까지 자연히 풀린다.
기본값도 없고 `constants`로도 주지 않으면 파이프라인 생성이 명확한 오류로 실패한다.

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

## 3-1. 런타임 크기 배열과 외부 텍스처

```wgsl
struct Particles {
  count: u32,
  items: array<vec4f>,      // 구조체 말미의 런타임 크기 배열도 된다
}
@group(0) @binding(0) var<storage, read_write> particles: Particles;
@group(0) @binding(1) var frame: texture_external;
@group(0) @binding(2) var frameSampler: sampler;

fn count_and_sample(uv: vec2f) -> vec4f {
  let n = arrayLength(&particles.items);                              // 바인딩된 크기 기준
  return textureSampleBaseClampToEdge(frame, frameSampler, uv) * f32(n);
}
```

**`arrayLength()`** — Metal 셰이더는 버퍼 크기를 알 수 없다. 그래서 바인딩된 버퍼의 바이트 수를 담은
작은 표를 예약 인덱스(`WGSLMetalLimits.bufferSizesIndex` = 22)에 꽂아 주고, `arrayLength(&g)`를
`(크기표[i] - 멤버오프셋) / sizeof(원소)`로 번역한다 (Dawn과 같은 방식). 길이는 **바인딩된 크기**를
따르므로 `{ buffer, offset, size }`로 일부만 묶으면 그만큼만 센다. 쓰지 않는 셰이더에는 표를 넘기지 않는다.

**`texture_external`** — 샘플링 가능한 2D 텍스처로 내려간다. 바인드 그룹에는 보통
`GPUTextureView`를 묶으면 된다 (한 면짜리 비디오 프레임과 같은 모양).
`textureSampleBaseClampToEdge`는 밉 0에서 좌표를 텍셀 절반만큼 안쪽으로 물려 샘플하므로,
`repeat` 샘플러라도 프레임 경계에서 반대쪽이 감겨 들어오지 않는다.

## 4. 지원하지 않는 것

명시적으로 **거부**한다 (조용히 틀리게 번역하지 않는다):

| 기능 | 이유 / 대안 |
|---|---|
| `atomicCompareExchangeWeak` | 반환 구조체를 옮기지 못한다 |
| `modf` / `frexp` | 반환 구조체를 옮기지 못한다. `floor`/`fract`로 나눠 쓸 것 |
| `workgroupUniformLoad` | 미지원 |
| `break if cond;` | `continuing` 블록 전용 구문. `if (cond) { break; }`로 바꿔 쓸 것 |
| `continuing` + `continue` 조합 | `continue`가 `continuing` 블록을 건너뛰게 되어 의미가 달라진다. `for`로 바꿀 것 |

암묵적 제약:
- **성분 타입이 생략되고 인자가 전부 정수 상수식인 벡터 생성자**(`vec2(4, 1)`, `vec3(1)`)는
  WGSL 명세대로 **쓰이는 자리의 타입으로 굳는다** — f32 자리에서는 f32, u32 자리에서는 u32.
  프렐류드의 `wgpu_aint<N>` 대역이 그 결정을 C++ 오버로드 해석에 넘긴다 (§2-1).
  단 대역 값을 **Metal 내장 함수에 그대로 넘기는 자리**(`normalize(vec3(1))`)는 타입 추론이
  걸릴 수 있다. 그럴 때는 `vec3f(1)`처럼 성분 타입을 적으면 된다.
- 여기 표에 없는 내장 함수는 **이름 그대로 통과**한다. MSL에 같은 이름이 없으면 파이프라인 생성 시
  "MSL 컴파일 실패" 오류와 함께 생성된 MSL 전문이 나온다.

## 4-1. 실제 셰이더로 재 본 호환성

공식 [webgpu-samples](https://github.com/webgpu/webgpu-samples)의 WGSL 68개를 그대로 통과시켜 본 결과
(번역 + **실제 Metal 컴파일**까지):

| 결과 | 수 |
|---|---|
| 그대로 통과 | **60 / 67 (89%)** |
| 호스트가 `constants`를 주면 동작 (`override` 사용) | 4 |
| 코퍼스 자체가 단독 파일이 아님 (다른 파일과 이어 붙이거나 호스트가 문자열을 치환해 쓰는 조각) | 3 |

**남은 3건은 트랜스파일러의 빈틈이 아니다**: `cornell/rasterizer.wgsl`과 `skinnedMesh/gltf.wgsl`은
선언이 다른 `.wgsl` 파일에 있고(샘플이 이어 붙여 쓴다), `cornell/tonemapper.wgsl`은
`texture_storage_2d<{OUTPUT_FORMAT}, write>`처럼 JS가 치환할 자리를 그대로 담고 있다.

재현 방법은 `docs/TESTING.md` §7.

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
