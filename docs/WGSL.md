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

### 컴파일 시점 정수 (속성 인자 · 배열 길이)

`@group` `@binding` `@location` `@align` `@size`의 인자와 `array<T, N>`의 길이는 **상수식으로 계산**한다
(`WGSLConstantEvaluator`). 지원 범위:

- 정수 리터럴 — 명세 문법 그대로: 10진(`0`, 선행 0 없음) · 16진(`0x10` `0X1F`) · 접미사 `i`(i32) `u`(u32).
  접미사 없는 값은 AbstractInt(i64 범위). 형식이 틀리거나 타입 범위를 넘으면 거부한다
- 모듈 스코프 `const` 이름 (선언이 사용보다 뒤에 와도 된다) · 괄호 · 단항 `-` · `+ - * / %` ·
  변환 `i32(e)` `u32(e)` (값이 대상 타입 범위에 들 때)
- 배열 길이에서만: `override` 이름(§2-2)과 함수 지역 `const` (모듈 전체에서 값이 하나로 정해질 때)
- `@workgroup_size(x, y?, z?)`도 같은 계산을 거치고 `override` 이름을 받는다 (§2-2). 생략한 y·z는 1

속성은 const-expression 자리라 `override`를 받지 않는다 (`@workgroup_size` 제외). 계산할 수 없거나 범위를 벗어난 값
(`@binding(-1)`, `@align(3)`, 순환하는 `const`, 지원하지 않는 연산자·함수 호출)은 **셰이더 모듈 생성
오류**다 — 줄 번호와 이유가 함께 나온다. 중첩(연산자·`const` 사슬 합산)이 256단을 넘는 식도
같은 오류다 (JS 스레드의 작은 스택을 넘치지 않게). `const` 이름은 한 평가 안에서 **한 번만** 계산한다 —
`const x1 = x0 + x0; const x2 = x1 + x1; …`처럼 참조가 갈라지는 사슬을 매번 다시 걸으면 줄 수에
지수로 늘어 스무 줄이 모듈 생성을 몇 분씩 붙잡는다. 예전에는 숫자 아닌 글자를 지워 읽어서 `@binding(0x10)`이
10, `array<u32, 0x100>`이 길이 100이 되는 식으로 **조용히 틀렸다**.
`@workgroup_size`는 인자를 `Int(글자) ?? 1`로 읽어서 `64u` · `0x40` · `const` 이름이 전부 **스레드 1개**가
됐다. 이제 인자가 1~3개가 아니거나, 계산할 수 없거나, `override`가 아닌 값이 1보다 작으면 셰이더 모듈
생성 오류다. `@compute` 진입점에 `@workgroup_size`가 없는 것도 같은 오류다 (명세상 필수).

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
| `textureSampleGrad(t, s, uv, dx, dy)` | `t.sample(s, uv, gradient2d(dx, dy))` (3d는 `gradient3d`, 큐브는 `gradientcube`) |
| `textureSampleCompare(t, s, uv, ref)` | `t.sample_compare(s, uv, ref)` |
| `textureSampleCompareLevel(t, s, uv, ref)` | `t.sample_compare(s, uv, ref, level(0))` — 정점·컴퓨트 셰이더에서도 된다 |
| 위 샘플 함수의 끝 인자 `offset` | 맨 뒤(LOD 옵션 다음)에 `int2(offset)` / 3d는 `int3(offset)`. 큐브에 주면 오류 |
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
| `metal` 네임스페이스의 이름 (`bias` `level` `coord` `address` `component` `mesh` 같은 샘플러 옵션·좌표 구조체) | `wgpu_id_<이름>` — 프렐류드의 `using namespace metal;` 때문에 모듈 스코프 `const`/`override`가 이 이름이면 모든 사용처가 "ambiguous"로 컴파일에 실패했다. 내장 함수 이름(`min` `length` …)은 호출이 그대로 통과해야 하므로 여기 넣지 않는다 |
| `wgpu_`로 시작하는 사용자 식별자 (종류 무관) | `wgpu_id_<이름>` |

JS에서는 원래 이름(`entryPoint: 'main'`)을 그대로 쓴다. 런타임이
`WGSLShaderModule.mslFunctionName(for:)`으로 변환해 `MTLLibrary`에서 찾는다.

**`wgpu_` 이름공간은 방출기 전용이다** — 프렐류드 헬퍼(`wgpu_mod` …), 패딩 멤버(`wgpu_padN`),
진입점 래퍼(`wgpu_ep_<진입점>_inner/_in/_out`), `discard` 플래그(`wgpu_discarded`) 같은 내부 이름이
모두 여기 산다. 사용자 식별자가 이 접두사로 시작하면 위 표대로 `wgpu_id_`로 옮겨지므로 내부 이름을
재선언하거나 프렐류드 헬퍼에 **조용히 오버로드로 끼어드는** 일이 없다 (방출기가 직접 만드는 이름 중
`wgpu_id_`로 시작하는 것은 없다). 셰이더는 고칠 필요가 없다.

**모듈 스코프 이름은 하나의 이름공간이다** — 구조체·별칭·함수·`var`·`const`·`override` 중 같은 이름이
두 번 선언되면 `createShaderModule`이 파싱 오류(`redeclaration of '…'`, 두 번째 선언의 줄 번호)로
거부한다. 명세가 금지하고 Tint도 같은 자리에서 거부한다. 함수 안에서 모듈 스코프 이름을 가리는
지역 선언은 그대로 허용된다.

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

값은 명세대로 **`override`의 WGSL 타입으로 WebIDL 변환**한다 — `i32`/`u32`는 `[EnforceRange] long`/
`unsigned long`(소수는 0 쪽으로 버리고, 범위 밖·`NaN`·`Infinity`는 거부), `f32`는 유한하고 f32로
반올림해도 무한대가 아니어야 하며(`f16`은 그 위에 절댓값 65504 이하 — Dawn과 같은 기준), `bool`은
0·`NaN`만 `false`다. 변환이 안 되는 값은 파이프라인 생성의 **검증 오류**가 된다. 타입 표기가 없는
`override`는 초기식으로 타입이 정해진다 (`override s = 0.5;`는 f32 — `constants: { s: 2 }`를 줘도
정수가 되지 않는다). 리터럴·연산·다른 상수·별칭 체인·내장 함수(`min`/`max`/`clamp`/`select`/`pow`,
`countOneBits` 같은 비트 함수, `all`/`any`, `pack*`, `dot`/`length`, `bitcast<T>`)·벡터/배열의 멤버·인덱스 접근까지 읽고,
그래도 타입을 알 수 없는 초기식이면 정수로 단정할 수 없으므로 **f32로 박는다**.

`@workgroup_size`도 `override`를 받는다 (webgpu-samples `gameOfLife`의
`@workgroup_size(blockSize, blockSize)`). 크기는 **`constants`를 적용한 모듈에서 파이프라인 생성 시점에**
계산하고, 디스패치의 threadsPerThreadgroup과 `var<workgroup>` 0 초기화 루프의 보폭이 같은 값을 쓴다
(어긋나면 실제 스레드 수 너머의 칸이 0으로 깔리지 않는다). 파이프라인 생성 시점의 검사 —
어기면 validation 오류다:

- 각 차원 ≥ 1 (`override`가 0이나 음수로 정해진 경우)
- x ≤ `maxComputeWorkgroupSizeX`, y ≤ `…SizeY`, z ≤ `…SizeZ`, x·y·z ≤ `maxComputeInvocationsPerWorkgroup` —
  한도는 어댑터가 알리는 값이다 (Metal: `maxThreadsPerThreadgroup`. 명세 기본값은 256 · 256 · 64 · 256)
- 커널 자체의 `maxTotalThreadsPerThreadgroup`(레지스터 사용량에 따라 기기 한도보다 낮을 수 있다)을 넘으면
  backend 오류 — Metal 단언으로 죽지 않게

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
- 별칭(`alias M = mat4x4f;`, `alias I2 = I1; alias I1 = Inner;`)은 가리키는 타입 그대로 배치한다 —
  몇 단을 거치든 끝까지 따라간다.

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
`(크기표[i] - 멤버오프셋) / sizeof(원소)`로 번역한다 (Dawn과 같은 방식). 뺄셈은 **0에서 멈춘다** —
바인딩이 멤버 오프셋보다 작으면 uint32 뺄셈이 ~40억으로 감겨 그 길이에 기댄 범위 자르기가 통째로
무력해지기 때문이다. 길이가 0인 배열(원소 하나도 못 담는 바인딩)에 대한 인덱스 쓰기는 버려진다
(명세가 허용하는 범위 밖 쓰기 처리). 길이는 **바인딩된 크기**를
따르므로 `{ buffer, offset, size }`로 일부만 묶으면 그만큼만 센다. 쓰지 않는 셰이더에는 표를 넘기지 않는다.
`let p = &particles.items; arrayLength(p)`처럼 `let` 포인터를 넘겨도 된다.

**`texture_external`** — 샘플링 가능한 2D 텍스처로 내려간다. 바인드 그룹에는 보통
`GPUTextureView`를 묶으면 된다 (한 면짜리 비디오 프레임과 같은 모양).
**JS API `importExternalTexture()`는 없다** — Lynx에 비디오 엘리먼트 핸들이 없기 때문이다
(`docs/WEBGPU-API.md` §8). 즉 **WGSL 쪽은 열려 있고 소스만 직접 올리면 된다**:
프레임을 `writeTexture`/`copyExternalImageToTexture`로 텍스처에 올린 뒤 그 뷰를 이 자리에 묶는다.
`textureSampleBaseClampToEdge`는 밉 0에서 좌표를 텍셀 절반만큼 안쪽으로 물려 샘플하므로,
`repeat` 샘플러라도 프레임 경계에서 반대쪽이 감겨 들어오지 않는다.

## 3-2. 안전 변환 — WGSL이 정의하는데 MSL에서는 정의되지 않은 자리

WGSL에는 값이 정해져 있는데 MSL(C++)에서는 **정의되지 않은 동작**인 지점들이 있다. 그대로
옮기면 드라이버가 무엇을 하든 이상하지 않고, 최적화가 "일어날 수 없는 일"로 보고 주변 코드를
지우기도 한다. 방출기가 이것들을 메운다 (Tint가 같은 이름의 변환들로 하는 일이다):

| 자리 | 무엇을 하나 | 왜 |
|---|---|---|
| **인덱싱 범위** (`a[i]` `v[i]` `m[i]`) | 상한으로 자른다 (`wgpu_at` · `wgpu_store`). 고정 크기는 C++ 템플릿이 크기를 알고, 런타임 크기 배열은 **버퍼 크기 표**로 구한다 | 범위를 벗어나면 인접 GPU 메모리를 읽거나 덮어쓴다. **인덱스는 대개 유니폼·스토리지에서 오므로 결국 번들(JS)이 정하는 값**이다 |
| 정수 `/` `%` | 분모가 0이거나 `INT_MIN / -1`이면 분모를 1로 바꾼다 (`wgpu_div` · `wgpu_mod`) | WGSL은 `x / 0 == x`, `x % 0 == 0`으로 정의한다. C++에서는 UB |
| `<<` `>>` | 시프트 폭의 하위 5비트만 남긴다 (`wgpu_shl` · `wgpu_shr`) | 폭 이상의 시프트가 UB |
| `i32()` `u32()` (f32에서) | 포화시킨다 (`wgpu_ftoi` · `wgpu_ftou`) | WGSL은 범위 밖 변환을 포화로 정의한다. C++ 캐스트는 UB |
| `var<workgroup>` | 진입점 앞에서 스레드들이 나눠 0으로 깔고 배리어를 친다 | WGSL은 0 초기화를 보장하지만 MSL의 threadgroup 저장소는 **이전 디스패치의 잔여 값**을 보여 준다 |
| `discard` | 플래그로 바꾸고 스토리지·텍스처 쓰기를 가린 뒤, 진입점 끝에서 실제로 폐기한다 | MSL의 `discard_fragment()`는 **즉시 종료가 아니다** — 뒤의 코드가 계속 돌아 버려진 프래그먼트가 스토리지를 오염시킨다. 끝에서 폐기하는 것은 같은 쿼드 이웃의 미분값을 살리기 위해서다 |
| `loop` / `while(true)` | 탈출 조건을 하나 더 붙인다 (u32 상한) | C++에서 무한 루프는 UB라, 컴파일러가 "끝난다"고 가정하고 주변 코드를 지울 수 있다 |

비용은 `min` 하나·분기 하나 수준이고, `discard`가 없는 셰이더에는 플래그 자체가 생기지 않는다.
검증은 `ShaderSafetyTests`(방출 + 실제 Metal 컴파일)와 `ShaderSafetyRenderTests`(**GPU에서 값으로**)가 한다.

**포인터를 거친 인덱싱은 어떤가** — 파서는 `ptr<…>`를 받는다(`WGSLParser`의 타입 파싱).
경우는 셋으로 갈린다:

- **고정 크기 배열**(`ptr<function, array<f32, 4>>` 등)은 **잘린다.** 자르기가 C++ 템플릿의
  타입 추론에 얹혀 있어서, `(*p)[i]`도 `wgpu_store((*p), i, …)`로 나가며 템플릿이 배열
  크기를 그대로 안다.
- **런타임 크기 배열을 가리키는 `let` 포인터**(`let p = &data; (*p)[i]`, `let q = &buf; (*q).items[i]`)는
  방출기가 포인터를 **전역 자리까지 되짚어** 같은 크기 표로 자른다 (`wgpu_at_n` · `wgpu_store_n`).
  `alias B = Buf;`처럼 별칭으로 쓴 저장 타입도 별칭을 풀어 같은 멤버를 찾는다.
- **상한을 뽑을 수 없는 모양** — `ptr<storage, Buf>` 매개변수처럼 호출자가 무엇을 가리킬지 정하는
  경우 — 은 방출기가 **번역 시점에 거부한다** ("WGSL: indexing a runtime-sized array through a pointer
  parameter is not supported"). 매개변수의 고정 크기 부분(`(*q).head`)은 그대로 쓸 수 있다.
  그 밖에 방출기가 놓친 모양도 **컴파일이 실패한다** — 프렐류드가 크기 없는 포인터용 `wgpu_at` ·
  `wgpu_store` 오버로드를 `= delete`로 막아 두었기 때문이다 (구조체 말미 `T items[1]`도 포인터로
  붕괴해 같은 곳에 걸린다). 조용히 범위 밖을 건드리는 대신 파이프라인 생성이 실패한다. `ptr<storage, array<f32>>` 매개변수는
  그 전에 방출기가 "WGSL: a runtime-sized array (array<T>) can only be used in a storage buffer binding"으로 거부한다.

즉 런타임 크기 배열의 상한은 **방출기가 크기 표 자리를 찾은 경우에만** 붙고, 못 찾은 경우는
프렐류드가 막는다. 새 접근 모양을 지원하려면 `MSLEmitter.storagePlace(of:)`를 넓히면 된다.

## 4. 지원하지 않는 것

명시적으로 **거부**한다 (조용히 틀리게 번역하지 않는다):

| 기능 | 이유 / 대안 |
|---|---|
| `atomicCompareExchangeWeak` | 반환 구조체를 옮기지 못한다 |
| `modf` / `frexp` | 반환 구조체를 옮기지 못한다. `floor`/`fract`로 나눠 쓸 것 |
| `workgroupUniformLoad` | 미지원 |
| `textureGather` / `textureGatherCompare` | WGSL은 성분 인덱스가 첫 인자, MSL `gather()`는 템플릿 인자라 이름 그대로는 옮겨지지 않는다 — Metal 컴파일 오류로 드러난다 |
| `firstLeadingBit` / `firstTrailingBit` · `dot4U8Packed` / `dot4I8Packed` · `pack4xI8` / `pack4xU8` / `pack4xI8Clamp` / `pack4xU8Clamp` / `unpack4xI8` / `unpack4xU8` | MSL 대응 이름이 없다 — Metal 컴파일 오류로 드러난다 |

`loop { … continuing { … } }`에서 `continue`와 `break if`는 **지원한다** — 본문을 `do { … } while (false)`로
감싸 `continue`가 `continuing`으로 떨어지게 하고, 본문의 `break`는 플래그를 세워 바깥 `while`을
빠져나간다 (Naga·Tint가 모든 `for`/`while`에 뱉는 모양이라 wgpu·Bevy를 거친 셰이더가 전부 이 길을 밟는다).
`round`는 WGSL대로 짝수 반올림(`rint`)이다 — MSL `round`(반올림 시 0에서 먼 쪽)와 다르다.
| 재귀 타입 (`struct S { a: array<S, 2> }`, 별칭·다른 구조체를 거친 순환 포함) | WGSL이 금지한다(크기가 무한). **셰이더 모듈 생성 시** 선언 줄 번호와 순환 경로(`S → A → S`)를 담은 검증 오류 |
| 타입·식·문장 중첩 128단 초과 (`((((…))))`, `array<array<…>>`, `{ { … } }`. `else if` 사슬의 길이는 중첩이 아니라 세지 않는다) | 파서가 재귀 하강이라 스택을 다 쓰지 않도록 묶는다 (Tint 파서와 같은 한도). 줄 번호를 단 검증 오류. 한도 안의 셰이더는 호출 스레드의 스택과 무관하게 번역된다 — 파싱·방출은 16MB 스택의 전용 스레드에서 돈다 (`WGSLDeepStack`) |
| 식 트리 높이 1024단 초과 (`a + b + c + …` · `v.x.x.…` · `m[0][0]…`처럼 **평탄하게 긴 사슬**) | 중첩 한도는 파서 재귀만 세고 사슬은 루프로 만들어지는데, 그 트리를 리플렉션·방출기가 항마다 재귀로 걷는다 — 10만 항에서 방출기 스택이 넘쳐 **호스트가 죽었다.** 그래서 파서가 완성된 식의 높이를 (재귀 없이) 재어 `WGSLParser.maximumExpressionHeight`(1024)에서 끊는다. 줄 번호를 단 검증 오류. 실제 셰이더는 이 근처에도 오지 않는다 |
| 합성 타입 중첩 255단 초과 | 배치 계산·방출이 타입을 재귀로 걷으므로 깊이를 묶는다 (Tint와 같은 한도. 명세 보장치는 15) |

암묵적 제약:
- **성분 타입이 생략되고 인자가 전부 정수 상수식인 벡터 생성자**(`vec2(4, 1)`, `vec3(1)`)는
  WGSL 명세대로 **쓰이는 자리의 타입으로 굳는다** — f32 자리에서는 f32, u32 자리에서는 u32.
  프렐류드의 `wgpu_aint<N>` 대역이 그 결정을 C++ 오버로드 해석에 넘긴다 (§2-1).
  단 대역 값을 **Metal 내장 함수에 그대로 넘기는 자리**(`normalize(vec3(1))`)는 타입 추론이
  걸릴 수 있다. 그럴 때는 `vec3f(1)`처럼 성분 타입을 적으면 된다.
- 인자를 **자리로 골라 옮기는** 내장 함수(`atomic*` · `texture*` · `bitcast` · `arrayLength` ·
  `quantizeToF16` · 배리어)는 방출 전에 인자 개수를 명세 범위와 대조한다
  (`MSLTypeMapping.builtinArities`). 어긋나면 `"WGSL: atomicLoad() expects 1 argument, got 0"`
  같은 검증 오류로 파이프라인 생성이 실패한다 — 셰이더 소스는 번들이 주는 값이므로 호스트가 죽으면 안 된다.
- **타입을 적지 않은 모듈 스코프 `const`는 `#define` 매크로로 나간다** (MSL에 `constant auto`가 없어서다 —
  타입을 적었거나 초기식에서 추론되면 `constant T` 전역). 그래서 `const x1 = (x0 + x0) / 2u; const x2 = (x1 + x1) / 2u; …`처럼
  참조가 갈라지는 사슬은 Metal **전처리기**에서 줄 수에 지수로 펼쳐진다 — 18단에 2.4초, 26단은 20분 넘게 걸린다.
  평가기 쪽 메모이제이션(위)과는 별개의 한계다. 그런 상수에는 타입을 적을 것 (`const x1: u32 = …`).
- 여기 표에 없는 내장 함수는 **이름 그대로 통과**한다. MSL에 같은 이름이 없으면 파이프라인 생성 시
  "MSL compilation failed" 오류와 함께 생성된 MSL 전문이 나온다.

## 4-1. 실제 셰이더로 재 본 호환성

공식 [webgpu-samples](https://github.com/webgpu/webgpu-samples)의 WGSL 69개를 그대로 통과시켜 본 결과
(번역 + **실제 Metal 컴파일**까지):

| 결과 | 수 |
|---|---|
| 그대로 통과 | **61 / 67 (91%)** — 그대로 60 + 같은 폴더 조각을 붙여 1 |
| 호스트가 `constants`를 주면 동작 (`override` 사용) | 4 (분모에 포함) |
| 실패 | 2 |

**남은 2건은 트랜스파일러의 빈틈이 아니다**: `skinnedMesh/gltf.wgsl`은 호스트가 glTF 접근자를 보고
`VertexInput`을 **런타임 생성**해 붙이고, `wireframe/wireframeBufferView.wgsl`은 `requires buffer_view;`로
**WGSL 확장**(코어 명세 밖의 `bufferView<T>()`)을 요구한다 — 지원하려면 확장 자체를 구현해야 한다.

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
