# `readPixels` 마이그레이션 가이드 (2026-08-01)

`WGPUOffscreenSurface.readPixels(queue:)`의 반환 타입이 `Data` → `WGPUPixelReadback`으로 바뀌었다.
호출하는 코드가 있으면 **컴파일 오류로 걸린다** — 조용히 동작이 달라지지는 않는다.

## 1. 왜 바꿨나

예전 구현은 픽셀당 4바이트를 **하드코딩**하고 있었다.

```swift
let bytesPerRow = texture.width * 4      // ← rgba8unorm 에서만 맞는 값
```

`rgba8unorm` 계열에서는 맞지만, `rgba16float`(픽셀당 8바이트) 표면에서는

- 되읽은 **길이가 절반**이라 아래쪽 절반이 잘리고,
- 그 절반마저 half-float 비트를 8비트 정수로 읽은 **의미 없는 값**이며,
- 그런데도 **오류가 나지 않는다.**

밝기가 좀 이상한 이미지가 나올 뿐이라, 원인을 이 함수까지 되짚기가 어렵다.
가장 나쁜 부분이 "조용히 틀린 값"이었다.

돌려주는 타입도 문제였다. `Data`만 넘기면 호출 측이 "RGBA8일 것"을 가정할 수밖에 없고,
그 가정은 코드 어디에도 적혀 있지 않다. 그래서 **해석에 필요한 것을 값과 함께 묶었다.**

## 2. 무엇이 바뀌었나

```swift
// 전
public func readPixels(queue: MTLCommandQueue) throws -> Data

// 후
public func readPixels(queue: MTLCommandQueue) throws -> WGPUPixelReadback
```

`WGPUPixelReadback`은 `LynxWebGPUCore`에 있다 (Metal 불필요 — GPU 없이 단위 테스트할 수 있다):

```swift
public struct WGPUPixelReadback: Sendable, Equatable {
    public let data: Data                  // 행 하나가 bytesPerRow 바이트, 행이 height개
    public let format: WGPUTextureFormat   // 이걸 보지 않고 해석하면 안 된다
    public let width: Int
    public let height: Int
    public let bytesPerRow: Int            // 패딩이 있으면 width * bytesPerPixel 보다 크다

    public var bytesPerPixel: Int { format.bytesPerPixel }

    /// 픽셀 하나를 RGBA float 으로 편다.
    public func rgba(x: Int, y: Int) throws -> SIMD4<Float>
}
```

행 간격은 이제 표면 포맷에서 나온다 (`texture.width * format.bytesPerPixel`).

## 3. 호출부 고치기

### 3-1. 바이트를 그대로 쓰던 코드

```swift
// 전
let data = try surface.readPixels(queue: queue)
let offset = (y * width + x) * 4
let r = data[offset]

// 후 — 행 간격과 픽셀 크기를 readback 에서 가져온다
let readback = try surface.readPixels(queue: queue)
let offset = y * readback.bytesPerRow + x * readback.bytesPerPixel
let r = readback.data[offset]
```

`data`만 필요하면 `try surface.readPixels(queue: queue).data` 한 줄이면 되지만,
**그렇게 쓸 거라면 포맷을 어디선가 고정해 두어야 한다.** 그게 이 변경이 없애려던 가정이다.

### 3-2. 채널 값이 필요한 코드 (권장)

```swift
let readback = try surface.readPixels(queue: queue)
let color = try readback.rgba(x: 32, y: 32)   // SIMD4<Float>
// rgba16float 이면 (2.5, 0.5, -0.25, 1.0) 처럼 SDR 범위 밖 값이 그대로 나온다
```

`rgba(x:y:)`의 규칙:

- 없는 채널은 **RGB가 0, A가 1**로 채워진다 (셰이더에서 텍스처를 읽을 때와 같다).
- `bgra8unorm` 계열은 **RGBA 순서로 바꿔서** 돌려준다.
- **색공간 변환은 하지 않는다.** `-srgb` 포맷도 저장된 값을 0~1로 정규화할 뿐이다.
  선형 값이 필요하면 호출 측이 변환한다.
- float 포맷은 정규화하지 않는다 → **1.0 초과·음수가 그대로 살아 나온다.**

### 3-3. 이미지로 굽던 코드

CGImage/PNG를 만들려면 8비트로 내려야 한다. `rgba()`로 편 뒤 0~1로 잘라 쓴다:

```swift
var bytes = Data(capacity: width * height * 4)
for y in 0..<readback.height {
    for x in 0..<readback.width {
        let color = try readback.rgba(x: x, y: y)
        for channel in 0..<4 {
            bytes.append(UInt8((min(max(color[channel], 0), 1) * 255).rounded()))
        }
    }
}
```

HDR 원본을 8비트로 내릴 때 **그냥 자르면 하이라이트가 뭉갠다.** 톤 매핑이 필요하면
자르기 전에 여기서 건다 (이 저장소는 톤 매핑을 제공하지 않는다 — 정책은 호출 측 몫이다).

## 4. 오류가 나는 경우

`readPixels`가 던지는 조건:

| 조건 | 오류 |
|---|---|
| `configure` 전 | `validation` — "the surface has not been configured yet" |
| depth/stencil 포맷 표면 | `validation` — 포맷 이름이 메시지에 들어간다 |

depth/stencil을 거부하는 것은 임의의 제약이 아니다. Metal blit은 aspect 지정 없이
depth/stencil을 한 덩어리로 복사하지 못하고, `depth32float-stencil8`은 픽셀당 바이트가
**연속된 한 블록으로 존재하지도 않는다.** 깊이 버퍼를 읽어야 한다면 별도 경로가 필요하다
(지금은 없다).

`rgba(x:y:)`가 던지는 조건:

| 조건 | 오류 |
|---|---|
| 좌표가 범위 밖 | `validation` |
| 바이트가 모자람 (`bytesPerRow`가 틀린 경우 등) | `validation` |
| 채널로 풀 수 없는 포맷 | `validation` — "cannot be expanded by rgba(x:y:) — interpret data directly" |

마지막 항목은 `rgb10a2unorm`·`rg11b10ufloat`처럼 비트가 채널 경계를 넘어 팩된 포맷과
정수 포맷(`rgba8uint` 등)이다. **`readPixels` 자체는 이들도 정상적으로 읽는다** — 바이트는
정확하고 포맷도 함께 알려 주므로, 해석만 호출 측이 하면 된다.

## 5. 되읽기가 되는 포맷

| 포맷 | `readPixels` | `rgba(x:y:)` |
|---|---|---|
| `r8unorm` / `rg8unorm` / `rgba8unorm`(+`-srgb`) | O | O |
| `bgra8unorm`(+`-srgb`) | O | O (RGBA 순서로 교정) |
| `r8snorm` / `rg8snorm` / `rgba8snorm` | O | O |
| `r16float` / `rg16float` / `rgba16float` | O | O |
| `r32float` / `rg32float` / `rgba32float` | O | O |
| `rgb10a2unorm` / `rg11b10ufloat` | O | X (팩된 비트) |
| 정수 포맷 (`*uint` / `*sint`) | O | X |
| depth / stencil | X | — |

## 6. 테스트 하네스 쪽 변화

`RenderHarness`(테스트 전용)도 함께 바뀌었다:

| 추가 | 용도 |
|---|---|
| `readback()` | 포맷·행 간격까지 통째로 |
| `pixelFloat(x:y:)` | 채널 값 그대로 (`SIMD4<Float>`) |
| `assertPixelFloat(x:y:equals:tolerance:)` | 기본 허용 오차 0.01 |

기존 `pixel(x:y:)` / `assertPixel(...)`(0~255 정수)는 **시그니처가 그대로**다. 다만 내부적으로
`rgba()`를 거치므로 **8비트 표면용**으로 봐야 한다 — float 표면에 쓰면 2.5가 638로 나온다.

`dumpPNG(named:)`도 어떤 포맷이든 굽는다 (float 표면은 0~1로 잘라서).

## 7. 회귀 가드

- `Tests/LynxWebGPUCoreTests/WGPUPixelReadbackTests.swift` — 해석 쪽. GPU가 필요 없다.
  half→float(서브노멀·Inf·NaN 포함), 1.0 초과·음수 보존, BGRA 순서, 행 패딩,
  **"같은 바이트를 다른 포맷으로 읽으면 다르게 나온다"**(이번 버그의 본질), 팩된/정수 포맷 거부.
- `Tests/LynxWebGPUTests/OffscreenReadbackTests.swift` — 표면 쪽.
  포맷별 행 간격·길이(1~16B/픽셀), depth/stencil 거부, configure 전 거부.
- `Tests/LynxWebGPUTests/RenderPipelineTests.swift` —
  `test_anRGBA16FloatSurfaceDoesNotLoseValuesOutsideSDR`.
  실제로 `rgba16float` 표면에 그리고 되읽어, 프래그먼트가 쓴 `(2.5, 0.5, -0.25, 1.0)`과
  클리어 값 `(4, 0, 0, 1)`이 살아 있는지 본다.

## 8. 알려진 한계

- **되읽기 자체가 동기다.** `waitUntilCompleted`로 GPU를 기다린다. 테스트와 익스포트처럼
  프레임 예산 밖에서 도는 경로를 전제한 API다.
- **패딩 없는 행 간격만 만든다.** `readPixels`는 항상 `width * bytesPerPixel`로 채운다.
  `bytesPerRow`를 굳이 노출한 것은 나중에 정렬 요구가 생겨도 호출부가 안 깨지게 하기 위해서다.
- **톤 매핑·색공간 변환은 제공하지 않는다.** 값을 있는 그대로 돌려주는 데까지가 이 API의 책임이다.
