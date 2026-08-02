# Lynx 연동 가이드

호스트 앱에 이 라이브러리를 붙이는 절차.

## 1. 패키지 추가

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/xenonClient/Lynx-WebGPU", from: "0.3.0"),
],
targets: [
    .target(name: "MyApp", dependencies: [
        .product(name: "LynxWebGPUBridge", package: "Lynx-WebGPU"),
    ]),
]
```

Xcode에서는 File ▸ Add Package Dependencies로 저장소 URL을 넣는다.

두 개의 product가 있다:

| product | 내용 | 언제 |
|---|---|---|
| `LynxWebGPUBridge` | 엔진 + Lynx NativeModule + `<webgpu-canvas>` | Lynx 번들의 JS가 GPU를 쓸 때 |
| `LynxWebGPU` | 엔진만 (Metal 백엔드 + WGSL 트랜스파일러) | Lynx 없이 Swift에서 WebGPU 커맨드를 실행할 때 |

Lynx SDK는 [xenonClient/Lynx-XCFramework](https://github.com/xenonClient/Lynx-XCFramework) 4.0.0을
transitive 의존성으로 끌어온다 — 8개 xcframework(Lynx/LynxBase/LynxService/LynxServiceAPI/PrimJS/
SDWebImage/SDWebImageWebPCoder/libwebp)가 `Lynx` 제품 하나로 묶여 있고, device(arm64) + simulator
슬라이스를 모두 포함하므로 **실기기 빌드도 된다**.

> 호스트 앱이 이미 Lynx를 다른 방식(CocoaPods 등)으로 링크하고 있다면 심볼이 중복된다.
> 그 경우 `LynxWebGPU` product만 쓰고, `Sources/LynxWebGPUBridge/`의 네 파일
> (`LynxWebGPUHost` / `WebGPUNativeModule` / `WebGPUCanvasUI` / `WebGPUFrameTicker`)을
> 앱 타깃으로 복사해 기존 Lynx에 대고 컴파일할 것.

Xcode 빌드 설정 **User Script Sandboxing = NO** (Lynx 공식 가이드 요구사항).

## 2. 호스트 코드

```swift
import Lynx
import LynxWebGPU
import LynxWebGPUBridge

final class GPUPageViewController: UIViewController {
    private var host: LynxWebGPUHost!
    private var lynxView: LynxView!

    override func viewDidLoad() {
        super.viewDidLoad()
        LynxEnv.sharedInstance()                     // 앱 시작 시 1회면 충분하다

        host = try! LynxWebGPUHost()                 // MTLDevice·큐·객체 레지스트리를 소유한다

        lynxView = LynxView { [host] builder in
            let config = LynxConfig(provider: MyTemplateProvider())
            LynxWebGPU.register(in: config, host: host!)   // 모듈 + <webgpu-canvas> 등록
            builder.config = config
            builder.screenSize = UIScreen.main.bounds.size
        }
        host.attach(to: lynxView)                    // ← 필수: 전역 이벤트/캔버스 연결

        view.addSubview(lynxView)
        lynxView.loadTemplate(fromURL: bundlePath, initData: nil)
    }

    deinit {
        host.detach()                                // 디스플레이 링크 정지 + GPU 객체 해제
    }
}
```

세 줄이 전부다:
1. `LynxWebGPUHost()` — 런타임 생성
2. `LynxWebGPU.register(in:host:)` — `NativeModules.WebGPU`와 `<webgpu-canvas>` 등록
3. `host.attach(to:)` — LynxView 연결 (**빠뜨리면** 캔버스 등록과 프레임 이벤트가 동작하지 않는다)

페이지를 떠날 때 `host.detach()`를 부르지 않으면 `CADisplayLink`가 계속 돌고 GPU 객체가 남는다.

## 3. Lynx 번들(JS) 쪽

`JS/webgpu.js`, `JS/webgpu.d.ts`, `JS/elements.d.ts`를 rspeedy 프로젝트의 `src/` 아래로 복사한다.
사용법은 `docs/JS-AUTHORING.md`, 최소 예제는 `Examples/HelloTriangle.tsx` 참고.

### 애셋 읽기 (`loadAsset`)

텍스처로 올릴 픽셀처럼 **JS 소스에 박기엔 큰 데이터**는 `loadAsset`으로 가져온다.
브라우저의 `fetch()`가 하던 역할을 최소한으로 대신한다.

```js
import { loadAsset } from './webgpu.js'

const buffer = await loadAsset('hdr-sample.bin')   // ArrayBuffer
device.queue.writeTexture({ texture }, new Uint8Array(buffer, offset, length), …)
```

이름 해석은 호스트의 `assetProvider`(`WGPUAssetProvider`)가 정한다 — Lynx가 번들 로딩을
`LynxTemplateProvider`에 맡기는 것과 같은 구조다. **기본 공급자**(`WGPUFileAssetProvider`)는
순서대로:

1. **등록된 이름** — 이미지 피커처럼 파일이 아니라 `Data`로 오는 것의 통로.
   ```swift
   // PHPicker 결과를 JS에 내줄 때
   provider.register(pickedData, for: "picked-image")
   ```
2. **절대 경로 · `file://` URL** — 피커·다운로드가 준 파일 URL 문자열을 JS에 넘기고
   (initData·이벤트 등으로) 그대로 `loadAsset`에 쓴다.
3. **번들 상대 경로** (`'hdr-sample.bin'`, `'LUTs/neutral.cube'`) — 앱 타깃의 리소스.
   Tuist라면 `resources: ["Resources/**"]`, 추가 후 `tuist generate` 재실행.
   `..`이나 숨김 이름은 거부한다.

**접근 범위는 기본이 전체 허용이다.** 번들(JS)을 신뢰할 수 없는 앱 — 서버에서 내려받는
번들 등 — 은 반드시 좁힐 것:

```swift
// 파일 경로 접근을 특정 디렉토리 아래로 제한 (심볼릭 링크를 풀어 비교한다)
host.assetProvider = WGPUFileAssetProvider(
    allowedRoots: [FileManager.default.temporaryDirectory]
)

// SPM 라이브러리가 동봉한 리소스를 내주려면 그쪽 번들을 지정
host.assetProvider = WGPUFileAssetProvider(bundle: .module)

// 해석 규칙 자체를 바꾸려면 프로토콜을 직접 구현한다
final class HandleOnlyProvider: WGPUAssetProvider { … }
```

- 네이티브가 `Data`로 돌려주면 Lynx가 `ArrayBuffer`로 바꿔 준다. base64 왕복이 없다.
- 파일 읽기는 백그라운드 큐에서 하고 콜백으로 돌려준다 — 수 MB짜리를 JS 스레드에서
  동기로 읽으면 프레임이 밀린다. 공급자를 직접 구현할 때도 같은 규칙을 지킬 것.

## 4. `<webgpu-canvas>`

```tsx
<webgpu-canvas
  canvas-id="main"
  pixel-ratio={1}
  bindcanvasresize={(e) => console.log(e.detail.width, e.detail.height)}
  style={{ width: '100%', height: '300px' }}
/>
```

| prop | 설명 |
|---|---|
| `canvas-id` (필수) | JS가 `gpu.getCanvasContext(id)`로 지목할 이름. 페이지 안에서 유일해야 한다 |
| `pixel-ratio` | CSS px → 드로어블 픽셀 배율. 생략하면 화면 배율. 부하가 크면 1로 낮춘다 |

| 이벤트 | detail |
|---|---|
| `bindcanvasresize` | `{ width, height, pixelRatio }` — 드로어블 픽셀 크기가 바뀔 때 |

UI 메서드 `getInfo()` → `{ canvasId, width, height, pixelRatio }` (resize 이벤트를 놓쳤을 때의 폴백).

내부적으로 뷰의 백킹 레이어 자체가 `CAMetalLayer`다 (`layerClass` 교체) — 서브레이어가 없어 합성 단계가 하나 줄어든다.

## 5. 터치·제스처 — 웹과 같은 규칙

`<webgpu-canvas>`는 **터치 이벤트를 따로 만들지 않는다.** Lynx 표준 이벤트를 그대로 쓴다:

```tsx
<webgpu-canvas
  canvas-id="main"
  bindtouchstart={(e) => { const t = e.touches[0]; /* t.x, t.y = 엘리먼트 기준 CSS px */ }}
  bindtouchmove={…}
  bindtouchend={…}
/>
```

이유가 중요하다. `UIView.touchesBegan`을 가로채면 동작은 하지만 **Lynx의 이벤트 라우팅을 통째로 우회한다** —
그러면 웹과 다르게 동작한다. Lynx의 `LynxEventHandler`는 자체 `hitTest`로 UI 트리를 훑으며
`pointer-events`, 페인트 순서(z-order), 버블링, `catch` 접두사, 제스처 아레나(스크롤 경쟁)를 모두 반영한다.
그 경로를 타야 다른 엘리먼트와 똑같이 굴러간다.

따라서 다음이 **웹과 같게** 보장된다:

| 상황 | 동작 |
|---|---|
| 캔버스 **위에 겹친** Lynx 엘리먼트를 누름 | 위 엘리먼트가 가져간다. 캔버스는 못 받는다 |
| 겹친 엘리먼트에 `pointer-events: none` | 통과해서 캔버스가 받는다 |
| 캔버스 이벤트에서 `stopPropagation()` | 부모로 버블링하지 않는다 |
| `catchtouchstart`로 바인딩 | 버블링이 그 지점에서 멈춘다 |
| `<scroll-view>` 안의 캔버스 | 스크롤 제스처가 이기면 터치가 취소된다 |

좌표 변환은 웹과 같다 — `touches[0].x/y`가 **엘리먼트 기준**이므로 캔버스 CSS 크기로 나누면 0~1이 된다.
CSS 크기는 `bindcanvasresize`의 `width / pixelRatio`로 얻는다 (detail의 크기는 **드로어블 픽셀**이다).

> 확인은 `interactive` 데모 씬으로 한다 — 캔버스 위에 Lynx 카드와 하단 바를 겹쳐 두고,
> 어떤 엘리먼트가 마지막 입력을 가져갔는지 화면에 표시한다.

## 6. 스레딩 계약

| 일 | 스레드 |
|---|---|
| `NativeModules.WebGPU.execute()` (커맨드 해석 + Metal 인코딩) | Lynx JS 백그라운드 스레드 |
| `CAMetalLayer` 프로퍼티 설정, 드로어블 크기 갱신 | 메인 (`configure`는 비동기로 넘어간다) |
| `CADisplayLink` 프레임 이벤트 | 메인 (이벤트 전송만) |
| `readBuffer` GPU 완료 대기 | 전용 readback 큐 |

호스트 코드에서 `LynxWebGPUHost`의 메서드는 어느 스레드에서 불러도 안전하다.

## 7. 확인

시뮬레이터에서 번들을 띄우고 다음을 본다:

1. `<webgpu-canvas>` 자리에 클리어 색이 칠해지는가 → 표면 등록·configure까지 정상
2. `bindcanvasresize`의 width/height가 화면 배율만큼 커져 있는가 → 드로어블 크기 정상
3. Xcode 콘솔의 `org.lynxwebgpu` 서브시스템 로그 (`Canvas`/`Device` 카테고리)
4. JS 콘솔에 `[WebGPU:validation] …` 이 뜨는가 → 커맨드 스트림 오류 (경로가 함께 나온다)

네이티브 쪽 렌더 파이프라인 자체는 **호스트 앱 없이** 오프스크린 하네스로 검증한다 (`docs/TESTING.md`).
