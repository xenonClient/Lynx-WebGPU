# Lynx 연동 가이드

호스트 앱에 이 라이브러리를 붙이는 절차.

## 1. 이 패키지는 Lynx를 가져오지 않는다

`Lynx-WebGPU`는 **외부 의존성이 0이다.** Lynx SDK의 버전과 배포처는 **앱이 정한다** —
SPM으로 받든, CocoaPods로 이미 쓰고 있든, 사내 배포본을 물리든, 다른 버전이 필요하든 그대로 붙는다.

그래서 패키지가 주는 SPM product는 둘이다:

| product | 내용 | 언제 |
|---|---|---|
| `LynxWebGPUCore` | 커맨드 스트림의 **계약** — `WebGPURuntime` 프로토콜, 디스크립터·커맨드 디코딩, 오류 모양. GPU 코드가 없다 | 항상 (브리지가 이것만 본다) |
| `LynxWebGPU` | 기본 런타임 — Metal 백엔드 + WGSL 트랜스파일러 (Lynx 무관) | 기본 엔진을 쓸 때 |

**GPU 백엔드도 앱이 정한다.** 브리지는 `WebGPURuntime`(=`LynxWebGPUCore`)만 알고, 실제
런타임 객체는 앱이 만들어 `LynxWebGPUHost(runtime:)`에 넣는다 — Lynx SDK를 여기서 가져오지
않는 것과 **같은 이유**다. 그래서 다른 백엔드(예: [Dawn](https://github.com/google/dawn) 위에
얹은 런타임)로 갈아끼워도 **브리지도 JS 번들도 손대지 않는다.**

Lynx 연동 레이어(`LynxWebGPUHost` · `WebGPUNativeModule` · `WebGPUCanvasUI` · `WebGPUFrameTicker`)는
`Sources/LynxWebGPUBridge/`에 **소스로** 들어 있고, 네 파일 전부가 `#if canImport(Lynx)` 안에 있다.
**Lynx가 보이는 타깃에서 컴파일하면 켜지고, 아니면 조용히 비어 있다.**

> **왜 SPM 타깃이 아닌가.** SPM 타깃의 `canImport(Lynx)`는 **매니페스트가 선언한 의존성만**
> 본다. 앱이 Lynx를 다른 경로(CocoaPods 등)로 링크해도 우리 타깃의 컴파일에는 보이지 않으므로,
> 브리지를 SPM 타깃으로 두면 그런 앱에서는 **영원히 빈 모듈**이 된다 — "product를 추가했는데
> API가 없다"가 되는 것보다, 소스를 앱 쪽에서 컴파일하게 하는 편이 정직하다.
> 반대로 여기서 Lynx를 선언해 버리면 버전이 패키지에 박혀 위 목적이 깨진다.

## 2. 연동 경로

어느 쪽이든 **엔진은 SPM으로, 브리지는 소스로** 가져간다는 모양은 같다.

### 2-1. 별도 브리지 타깃 (Tuist / Xcode 프로젝트)

앱 쪽에 브리지 타깃을 하나 만들고 거기서 소스를 컴파일한다. **데모 앱이 이 모양 그대로다**
(`Projects/WebGPUDemo/Project.swift` — 복사해 쓰면 된다):

```swift
packages: [
    .remote(url: "https://github.com/xenonClient/Lynx-WebGPU", requirement: .upToNextMajor(from: "0.4.0")),
    // Lynx는 **앱이 고른다.** 아래는 데모가 쓰는 배포본일 뿐이고, 다른 저장소·다른 버전으로
    // 바꿔도 브리지 소스는 그대로 컴파일된다.
    .remote(url: "https://github.com/xenonClient/Lynx-XCFramework", requirement: .exact("4.0.0")),
],
targets: [
    .target(
        name: "LynxWebGPUBridge",
        product: .staticFramework,
        sources: ["<체크아웃 경로>/Sources/LynxWebGPUBridge/**"],
        dependencies: [
            // 브리지는 계약만 본다 — GPU 백엔드를 모른다.
            .package(product: "LynxWebGPUCore"),
            .package(product: "Lynx"),        // ← 여기서 Lynx 버전을 고른다
        ]
    ),
    .target(name: "MyApp", dependencies: [
        .target(name: "LynxWebGPUBridge"),
        .package(product: "LynxWebGPU"),      // ← 여기서 GPU 백엔드를 고른다
    ]),
]
```

Xcode 프로젝트라면 같은 일을 GUI로 한다 — 프레임워크 타깃을 만들고, 브리지 네 파일을
Compile Sources에 넣고, `LynxWebGPU`와 Lynx를 그 타깃의 의존성에 넣는다.

### 2-2. 앱 타깃에 바로 넣기 (가장 간단)

별도 모듈이 필요 없으면 브리지 네 파일을 **앱 타깃에 그대로** 추가한다. 이때는
`import LynxWebGPUBridge`가 필요 없다 (타입이 앱 모듈에 들어온다).

**Lynx를 CocoaPods로 쓰는 앱도 이 경로다** — 앱 타깃에서 Lynx가 보이므로 가드가 켜진다.

```
Sources/LynxWebGPUBridge/
├── LynxWebGPUHost.swift        — 런타임(WebGPURuntime) 보유 + 프레임 티커·캔버스 연결
├── WebGPUNativeModule.swift    — NativeModules.WebGPU
├── WebGPUCanvasUI.swift        — <webgpu-canvas> 엘리먼트
└── WebGPUFrameTicker.swift     — CADisplayLink 프레임 틱
```

### 공통

Xcode 빌드 설정 **User Script Sandboxing = NO** (Lynx 공식 가이드 요구사항).

> **심볼 중복이 사라진다.** 예전에는 이 패키지가 Lynx를 transitive로 끌어와, 앱이 이미 Lynx를
> 링크하고 있으면 충돌했다. 이제 Lynx를 가져오는 곳은 앱 하나뿐이다.

## 3. 호스트 코드

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

        // **런타임(GPU 백엔드)을 고르는 유일한 자리.** 기본은 Metal 엔진이다.
        host = LynxWebGPUHost(runtime: try! LynxWebGPUContext())

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
1. `LynxWebGPUHost(runtime:)` — 런타임 주입 (`LynxWebGPUContext`가 기본 Metal 구현이다)
2. `LynxWebGPU.register(in:host:)` — `NativeModules.WebGPU`와 `<webgpu-canvas>` 등록
3. `host.attach(to:)` — LynxView 연결 (**빠뜨리면** 캔버스 등록과 프레임 이벤트가 동작하지 않는다)

페이지를 떠날 때 `host.detach()`를 부르지 않으면 `CADisplayLink`가 계속 돌고 GPU 객체가 남는다.

### 프레임 루프는 누가 시작하나 — `attach`만으로는 돌지 않는다

`attach(to:)`는 틱 콜백을 **배선만** 한다. 디스플레이 링크는 **JS가 프레임을 요청할 때** 돈다:

```
JS  startFrameLoop(handler)          (JS/webgpu.js)
 └▶ NativeModules.WebGPU.startFrameLoop({fps})
     └▶ LynxWebGPUHost.startFrameLoop(preferredFramesPerSecond:)  →  CADisplayLink 시작
         └▶ 매 틱: runtime.processEvents() → isReadyForNextFrame 검사 → `webgpu:frame` 전역 이벤트
```

프레임을 쓰지 않는 페이지에서 링크가 헛도는 것을 막는 배치다. 그래서 **애니메이션 없는
씬은 링크가 아예 돌지 않는 것이 정상**이고, 그런 씬에서도 `mapAsync` 완료가 굶지 않도록
펌프가 필요한 백엔드는 엔진이 자체 펌프를 돌린다 (`WGPUBackendCapabilities.needsEventPump`).

**"캔버스는 뜨는데 화면이 검다"면 이 사슬부터 본다** — 번들이 `startFrameLoop`(또는 그 위의
`requestAnimationFrame`)을 부르는지, 링크가 서 있는지. 호스트가 JS 밖에서 프레임을 몰아야
한다면 `host.startFrameLoop(preferredFramesPerSecond:)`를 직접 부를 수 있다 (public이다 —
네이티브가 커맨드 스트림을 직접 만드는 구성이나 JS 경로 진단용).

## 4. Lynx 번들(JS) 쪽

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

## 5. `<webgpu-canvas>`

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
| `passthrough-touches` | UIKit 터치 통과 (기본 꺼짐). 캔버스 **뒤** 네이티브 제스처가 필요할 때 켠다 — §6 참고 |

| 이벤트 | detail |
|---|---|
| `bindcanvasresize` | `{ width, height, pixelRatio }` — 드로어블 픽셀 크기가 바뀔 때 |

UI 메서드 `getInfo()` → `{ canvasId, width, height, pixelRatio }` (resize 이벤트를 놓쳤을 때의 폴백).

내부적으로 뷰의 백킹 레이어 자체가 `CAMetalLayer`다 (`layerClass` 교체) — 서브레이어가 없어 합성 단계가 하나 줄어든다.

## 6. 터치·제스처 — 웹과 같은 규칙

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

### 캔버스가 스크롤뷰 **위에** 겹칠 때 (`passthrough-touches`)

반대 방향 — 캔버스가 다른 네이티브 뷰 기반 엘리먼트를 덮는 경우다. 구조에 따라 갈린다:

- **스크롤뷰가 캔버스의 조상**이면 (`<scroll-view>` 안에 캔버스) 아무것도 필요 없다.
  UIScrollView의 팬 인식기는 조상 뷰에 붙어 있어 캔버스에 떨어진 터치도 받는다.
  가능하면 이 구조를 먼저 검토할 것 — 웹과 완전히 같은 동작이다.
- **스크롤뷰가 형제로 캔버스 뒤에** 있으면 기본값에서는 스크롤이 막힌다.
  이것도 웹과 같다 — 겹친 캔버스는 아래 형제의 포인터를 가린다. 웹과 달리 뚫고
  싶을 때만 `passthrough-touches`를 켠다:

  ```tsx
  <webgpu-canvas canvas-id="overlay" passthrough-touches
    bindtouchstart={…} />   // Lynx 이벤트는 계속 온다
  ```

  캔버스 백킹 뷰가 UIKit 히트 테스트에서 빠져 아래 네이티브 제스처가 통과한다.
  **캔버스 자신의 Lynx 이벤트는 계속 온다** — Lynx의 터치 인식기는 개별 뷰가 아니라
  rootView(LynxView)에 붙어 있고 타깃 결정도 Lynx 자체 hitTest가 하기 때문이다.
  통과한 제스처(스크롤)가 이기면 캔버스는 `bindtouchcancel`을 받는다 — 다른
  엘리먼트와 같은 경쟁 규칙이다.
- **Lynx 이벤트까지 뒤로 보내야** 하면 (캔버스가 아무 입력도 안 받아야 하면)
  Lynx 전역 속성을 함께 쓴다 — Lynx 레벨은 `user-interaction-enabled={false}`,
  LynxView 밖 네이티브까지 뚫으려면 `event-through`. 이 둘은 Lynx 기본 제공이라
  이 패키지와 무관하게 모든 엘리먼트에서 동작한다.

## 7. 스레딩 계약

| 일 | 스레드 |
|---|---|
| `NativeModules.WebGPU.execute()` (커맨드 해석 + Metal 인코딩) | Lynx JS 백그라운드 스레드 |
| `CAMetalLayer` 프로퍼티 설정, 드로어블 크기 갱신 | 메인 (`configure`는 비동기로 넘어간다) |
| `CADisplayLink` 프레임 이벤트 | 메인 (이벤트 전송만) |
| `readBuffer` GPU 완료 대기 | 전용 readback 큐 |

호스트 코드에서 `LynxWebGPUHost`의 메서드는 어느 스레드에서 불러도 안전하다.

## 8. 확인

시뮬레이터에서 번들을 띄우고 다음을 본다:

1. `<webgpu-canvas>` 자리에 클리어 색이 칠해지는가 → 표면 등록·configure까지 정상
2. `bindcanvasresize`의 width/height가 화면 배율만큼 커져 있는가 → 드로어블 크기 정상
3. Xcode 콘솔의 `org.lynxwebgpu` 서브시스템 로그 (`Canvas`/`Device` 카테고리)
4. JS 콘솔에 `[WebGPU:validation] …` 이 뜨는가 → 커맨드 스트림 오류 (경로가 함께 나온다)

네이티브 쪽 렌더 파이프라인 자체는 **호스트 앱 없이** 오프스크린 하네스로 검증한다 (`docs/TESTING.md`).
