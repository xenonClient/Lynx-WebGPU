# Lynx 연동 가이드

호스트 앱에 이 라이브러리를 붙이는 절차.

## 1. 패키지 추가

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/<org>/Lynx-WebGPU", from: "1.0.0"),
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

## 5. 스레딩 계약

| 일 | 스레드 |
|---|---|
| `NativeModules.WebGPU.execute()` (커맨드 해석 + Metal 인코딩) | Lynx JS 백그라운드 스레드 |
| `CAMetalLayer` 프로퍼티 설정, 드로어블 크기 갱신 | 메인 (`configure`는 비동기로 넘어간다) |
| `CADisplayLink` 프레임 이벤트 | 메인 (이벤트 전송만) |
| `readBuffer` GPU 완료 대기 | 전용 readback 큐 |

호스트 코드에서 `LynxWebGPUHost`의 메서드는 어느 스레드에서 불러도 안전하다.

## 6. 확인

시뮬레이터에서 번들을 띄우고 다음을 본다:

1. `<webgpu-canvas>` 자리에 클리어 색이 칠해지는가 → 표면 등록·configure까지 정상
2. `bindcanvasresize`의 width/height가 화면 배율만큼 커져 있는가 → 드로어블 크기 정상
3. Xcode 콘솔의 `org.lynxwebgpu` 서브시스템 로그 (`Canvas`/`Device` 카테고리)
4. JS 콘솔에 `[WebGPU:validation] …` 이 뜨는가 → 커맨드 스트림 오류 (경로가 함께 나온다)

네이티브 쪽 렌더 파이프라인 자체는 **호스트 앱 없이** 오프스크린 하네스로 검증한다 (`docs/TESTING.md`).
