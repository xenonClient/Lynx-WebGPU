# react-native-webgpu 대조 — Dawn을 최대한 활용하는 검증 레이어 분할

[wcandillon/react-native-webgpu](https://github.com/wcandillon/react-native-webgpu)를 확인하고
(2026-08-09), 그 분할을 이 저장소의 제약에 맞게 옮긴 기록이다.

> **고지.** react-native-webgpu는 MIT (`Copyright 2024-present, William Candillon`)다.
> 여기서 옮긴 것은 **레이어를 어디서 자를지에 대한 판단**이고, 코드는 이식하지 않았다 —
> `DawnBackend.swift`는 Dawn 공개 헤더(`webgpu.h`)와 이 저장소의 `WGPUBackend`만 보고 썼다. 결론부터: **위층(공유)은
브리징과 최소한의 예외처리만, 아래층은 두 갈래** — 렌더링 파이프라인·WGSL→MSL 변환·명세
검증을 직접 구현한 Metal 쪽과, 그 전부를 Dawn 자신이 처리하는 Dawn 쪽이다. 스위치는
`WGPUBackendCapabilities.validatesNatively`.

## 1. react-native-webgpu가 실제로 하는 것 (확인 사실)

- **JSI 직접 바인딩.** `packages/webgpu/cpp/rnwgpu/api/`에 `GPUDevice`·`GPUBuffer`·
  `GPUQueue`… WebGPU 객체당 어댑터 클래스 ~80-90개 + `Convertors.h` + `descriptors/`.
  메서드 본문은 **변환하고 Dawn을 부르는 것이 전부**다 — 명세 검증이 이 층에 없다.
- **검증·WGSL 컴파일은 전부 Dawn.** 오류는 Dawn의 uncaptured error 콜백으로 JS에 전달된다.
  기능 목록조차 `GPUFeatures.h`의 단일 목록에서 양방향 생성하고, 테스트가 설치된
  `webgpu_cpp.h`와 대조한다 — 명세 지식의 원본을 Dawn 헤더에 둔다.
- **Dawn은 서브모듈 + 프리빌트.** `externals/dawn` 핀 + `yarn install-dawn`이 릴리스
  태그(`dawn-chrome-m152` 류)에서 `.xcframework`/`.so`를 받는다.
- **표면은 우리와 같다.** `apple/`의 `WebGPUView`·`MetalView`(CAMetalLayer).
  비동기는 `async/`(`AsyncTaskHandle`·`RuntimeContext`)가 이벤트 루프에 물린다.

우리와의 근본 차이는 **JS 경계**뿐이다: RN은 JSI로 동기 호스트 객체를 심을 수 있고, Lynx는
NativeModule 딕셔너리 브리지라 프레임당 1왕복 커맨드 스트림이 그 자리를 대신한다
(`docs/ARCHITECTURE.md` §3). 즉 우리의 "JSI 어댑터층" = 커맨드 스트림 디코딩 + 백엔드 동사.

## 2. 옮긴 분할

```
                     ┌──────────────────────────────────────────┐
 공유 (위)            │ LynxWebGPUBridge — 레이어·이벤트만        │
                     │ WGPUBackendEngine(Core) —                │
                     │   디코딩 · 핸들 조회 · 와이어 정책(스코프·  │
                     │   배치 결과·지연 오류) · 프레임 수명 ·      │
                     │   직렬화 락 · 자가 펌프 ·                  │
                     │   **최소한의 예외처리**: 매핑 게이트,       │
                     │   패스 상태 가드, CPU 경로 보호(슬라이싱    │
                     │   경계·Range 구성), 트랩 방지 구조 가드     │
                     └───────────────┬──────────────────────────┘
                 validatesNatively == false          == true
                     ┌───────────────┴──────┐   ┌───────────────┐
 백엔드 (아래)        │ 직접 구현 (Metal)     │   │ Dawn          │
                     │ · 엔진의 명세 검사     │   │ · 명세 검증    │
                     │   전체가 이쪽 몫으로   │   │   = Dawn 검증기│
                     │   돈다               │   │ · WGSL = Tint  │
                     │ · WGSL→MSL 트랜스파일 │   │ · 파이프라인·  │
                     │ · 파이프라인·인코딩    │   │   인코딩       │
                     │ · 스테이징 풀         │   │ · 동사는 변환+ │
                     │                      │   │   dawn* 안전   │
                     │                      │   │   변환만       │
                     └──────────────────────┘   └───────────────┘
```

- **엔진의 명세 검사는 개념적으로 직접 구현 쪽의 일부다.** 범위·정렬·usage·occlusion
  중첩·번들 호환성·압축 제약 검사는 Metal이 관대(또는 단언으로 사망)해서 존재한다.
  `validatesNatively == true`면 전부 빠지고, 같은 잘못은 Dawn 검증기가 디바이스 오류로
  거부한다 — react-native-webgpu가 아무 검증 없이 Dawn을 부르는 것과 같은 자리다.
- **위층에 남는 예외처리의 기준**은 "검증 주체와 무관하게 우리 프로세스를 지키는 것"이다:
  없는 핸들 조회(와이어 계약), `mapAsync` 매핑 게이트(와이어 상태 — Dawn은 알 수 없다),
  패스 상태 가드(동사가 조용한 no-op이 되는 것 방지), CPU 슬라이싱 경계
  (`copyExternalImageToTexture`의 memcpy), Range 구성·음수 같은 트랩 방지.
- **Dawn 동사의 유일한 방어는 `dawnU32` 계열 안전 변환**이다 — JS 유래 정수가 GPU 인자
  폭을 벗어나면 트랩 대신 validation으로 거부한다 (`DawnHardeningTests`가 이 층을 판정).
- **배치 진단은 uncaptured 콜백으로 모은다.** 디바이스 오류 스코프는 첫 오류 하나만
  돌려줘서 한 배치의 다중 거부가 뭉개진다 — uncaptured는 오류마다 개별 전달이라
  브라우저의 오류 밀도와 같다 (`DawnBackend.collectBatchDiagnostics`).

## 3. 알려진 와이어 차이 (설계상 감수)

명세 검증의 주체가 바뀌면 **오류의 문구·시점**이 달라진다. 계약(`docs/COMMAND-STREAM.md`)은
"오류가 온다"까지만 정하므로 위반은 아니지만, Metal 런타임의 문구·시점을 탐침하는 코드는
갈린다 — `contracts` 씬의 3건이 그 표본이다 (`docs/TESTING.md` §2-1):

| 갈리는 것 | Metal (엔진 검사) | Dawn (네이티브) |
|---|---|---|
| 오류 문구 | 한국어, `commands[i].필드` 경로 | Dawn 영문, 경로 없음 |
| 번들 내부 검증 시점 | 실행(replay) 시점 | **명세대로** `finish()` 시점 |
| 한 배치의 다중 거부 | 명령마다 동기 오류 | Dawn이 내는 만큼 (비동기 회수) |
| 무효 객체의 여파 | 명령 단위로 국소화 | 명세대로 연쇄 ("invalid due to previous error") |

## 4. 더 가져올 수 있는 것 (미착수)

- **dawn.json 코드젠.** RN-webgpu의 `GPUFeatures.h` 단일 목록 모델처럼, Dawn 소스의
  `dawn.json`에서 `DawnEnum` 매핑·동사 변환을 생성하면 열거형 추가를 손으로 쫓지 않는다.
- **프리빌트 배포 모델.** `externals/dawn` 핀 + 릴리스 태그 다운로드는
  [Dawn-xcFramework](https://github.com/xenonClient/Dawn-xcFramework) 저장소가 이미 같은
  모양이다 — 별도 저장소 분리(검토 문서 §3-6) 때 그대로 쓴다.
