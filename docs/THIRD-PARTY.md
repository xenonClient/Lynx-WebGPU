# 서드파티 구성요소와 라이선스

이 저장소가 **참고한 것 · 링크하는 것 · 배포물에 싣는 것**을 구분해 적는다. 셋은 의무가
다르다 — 참고에는 의무가 없고, 링크·번들에는 고지 의무가 따라온다.

확인 시점: 2026-08-09 (SPDX·저작권 줄은 각 저장소의 LICENSE 원문에서 그대로 옮겼다).

## 0. 이 저장소 자체

**LICENSE 파일이 없다.** 라이선스 선택은 저장소 소유자의 결정이라 여기서 고르지 않았다 —
공개 배포 전에 루트에 `LICENSE`를 두고 아래 §3의 고지 묶음을 함께 넣을 것.

## 1. 한눈에

| 구성요소 | 라이선스 | 우리가 소비하는 방식 | 배포물에 들어가나 |
|---|---|---|---|
| [react-native-webgpu](https://github.com/wcandillon/react-native-webgpu) | MIT | **참고만** — 코드 미이식 | 아니오 |
| [Dawn](https://dawn.googlesource.com/dawn) (google/dawn) | BSD-3-Clause (일부 Apache-2.0) | 프리빌트 `.xcframework` 링크 (DawnCheck·DawnDemo 타깃 **한정**) | DawnCheck·DawnDemo 앱 |
| [Lynx](https://github.com/lynx-family/lynx) SDK | Apache-2.0 | 앱 타깃이 XCFramework 링크 (**라이브러리는 의존하지 않는다**) | WebGPUDemo·DawnDemo 앱 |
| [three.js](https://github.com/mrdoob/three.js) 0.185.1 | MIT | 데모 번들이 import (`three`·`threelab` 씬) | 데모 `.lynx.bundle` |

**SPM product(`LynxWebGPU`·`LynxWebGPUCore`·`LynxWebGPUConformance`)만 쓰는 앱에는 위 어느
것도 들어가지 않는다** — 이 패키지의 외부 의존성은 0이다 (`README.md`「모듈」). 아래 의무는
전부 **데모/검증 앱을 배포할 때** 붙는 것이다.

## 2. 구성요소별

### react-native-webgpu — MIT · **참고만, 코드 미이식**

```
MIT License
Copyright 2024-present, William Candillon.
```

`docs/extra/RN-WEBGPU-LAYERING.md`가 이 프로젝트의 **레이어 분할을 대조**한 기록이다.
읽고 배운 것은 "얇은 어댑터가 변환만 하고 검증은 Dawn에 맡긴다"는 **구조적 판단**이고,
그 판단을 우리 제약(Lynx 딕셔너리 브리지 · 커맨드 스트림)에 맞춰 **처음부터 다시 썼다**:

- 소스·헤더·생성 코드·빌드 스크립트를 **한 줄도 옮기지 않았다.** `DawnBackend.swift`는
  Dawn의 공개 헤더(`webgpu.h`)와 이 저장소의 `WGPUBackend` 프로토콜만 보고 쓴 것이다.
- 우리에겐 JSI 계층 자체가 없다 (Lynx는 NativeModule 딕셔너리 경계다).

그래서 배포물에 rn-webgpu의 저작물이 없고 **MIT 고지 의무도 발생하지 않는다.** 그럼에도
출처를 문서에 남기는 것은 예의이자 설계 근거의 추적성 때문이다.

### Dawn — BSD-3-Clause (파일 일부는 Apache-2.0)

LICENSE 원문의 두 갈래 (`Files: *`가 BSD, generator 템플릿·Android 도구 등이 Apache-2.0):

```
// Copyright 2017-2026 The Dawn & Tint Authors
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
```

소비 방식: [Dawn-xcFramework](https://github.com/xenonClient/Dawn-xcFramework)가 배포하는
프리빌트 `Dawn.xcframework`를 SPM binaryTarget으로 링크한다. **`Projects/DawnCheck`의 두
타깃(DawnCheck·DawnDemo)에서만** 쓰고, 라이브러리 쪽(`Sources/`)은 Dawn을 모른다.

> **바이너리 재배포 의무 (BSD-3-Clause 2항).** 바이너리로 재배포할 때는 위 저작권 고지와
> 조건 목록, 면책 조항을 **문서 또는 부속 자료에 재현**해야 한다. DawnDemo 같은 앱을
> 외부에 배포한다면 앱의 오픈소스 고지 화면(또는 동봉 문서)에 Dawn 고지를 넣을 것.
> 3항은 "이름으로 보증·홍보 금지"라 별도 조치가 필요 없다.

> **확인 필요 (미해결).** `xenonClient/Dawn-xcFramework` 저장소에는 LICENSE·NOTICE가
> **선언돼 있지 않다** (GitHub 라이선스 API 404). 재배포되는 산출물이 Dawn(BSD-3-Clause)
> 바이너리이므로, 그 저장소에 업스트림 LICENSE 사본과 출처를 넣어 두는 것이 맞다 —
> 이 저장소의 결정 범위 밖이라 여기서는 사실만 적어 둔다.

### Lynx SDK — Apache-2.0

`lynx-family/lynx`가 Apache-2.0이다. 이 패키지는 **Lynx를 의존성으로 가져오지 않는다**
(버전·배포처를 앱이 정하게 하려고 — `docs/LYNX-INTEGRATION.md` §1). 데모·DawnDemo 앱 타깃만
[Lynx-XCFramework](https://github.com/xenonClient/Lynx-XCFramework)를 링크한다.

> 배포 시: Apache-2.0 §4는 라이선스 사본 첨부와 **NOTICE 파일 보존**을 요구한다.
> (이 XCFramework 저장소에도 라이선스 선언이 없다 — Dawn 쪽과 같은 확인 항목이다.)

### three.js 0.185.1 — MIT

```
The MIT License
Copyright © 2010-2026 three.js authors
```

`Projects/WebGPUDemo/DemoSrc`의 `three`·`threelab` 씬이 import하므로 **번들러가 `.lynx.bundle`
안에 함께 넣는다** — 즉 데모 앱 리소스로 재배포된다. MIT는 저작권 고지와 허가 고지를
사본에 포함할 것을 요구하므로, 데모를 배포한다면 고지 묶음에 three.js를 넣을 것.

## 3. 배포 형태별 고지 체크리스트

| 배포하는 것 | 넣어야 할 고지 |
|---|---|
| SPM 라이브러리만 (`LynxWebGPU`·`Core`·`Conformance`) | 없음 (외부 의존성 0) |
| WebGPUDemo 앱 | Lynx(Apache-2.0) · three.js(MIT) |
| DawnDemo / DawnCheck 앱 | 위 + **Dawn(BSD-3-Clause, 일부 Apache-2.0)** |

## 4. 상표·명세

- **WebGPU**·**WGSL**은 W3C 명세다. 이 구현은 명세의 이름과 열거형 철자를 그대로 쓰지만
  (`docs/WEBGPU-API.md`), W3C의 인증이나 보증을 받은 구현이 아니다.
- **Metal**은 Apple Inc.의 상표다. **Dawn**·**Tint**는 Google이 관리하는 프로젝트 이름이다.
  이름은 상호운용성을 설명하기 위해 쓰였을 뿐, 어떤 제휴나 보증도 뜻하지 않는다.
