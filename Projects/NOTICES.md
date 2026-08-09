# 오픈소스 고지 (데모·검증 앱 배포용)

이 저장소의 **앱 산출물**(`WebGPUDemo` · `DawnDemo` · `DawnCheck`)에 포함되는 서드파티
구성요소의 고지다. 앱을 배포한다면 이 파일의 내용을 그대로 앱의 오픈소스 고지 화면이나
동봉 문서에 실으면 된다.

**SPM 라이브러리 product(`LynxWebGPU` · `LynxWebGPUCore` · `LynxWebGPUConformance`)만 쓰는
앱에는 해당 사항이 없다** — 이 패키지의 외부 의존성은 0이다.

| 구성요소 | 라이선스 | 들어가는 산출물 |
|---|---|---|
| Dawn | BSD-3-Clause | DawnDemo · DawnCheck |
| Lynx | Apache-2.0 | WebGPUDemo · DawnDemo |
| three.js 0.185.1 | MIT | 데모 `.lynx.bundle` (WebGPUDemo · DawnDemo) |

---

## Dawn

[dawn.googlesource.com/dawn](https://dawn.googlesource.com/dawn) — 프리빌트
`Dawn.xcframework`로 링크한다. iOS 바이너리에 들어가는 부분은 아래 BSD-3-Clause 절이다
(업스트림 LICENSE에서 Apache-2.0으로 표시된 두 경로 `generator/templates/art/*`,
`tools/android/webgpu/*`는 generator 템플릿과 Android 도구라 이 산출물에 포함되지 않는다).

```
Copyright 2017-2026 The Dawn & Tint Authors

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## Lynx

[lynx-family/lynx](https://github.com/lynx-family/lynx) — 앱 타깃이 XCFramework로 링크한다.
Apache License 2.0으로 배포되며, 전문은 <http://www.apache.org/licenses/LICENSE-2.0>에 있다.
Apache-2.0 §4(d)에 따라 업스트림 NOTICE의 귀속 고지를 그대로 싣는다:

```
Lynx Project
Copyright (c) 2018-2024 ByteDance Inc.
Copyright (c) 2024 TikTok Inc.
All rights reserved.
```

---

## three.js

[mrdoob/three.js](https://github.com/mrdoob/three.js) 0.185.1 — 데모 번들(`three`·`threelab`
씬)이 import하므로 `.lynx.bundle` 안에 함께 실린다.

```
The MIT License

Copyright © 2010-2026 three.js authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
