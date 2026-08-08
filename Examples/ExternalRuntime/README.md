# ExternalRuntime — 외부 백엔드 주입 검증 픽스처

이 패키지는 **저장소 밖에서 만든 GPU 백엔드가 보는 시야**를 흉내 낸다:
`LynxWebGPUCore`(계약)와 `LynxWebGPUConformance`(증명 수단)만 링크하고,
Metal 엔진(`LynxWebGPU`)은 보지 않는다.

```zsh
swift build --package-path Examples/ExternalRuntime                      # 컴파일 검증
swift run --package-path Examples/ExternalRuntime external-runtime-check # 적합성 판정 검증
```

- 여기서 빌드가 깨지면 — product 구성이나 접근 수준이 외부 구현을 막고 있다는 뜻이다.
  Core의 public 표면을 바꾼 뒤에는 이 빌드를 돌려 볼 것 (`docs/TESTING.md`).
- `StubRuntime`은 GPU 없이 와이어 정책(디스패치 `WGPUCommand` · 오류 스코프
  `WGPUErrorScopeStack` · 응답 `WGPUBatchResult`)을 Core 공용 타입으로 조립하는 예시다 —
  Dawn 런타임을 시작할 때 이 파일을 출발점으로 삼으면 된다
  (`docs/extra/DAWN-BACKEND-REVIEW.md` §3-6).
- 스텁이므로 GPU 검사는 실패가 정상이다. 실행 파일의 성공 기준은 통과율이 아니라
  **전 검사가 크래시 없이 판정되는 것**이다.
