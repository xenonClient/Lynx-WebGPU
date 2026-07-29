import ProjectDescription

// 라이브러리 자체는 SPM(`swift build` / `swift test`)만으로 완결된다.
// Tuist는 **데모 호스트 앱** 하나를 만들기 위해서만 쓴다 — 실기기·시뮬레이터에서
// Lynx 런타임과 함께 눈으로 확인하는 용도다 (자동 검증은 오프스크린 하네스가 담당).
let tuist = Tuist()
