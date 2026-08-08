import ProjectDescription

// DawnCheck 전용 Tuist 루트 — 데모 워크스페이스와 **분리**한다.
// Dawn XCFramework(바이너리, GitHub Release) 해석을 데모 개발 루프에 끌어들이지 않기 위해서다.
// 생성: mise exec -- tuist generate --path Projects/DawnCheck --no-open
let tuist = Tuist()
