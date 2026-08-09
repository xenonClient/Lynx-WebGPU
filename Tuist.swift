import ProjectDescription

// The library itself is complete through SPM alone (`swift build` / `swift test`).
// Tuist exists only to build the **demo host app** — for checking things by eye alongside the Lynx
// runtime on a device or simulator (automated verification is the offscreen harness's job).
let tuist = Tuist()
