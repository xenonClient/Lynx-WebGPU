import Foundation
import os

/// 라이브러리 공통 로거.
///
/// 프레임 루프 안에서는 로그를 남기지 않는다 — 60fps에서 os_log 한 줄도 누적되면 무시할 수 없다.
/// 생성/해제/오류처럼 빈도가 낮은 경로에만 쓴다.
public enum WGPULog {
    public static let subsystem = "org.lynxwebgpu"

    public static func logger(category: String) -> Logger {
        Logger(subsystem: subsystem, category: category)
    }

    public static let device = logger(category: "Device")
    public static let shader = logger(category: "Shader")
    public static let bridge = logger(category: "Bridge")
    public static let canvas = logger(category: "Canvas")
}
