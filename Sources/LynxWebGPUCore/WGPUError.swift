import Foundation

/// WebGPU 호출이 실패한 이유.
///
/// WebGPU는 **안전한 API**다 — 잘못된 인자로 프로세스를 죽이지 않고 오류를 돌려준다.
/// 이 라이브러리도 같은 계약을 지킨다: 커맨드 해석 중 발생하는 모든 실패는
/// `WGPUError`로 수집되어 JS에 배열로 돌아가며, 네이티브에서 트랩하지 않는다.
public struct WGPUError: Error, Equatable, CustomStringConvertible {
    /// WebGPU `GPUError` 계열에 대응하는 분류.
    public enum Kind: String, Sendable {
        /// 잘못된 인자·상태 (WebGPU `GPUValidationError`). 대부분 호출 측 버그다.
        case validation
        /// 리소스 생성 실패 (WebGPU `GPUOutOfMemoryError`).
        case outOfMemory = "out-of-memory"
        /// 이 구현이 아직 지원하지 않는 WebGPU 기능. 명세상 유효한 요청이다.
        case unsupported
        /// 백엔드(Metal/셰이더 컴파일) 내부 오류.
        case backend
    }

    public let kind: Kind
    public let message: String
    /// 커맨드 스트림 상 위치 (`commands[3].vertex.buffers[0].format`). 디버깅 단서.
    public let path: String?

    public init(kind: Kind, message: String, path: String? = nil) {
        self.kind = kind
        self.message = message
        self.path = path
    }

    public var description: String {
        guard let path, !path.isEmpty else { return "[\(kind.rawValue)] \(message)" }
        return "[\(kind.rawValue)] \(path): \(message)"
    }

    /// JS로 되돌릴 직렬화 형태.
    public var payload: [String: Any] {
        var result: [String: Any] = ["kind": kind.rawValue, "message": message]
        if let path, !path.isEmpty { result["path"] = path }
        return result
    }

    public static func validation(_ message: String, path: String? = nil) -> WGPUError {
        WGPUError(kind: .validation, message: message, path: path)
    }

    public static func unsupported(_ message: String, path: String? = nil) -> WGPUError {
        WGPUError(kind: .unsupported, message: message, path: path)
    }

    public static func backend(_ message: String, path: String? = nil) -> WGPUError {
        WGPUError(kind: .backend, message: message, path: path)
    }

    public static func outOfMemory(_ message: String, path: String? = nil) -> WGPUError {
        WGPUError(kind: .outOfMemory, message: message, path: path)
    }
}

/// `device.pushErrorScope(filter)`가 받는 필터 (`GPUErrorFilter`).
///
/// 명세는 세 가지만 정의하는데 이 구현의 오류 분류는 넷이다. 남는 둘을 어디에 붙일지는
/// **JS 작성자가 브라우저에서 보게 될 것**을 기준으로 정한다:
///
/// - `unsupported`는 "명세상 유효하지만 이 구현이 아직 안 하는 것"이다. 브라우저에서
///   같은 코드를 돌리면 그 호출은 유효하므로 오류가 안 나거나 validation으로 난다.
///   그래서 `validation` 스코프가 잡는다 — 못 잡으면 대응할 방법이 없다.
/// - `backend`(Metal/셰이더 컴파일 실패)는 호출이 유효한데 시스템 쪽 사정으로 실패한
///   경우이므로 명세의 `internal`에 해당한다.
public enum WGPUErrorFilter: String, CaseIterable, Sendable {
    case validation
    case outOfMemory = "out-of-memory"
    case `internal`

    public func captures(_ kind: WGPUError.Kind) -> Bool {
        switch self {
        case .validation: return kind == .validation || kind == .unsupported
        case .outOfMemory: return kind == .outOfMemory
        case .internal: return kind == .backend
        }
    }
}
