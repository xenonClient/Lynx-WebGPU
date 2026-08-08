import Foundation

/// `popErrorScope` 결과의 세 가지 상태. JS는 이것을 보고 Promise를 resolve할지 reject할지 정한다.
public enum WGPUPoppedErrorScope: Equatable {
    /// 스코프는 있었고 잡힌 오류는 없었다 → `null`로 resolve.
    case clean
    /// 스코프가 오류를 잡았다 → 그 오류로 resolve.
    case captured(WGPUError)
    /// `push`와 짝이 맞지 않는다 → 명세대로 `OperationError`로 **reject**한다.
    /// 이 실패는 GPUError가 아니므로 오류를 생성하지 않는다 — 명세에 없는 GPUError가
    /// 앱의 전역 핸들러·텔레메트리에 섞이지 않게.
    case unmatched

    /// JS로 되돌릴 직렬화 형태 (배치 결과 `errorScopes` 배열의 원소).
    public var payload: Any {
        switch self {
        case .clean: return NSNull()
        case .captured(let error): return error.payload
        case .unmatched: return ["rejected": true]
        }
    }
}

/// 열려 있는 오류 스코프 스택 (`GPUDevice.pushErrorScope`/`popErrorScope`).
///
/// **백엔드와 무관한 와이어 정책**이라 여기(Core) 있다 — 어느 런타임이든 같은 규칙으로 잡고
/// 같은 모양으로 돌려줘야 적합성 검사(`error-scope-capture`)를 통과한다. 규칙은 전부 명세에서 온다:
///
/// - **배치 사이에도 살아 있다** — WebGPU에서 오류 스코프는 디바이스 상태이고, `push`와 `pop`
///   사이에 `submit`이 몇 번이든 들어갈 수 있다. 런타임은 이 스택을 배치 수명 밖에 둘 것.
/// - 오류는 **가장 안쪽의 맞는 스코프**가 잡는다 (`WGPUErrorFilter.captures`).
/// - 스코프가 돌려주는 것은 **처음 잡힌 오류 하나**다.
/// - 잡힌 오류는 배치 결과의 `errors`에 실리지 않는다 — 그래야 JS의 전역 핸들러
///   (`device.onError`)가 "내가 이미 처리하기로 한 오류"를 다시 보고하지 않는다.
public struct WGPUErrorScopeStack {
    private var scopes: [(filter: WGPUErrorFilter?, captured: WGPUError?)] = []

    public init() {}

    /// 열려 있는 스코프 수 (안쪽이 뒤).
    public var depth: Int { scopes.count }

    /// 스코프를 연다.
    ///
    /// `filter`가 nil이면 **아무것도 잡지 않는 자리표시자**다 — 필터 파싱이 실패했을 때도
    /// 스택 깊이는 맞춰야 해서 있다. 안 쌓으면 이후 pop이 바깥 스코프를 가져가서, 앱이
    /// 안쪽 구간의 결과라고 믿는 값이 실제로는 바깥 구간의 결과가 된다.
    public mutating func push(_ filter: WGPUErrorFilter?) {
        scopes.append((filter, nil))
    }

    /// 가장 안쪽 스코프를 닫고 결과를 돌려준다. 빈 스택이면 `.unmatched`다.
    public mutating func pop() -> WGPUPoppedErrorScope {
        guard let scope = scopes.popLast() else { return .unmatched }
        return scope.captured.map(WGPUPoppedErrorScope.captured) ?? .clean
    }

    /// 오류를 가장 안쪽의 맞는 스코프에 넣는다.
    ///
    /// 잡았으면 true — 호출자는 그 오류를 배치 `errors`로 내보내지 않는다.
    /// 못 잡았으면 false — 호출자가 배치 결과로 내보낸다.
    public mutating func capture(_ error: WGPUError) -> Bool {
        for index in scopes.indices.reversed()
        where scopes[index].filter?.captures(error.kind) == true {
            if scopes[index].captured == nil { scopes[index].captured = error }
            return true
        }
        return false
    }

    /// 디바이스를 버릴 때 (`GPUDevice.destroy`) 열려 있던 스코프도 함께 버린다 —
    /// 남겨 두면 다음 페이지의 오류가 죽은 스코프에 잡혀 아무 데도 보고되지 않는다.
    public mutating func discardAll() {
        scopes.removeAll()
    }
}
