import Foundation

/// 배치 결과 `canvases`에 싣는 캔버스 보고 — 이번 배치가 드로어블을 내준 표면의 픽셀 크기.
public struct WGPUCanvasReport: Equatable {
    public let width: Int
    public let height: Int

    public init(width: Int, height: Int) {
        self.width = width
        self.height = height
    }
}

/// `execute` 한 배치의 응답 조립기 — `{ok, commandCount, objects, errors, canvases, errorScopes}`.
///
/// 키 철자와 생략 규칙이 **와이어 계약**이다 (`docs/COMMAND-STREAM.md` §2). 백엔드마다 이 모양을
/// 손으로 다시 만들면 철자 하나로 어긋나므로, 어느 런타임이든 이 타입으로 조립한다 —
/// 적합성 검사(`error-accumulation`)가 판정하는 것이 정확히 이 모양이다.
public struct WGPUBatchResult {
    public var commandCount: Int
    /// live 네이티브 객체 수 — JS가 destroy 누락(레지스트리 증식)을 감시할 수 있게.
    public var liveObjectCount: Int
    /// 스코프에 잡히지 않은 오류들 (`WGPUErrorScopeStack.capture`가 false를 돌려준 것).
    public var errors: [WGPUError]
    public var canvases: [String: WGPUCanvasReport]
    /// 이번 배치에서 pop된 스코프의 결과 (pop 순서 — JS의 Promise 순서와 1:1로 맞춘다).
    public var poppedScopes: [WGPUPoppedErrorScope]

    public init(
        commandCount: Int,
        liveObjectCount: Int,
        errors: [WGPUError] = [],
        canvases: [String: WGPUCanvasReport] = [:],
        poppedScopes: [WGPUPoppedErrorScope] = []
    ) {
        self.commandCount = commandCount
        self.liveObjectCount = liveObjectCount
        self.errors = errors
        self.canvases = canvases
        self.poppedScopes = poppedScopes
    }

    /// JS로 돌아가는 형태.
    ///
    /// `ok`/`commandCount`/`objects`는 항상 싣고, `errors`/`canvases`/`errorScopes`는
    /// **비지 않을 때만** 싣는다 — 키 생략도 계약의 일부다.
    public var payload: [String: Any] {
        var result: [String: Any] = [
            "ok": errors.isEmpty,
            "commandCount": commandCount,
            "objects": liveObjectCount,
        ]
        if !errors.isEmpty {
            result["errors"] = errors.map(\.payload)
        }
        if !canvases.isEmpty {
            result["canvases"] = canvases.mapValues { report -> [String: Any] in
                ["width": report.width, "height": report.height]
            }
        }
        if !poppedScopes.isEmpty {
            result["errorScopes"] = poppedScopes.map(\.payload)
        }
        return result
    }

    /// 배치에 들어가 보지도 못한 최상위 실패 — 페이로드에 `commands` 배열이 없을 때 등.
    /// 명령을 하나도 안 봤으므로 `commandCount` 같은 배치 결과 키는 싣지 않는다.
    public static func failure(_ errors: [WGPUError]) -> [String: Any] {
        ["ok": false, "errors": errors.map(\.payload)]
    }
}
