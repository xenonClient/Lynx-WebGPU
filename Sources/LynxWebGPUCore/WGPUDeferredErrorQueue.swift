import Foundation

/// GPU 비동기 실패의 지연 보고 큐 — **완료 핸들러(임의 스레드)가 채우고 다음 배치가 비운다.**
///
/// 배치의 오류 수집이 도는 시점은 커밋 전이라, GPU 측 실패(메모리 부족·타임아웃·디바이스 제거)는
/// 구조상 그때 잡을 수 없다. 그래서 완료 핸들러가 여기 모아 두었다가 **다음 배치 결과**에 실어
/// 보낸다 — 이것이 없으면 그 실패들이 어디에도 나타나지 않고 무성으로 남는다.
///
/// 이 지연 전달 모델 자체가 와이어 계약이고 (`docs/COMMAND-STREAM.md` §2) 어느 백엔드든 같은
/// 통로가 필요해서 여기(Core) 있다 — Dawn이라면 uncaptured error 콜백이 채우는 자리다.
public final class WGPUDeferredErrorQueue {
    private let lock = NSLock()
    private var pending: [WGPUError] = []

    public init() {}

    /// 실패를 쌓는다. 임의 스레드에서 안전하다.
    public func report(_ error: WGPUError) {
        lock.lock()
        pending.append(error)
        lock.unlock()
    }

    /// 모아 둔 실패를 꺼내 비운다 (꺼낸 것은 이번 배치의 오류 수집으로 흘려보낸다).
    public func drain() -> [WGPUError] {
        lock.lock()
        defer { lock.unlock() }
        let failures = pending
        pending.removeAll()
        return failures
    }
}
