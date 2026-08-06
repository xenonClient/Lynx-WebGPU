import Foundation

/// 배치 하나가 프레임 경계에서 무엇을 해야 하는가.
///
/// 명세는 현재 텍스처의 만료를 `submit()`이 아니라 **"updating the rendering"**, 즉 프레임의
/// 끝에 건다. 그래서 한 프레임이 배치 여러 개로 쪼개질 수 있고, 어느 배치가 프레임의 끝인지는
/// 클라이언트(shim)가 `present` 플래그로 알려 준다 (`docs/COMMAND-STREAM.md` §1).
///
/// 이 규칙은 **백엔드와 무관하다** — Metal이든 Dawn이든 같은 시점에 present해야 브라우저와
/// 같은 그림이 나온다. 그래서 값 하나로 뽑아 두고 양쪽이 같은 것을 본다.
public struct WGPUFrameBoundary: Equatable {
    /// 획득해 둔 드로어블을 화면에 내보내고 프레임 스코프 핸들을 만료시키는가.
    public let presents: Bool
    /// 명령 없이 **present만 하는** 배치인가 (프레임 루프 콜백의 끝).
    ///
    /// 이런 배치는 커맨드 버퍼가 없어도 드로어블을 내보내야 한다 — 그냥 지나가면 화면이
    /// 멈춘 채 아무 말이 없다. 조건을 "명령이 비었을 때"로 **좁힌 것이 중요하다**: 명령은
    /// 있는데 커맨드 버퍼가 안 생긴 배치(드로어블만 얻고 끝난 경우)까지 present하면
    /// 그리지도 않은 화면이 나간다.
    public let closesFrame: Bool

    /// - Parameter requestedPresent: `execute({present})`. false면 프레임 **중간**의 내부
    ///   제출이다 — shim의 `popErrorScope`·`mapAsync`가 결과를 받으려고 흘려보낸 배치.
    public init(requestedPresent: Bool, commandCount: Int) {
        presents = requestedPresent
        closesFrame = requestedPresent && commandCount == 0
    }
}

/// 캔버스별 in-flight 프레임 회계.
///
/// ## 왜 필요한가
///
/// GPU가 프레임을 소화하지 못하고 밀리면 스왑체인의 드로어블 풀이 고갈되고, 드로어블 획득이
/// **JS 스레드 전체를 최대 1초까지 세운다** — 캔버스뿐 아니라 그 페이지의 터치 핸들러·타이머·
/// 네트워크 콜백까지 함께 멈추는 최악의 백프레셔다.
///
/// 그래서 커밋된 프레임 수를 세어 두고, 한도에 닿은 캔버스가 있으면 **프레임 티커가 그 틱을
/// 통째로 건너뛴다.** JS는 깨어나지도 않으므로 블록될 일이 없고, GPU가 완료를 돌려주면
/// 다음 틱부터 자연히 재개된다. 화면에는 프레임 드랍으로 보인다 — 밀린 프레임을 기다렸다
/// 몰아서 그리는 것보다 낫다.
///
/// ## 왜 백엔드 밖인가
///
/// **Dawn이 이것을 대신해 주지 않는다.** `wgpuSurfacePresent`에는 "지금 present하면
/// 블록되는가"를 묻는 통로가 없어, 어느 백엔드를 쓰든 제출/완료를 직접 세어야 한다.
/// 그래서 GPU 타입을 하나도 모르는 이 자리에 둔다 — 새 런타임은 커밋할 때
/// `noteCommitted(canvas:)`, 완료 핸들러에서 `noteCompleted(canvas:)`만 부르면 된다.
///
/// ## 스레딩
///
/// `noteCommitted`는 JS 스레드, `noteCompleted`는 GPU 완료 핸들러(임의 스레드),
/// `isReadyForNextFrame`은 메인 스레드(프레임 티커)에서 온다. 전부 락으로 감싼다.
/// (`final`이 아닌 이유는 테스트가 통지를 세기 위해 상속하기 때문이다 — 외부 모듈에서
/// 상속할 수는 없다.)
public class WGPUFrameCoordinator {
    /// 동시에 GPU에 걸어 둘 프레임 수 상한. `CAMetalLayer`의 드로어블 풀 크기(기본 3)와 같다 —
    /// 이보다 밀리면 획득이 JS 스레드를 세우므로 그 전에 프레임을 거른다.
    public static let defaultMaxFramesInFlight = 3

    public let maxFramesInFlight: Int

    /// 페이싱 대상 캔버스별 in-flight 수. **여기 없는 캔버스는 항상 준비 상태다** —
    /// 오프스크린 표면처럼 드로어블 풀이 없는 것은 밀릴 일이 없다.
    private var inFlight: [String: Int] = [:]
    private let lock = NSLock()

    public init(maxFramesInFlight: Int = WGPUFrameCoordinator.defaultMaxFramesInFlight) {
        self.maxFramesInFlight = max(maxFramesInFlight, 1)
    }

    // MARK: - 등록

    /// 이 캔버스를 프레임 페이싱 대상으로 삼는다 (스왑체인이 있는 표면).
    public func track(canvas: String) {
        lock.lock()
        if inFlight[canvas] == nil { inFlight[canvas] = 0 }
        lock.unlock()
    }

    /// 캔버스가 사라졌다. **남겨 두면 죽은 캔버스의 카운터가 영원히 틱을 막는다.**
    public func forget(canvas: String) {
        lock.lock()
        inFlight.removeValue(forKey: canvas)
        lock.unlock()
    }

    public func reset() {
        lock.lock()
        inFlight.removeAll()
        lock.unlock()
    }

    // MARK: - 회계

    /// 이 캔버스의 드로어블을 실은 작업이 GPU에 제출됐다.
    ///
    /// 추적 대상이 아닌 캔버스면 아무 일도 하지 않는다 — 세어 봐야 쓰이지 않는다.
    public func noteCommitted(canvas: String) {
        lock.lock()
        if let count = inFlight[canvas] { inFlight[canvas] = count + 1 }
        lock.unlock()
    }

    /// 그 작업이 GPU에서 끝났다 (임의 스레드).
    public func noteCompleted(canvas: String) {
        lock.lock()
        if let count = inFlight[canvas] { inFlight[canvas] = max(count - 1, 0) }
        lock.unlock()
    }

    // MARK: - 조회

    public func framesInFlight(canvas: String) -> Int {
        lock.lock()
        defer { lock.unlock() }
        return inFlight[canvas] ?? 0
    }

    public func isReady(canvas: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard let count = inFlight[canvas] else { return true }
        return count < maxFramesInFlight
    }

    /// 추적 중인 **모든** 캔버스가 새 프레임을 받을 수 있는가.
    ///
    /// 하나라도 포화되면 틱을 통째로 건너뛴다. 캔버스마다 따로 걸러도 되지만, 프레임 이벤트는
    /// 페이지 전체에 하나뿐이라 어차피 나뉘지 않는다.
    public var isReadyForNextFrame: Bool {
        lock.lock()
        defer { lock.unlock() }
        return inFlight.values.allSatisfy { $0 < maxFramesInFlight }
    }

    /// 지금 추적 중인 캔버스들 (테스트·진단용).
    public var trackedCanvases: [String] {
        lock.lock()
        defer { lock.unlock() }
        return inFlight.keys.sorted()
    }
}
