import Foundation
import CoreGraphics
import QuartzCore

/// `createImageBitmap()` 디코딩 옵션 (명세 `ImageBitmapOptions`의 부분집합).
///
/// 값만 담는 타입이라 백엔드와 무관하다 — 그래서 디코더(`WGPUImageDecoder`, ImageIO)와 달리
/// 여기 있다. `WGPUImageDecoder.Options`는 이것의 별칭이다.
public struct WGPUImageDecodeOptions {
    /// 위아래를 뒤집는다. 웹 이미지의 원점은 좌상단이고 텍스처 좌표계에 따라
    /// 뒤집어야 할 때가 있다 (`copyExternalImageToTexture`의 `flipY`).
    public var flipY: Bool
    /// 알파를 색에 곱해 둘 것인가. 명세 `premultiplyAlpha`의 `'premultiply'`/`'none'`.
    /// `'default'`는 소스가 정하는데, ImageIO 경로에서는 곱하지 않은 값을 기준으로 삼는다.
    public var premultiplyAlpha: Bool
    /// 목표 크기 (`resizeWidth`/`resizeHeight`). nil이면 원본 크기다.
    public var resize: (width: Int, height: Int)?

    public init(flipY: Bool = false, premultiplyAlpha: Bool = false,
                resize: (width: Int, height: Int)? = nil) {
        self.flipY = flipY
        self.premultiplyAlpha = premultiplyAlpha
        self.resize = resize
    }
}

/// 커맨드 스트림의 **반대편** — 명령을 받아 GPU를 모는 쪽.
///
/// ## 왜 프로토콜인가
///
/// 커맨드 스트림(`docs/COMMAND-STREAM.md`)은 순수 데이터다. `{op: …}` 배열이 들어가고
/// `{ok, errors, canvases, errorScopes}`가 나온다 — 그 사이에서 무엇이 도는지는 계약에 없다.
/// 그래서 구현을 갈아끼울 수 있고, 이 프로토콜이 그 자리다.
///
/// 이 패키지가 기본으로 주는 구현은 `LynxWebGPU`의 `LynxWebGPUContext`(Metal 직접)다.
/// 다른 구현(예: [Dawn](https://github.com/google/dawn) 위에 얹은 런타임)은 **이 패키지 밖에서**
/// 만들어 넣는다 — Lynx SDK를 여기서 가져오지 않는 것과 같은 이유다 (`docs/LYNX-INTEGRATION.md` §1):
/// 버전·배포처·바이너리 크기를 **앱이 정하게** 한다.
///
/// ```swift
/// let host = LynxWebGPUHost(runtime: try LynxWebGPUContext())   // 기본 (Metal)
/// let host = LynxWebGPUHost(runtime: try DawnWebGPURuntime())   // 앱이 넣는 구현
/// ```
///
/// 브리지(`LynxWebGPUBridge`)는 이 프로토콜만 본다. **JS 번들은 어느 쪽이든 손대지 않는다** —
/// 그 성질을 지키는 것이 이 경계의 존재 이유다.
///
/// ## 스레딩
///
/// `execute`는 Lynx의 **JS 백그라운드 스레드**에서 불린다. 구현이 직렬화 책임을 진다.
/// 캔버스 붙이기·크기 갱신은 **메인 스레드**에서 온다 (UI 레이아웃). 두 스레드가 만나므로
/// 구현은 표면 등록부를 잠가야 한다.
public protocol WebGPURuntime: AnyObject {

    // MARK: - 커맨드 스트림

    /// 한 배치를 실행한다.
    ///
    /// - Parameter payload: `{"commands": [{op: …}, …], "present": Bool}`.
    ///   `present`가 false면 프레임 **중간**의 내부 제출이다 — 커밋은 하되 드로어블 present와
    ///   프레임 스코프 핸들 만료는 뒤따라올 진짜 프레임 제출로 미룬다.
    /// - Returns: `{"ok": Bool, "errors": [...], "canvases": {...}, "errorScopes": [...]}`.
    ///   JS로 그대로 돌아간다 — 모양은 `docs/COMMAND-STREAM.md` §2가 정한다.
    func execute(_ payload: [String: Any]) -> [String: Any]

    // MARK: - 조회

    /// `navigator.gpu.requestAdapter()`가 쓰는 어댑터 정보·한계값·기능.
    /// 키는 **명세 철자 그대로**여야 한다 — 웹 라이브러리가 그 이름으로 읽는다.
    func adapterInfo() -> [String: Any]

    /// `GPUShaderModule.getCompilationInfo()`.
    func shaderCompilationInfo(handle: Int) -> [String: Any]

    /// 캔버스의 현재 픽셀 크기와 포맷 (`{ok, width, height, format}`).
    func canvasInfo(identifier: String) -> [String: Any]

    // MARK: - 비동기

    /// `GPUBuffer.mapAsync` + `getMappedRange` — 앞선 GPU 작업이 끝난 뒤 내용을 돌려준다.
    func readBuffer(handle: Int, offset: Int, size: Int?, completion: @escaping ([String: Any]) -> Void)

    /// `createImageBitmap()` — 인코딩된 이미지를 풀어 **JS가 발급한 핸들**에 등록한다.
    ///
    /// - Parameter data: 이미지 바이트. nil이면 `name`을 `provider`로 해석한다.
    func decodeImage(
        handle: Int,
        data: Data?,
        name: String?,
        options: WGPUImageDecodeOptions,
        provider: WGPUAssetProvider?,
        completion: @escaping ([String: Any]) -> Void
    )

    // MARK: - 캔버스

    /// 화면 표면을 붙인다. JS는 `configure({canvas: identifier})`로 이것을 지목한다.
    ///
    /// **레이어 타입이 `CAMetalLayer`인 것은 두 백엔드의 공통분모다** — Dawn도 Apple 플랫폼에서는
    /// `WGPUSurfaceSourceMetalLayer`로 같은 레이어를 받는다. 그래서 `<webgpu-canvas>` 엘리먼트는
    /// 어느 런타임을 쓰든 코드가 같다.
    func attachCanvas(identifier: String, layer: CAMetalLayer)

    /// 화면 없이 그리는 표면을 붙인다.
    ///
    /// 적합성 테스트가 **렌더 결과를 픽셀로 읽는 통로**다 (`readCanvasPixels`와 짝).
    /// 화면이 없는 환경에서 두 런타임이 같은 그림을 그리는지 기계로 확인할 수 있어야 한다.
    func attachOffscreenCanvas(identifier: String, size: CGSize) throws

    /// 드로어블 크기(픽셀)를 갱신한다. **메인 스레드**에서 레이아웃이 바뀔 때 불린다.
    func resizeCanvas(identifier: String, drawableSize: CGSize)

    func detachCanvas(identifier: String)

    /// 오프스크린 표면의 픽셀을 **설정된 포맷 그대로** 읽는다. 부르기 전에 GPU 작업이 끝나 있어야 한다.
    func readCanvasPixels(identifier: String) throws -> WGPUPixelReadback

    // MARK: - 프레임

    /// 등록된 모든 표면이 새 프레임을 받을 수 있는가.
    ///
    /// 프레임 티커가 이 값을 보고 포화 시 틱을 건너뛴다 — GPU가 밀려 있을 때 JS가 프레임을
    /// 만들면 드로어블 획득에서 **JS 스레드 전체**가 서기 때문이다.
    var isReadyForNextFrame: Bool { get }

    // MARK: - 수명

    /// 모든 GPU 객체를 버린다 (페이지 이탈·핫 리로드).
    func reset()
}

public extension WebGPURuntime {
    /// 편의 오버로드 — 버퍼 전체를 읽는다.
    func readBuffer(handle: Int, completion: @escaping ([String: Any]) -> Void) {
        readBuffer(handle: handle, offset: 0, size: nil, completion: completion)
    }

    /// 편의 오버로드 — 커맨드 배열을 그대로 넘긴다 (테스트 하네스·네이티브 단독 사용).
    @discardableResult
    func execute(commands: [[String: Any]], present: Bool = true) -> [String: Any] {
        execute(["commands": commands, "present": present])
    }
}
