#if canImport(Lynx)
import Foundation
import QuartzCore
import UIKit

/// `CADisplayLink` 기반 프레임 드라이버.
///
/// 항상 메인 스레드에서 돌며, 콜백에서는 전역 이벤트 하나만 보낸다 (GPU 작업은 JS 스레드에서 한다).
final class WebGPUFrameTicker {
    /// (타임스탬프 초, 직전 프레임과의 간격 초)
    var onFrame: ((CFTimeInterval, CFTimeInterval) -> Void)?

    private var displayLink: CADisplayLink?
    private var lastTimestamp: CFTimeInterval = 0

    func start(preferredFramesPerSecond: Int) {
        let begin = { [weak self] in
            guard let self else { return }
            self.displayLink?.invalidate()
            let link = CADisplayLink(target: WeakProxy(self), selector: #selector(WeakProxy.tick(_:)))
            if preferredFramesPerSecond > 0 {
                link.preferredFrameRateRange = CAFrameRateRange(
                    minimum: Float(max(preferredFramesPerSecond / 2, 1)),
                    maximum: Float(preferredFramesPerSecond),
                    preferred: Float(preferredFramesPerSecond)
                )
            }
            link.add(to: .main, forMode: .common)
            self.displayLink = link
            self.lastTimestamp = 0
        }
        if Thread.isMainThread { begin() } else { DispatchQueue.main.async(execute: begin) }
    }

    func stop() {
        let end = { [weak self] in
            self?.displayLink?.invalidate()
            self?.displayLink = nil
        }
        if Thread.isMainThread { end() } else { DispatchQueue.main.async(execute: end) }
    }

    deinit {
        displayLink?.invalidate()
    }

    fileprivate func handle(_ link: CADisplayLink) {
        let delta = lastTimestamp > 0 ? link.timestamp - lastTimestamp : 0
        lastTimestamp = link.timestamp
        onFrame?(link.timestamp, delta)
    }

    /// CADisplayLink는 타깃을 강하게 잡으므로 프록시로 순환 참조를 끊는다.
    private final class WeakProxy: NSObject {
        private weak var owner: WebGPUFrameTicker?
        init(_ owner: WebGPUFrameTicker) { self.owner = owner }

        @objc func tick(_ link: CADisplayLink) {
            owner?.handle(link)
        }
    }
}
#endif
