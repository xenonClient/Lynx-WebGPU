import Foundation
import Metal
import LynxWebGPUCore

/// `writeBuffer` / `writeTexture` 업로드용 스테이징 버퍼 풀.
///
/// 업로드마다 `makeBuffer`를 부르면 매 프레임 할당·해제가 쌓인다. 대신 프레임 커맨드 버퍼가
/// **완료되는 시점에** 그 프레임의 스테이징 버퍼를 돌려받아 다음 프레임에 재사용한다.
/// 완료 전에는 커맨드 버퍼가 버퍼를 참조하므로(리테인) 재사용해도 안전한 시점이 이때뿐이다.
///
/// - 크기는 2의 거듭제곱 클래스(4KB 바닥)로 반올림해 재사용률을 높인다.
/// - 풀 총량에 상한을 두어, 일시적 대형 업로드가 메모리를 계속 붙잡지 않게 한다.
///
/// 스레딩 — `acquire`는 JS 스레드, `recycle`은 Metal 완료 핸들러 스레드에서 온다. 락으로 직렬화한다.
final class WGPUStagingPool {
    private let device: MTLDevice
    private let lock = NSLock()
    private var free: [MTLBuffer] = []
    private var pooledBytes = 0
    /// 풀에 남겨 둘 총 바이트 상한. 넘치는 버퍼는 회수하지 않고 그대로 해제한다.
    private let maxPooledBytes: Int

    init(device: MTLDevice, maxPooledBytes: Int = 32 << 20) {
        self.device = device
        self.maxPooledBytes = maxPooledBytes
    }

    /// `data`를 채운 스테이징 버퍼를 준다. `minimumLength`가 더 크면 그만큼 여유 있게 잡는다.
    func acquire(_ data: Data, minimumLength: Int = 0) throws -> MTLBuffer {
        let buffer = try buffer(fitting: max(data.count, minimumLength, 1))
        data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            buffer.contents().copyMemory(from: base, byteCount: data.count)
        }
        return buffer
    }

    /// 프레임 커맨드 버퍼 완료 시점에 그 프레임의 스테이징 버퍼들을 돌려받는다.
    func recycle(_ buffers: [MTLBuffer]) {
        lock.lock()
        defer { lock.unlock() }
        for buffer in buffers where pooledBytes + buffer.length <= maxPooledBytes {
            free.append(buffer)
            pooledBytes += buffer.length
        }
        // 상한 때문에 못 들어온 버퍼는 여기서 마지막 참조가 끊겨 해제된다.
    }

    private func buffer(fitting needed: Int) throws -> MTLBuffer {
        lock.lock()
        // 맞는 것 중 가장 작은 버퍼를 고른다 — 작은 업로드가 큰 버퍼를 붙잡지 않게.
        var bestIndex: Int?
        for (index, candidate) in free.enumerated() where candidate.length >= needed {
            if bestIndex == nil || candidate.length < free[bestIndex!].length {
                bestIndex = index
            }
        }
        if let bestIndex {
            let buffer = free.remove(at: bestIndex)
            pooledBytes -= buffer.length
            lock.unlock()
            return buffer
        }
        lock.unlock()

        let length = Self.sizeClass(for: needed)
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw WGPUError.outOfMemory("업로드 staging 버퍼 생성 실패 (\(length)B)")
        }
        buffer.label = "webgpu.staging"
        return buffer
    }

    /// 4KB 바닥의 2의 거듭제곱 반올림.
    static func sizeClass(for length: Int) -> Int {
        var size = 4096
        while size < length { size <<= 1 }
        return size
    }

    // MARK: - 테스트 관찰용

    var pooledBufferCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return free.count
    }

    var pooledByteCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return pooledBytes
    }
}
