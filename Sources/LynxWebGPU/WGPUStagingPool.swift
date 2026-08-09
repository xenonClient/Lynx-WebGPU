import Foundation
import Metal
import LynxWebGPUCore

/// Staging buffer pool for `writeBuffer` / `writeTexture` uploads.
///
/// Calling `makeBuffer` per upload piles up allocations and frees every frame. Instead, the frame's
/// staging buffers come back **when the frame command buffer completes** and are reused next frame.
/// Before completion the command buffer references (retains) them, so that is the only safe moment to reuse.
///
/// - Sizes round up to power-of-two classes (with a 4KB floor) to raise the reuse rate.
/// - The pool total is capped so a one-off large upload does not hold memory indefinitely.
///
/// Threading — `acquire` comes from the JS thread, `recycle` from a Metal completion handler thread. A lock serializes them.
final class WGPUStagingPool {
    private let device: MTLDevice
    private let lock = NSLock()
    private var free: [MTLBuffer] = []
    private var pooledBytes = 0
    /// Cap on total bytes kept in the pool. Buffers over it are not recycled but simply released.
    private let maxPooledBytes: Int

    init(device: MTLDevice, maxPooledBytes: Int = 32 << 20) {
        self.device = device
        self.maxPooledBytes = maxPooledBytes
    }

    /// Hands back a staging buffer filled with `data`. A larger `minimumLength` reserves that much headroom.
    func acquire(_ data: Data, minimumLength: Int = 0) throws -> MTLBuffer {
        let buffer = try buffer(fitting: max(data.count, minimumLength, 1))
        data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            buffer.contents().copyMemory(from: base, byteCount: data.count)
        }
        return buffer
    }

    /// Takes the frame's staging buffers back when the frame command buffer completes.
    func recycle(_ buffers: [MTLBuffer]) {
        lock.lock()
        defer { lock.unlock() }
        for buffer in buffers where pooledBytes + buffer.length <= maxPooledBytes {
            free.append(buffer)
            pooledBytes += buffer.length
        }
        // A buffer refused by the cap loses its last reference here and is released.
    }

    private func buffer(fitting needed: Int) throws -> MTLBuffer {
        lock.lock()
        // Choose the smallest buffer that fits — so a small upload does not tie up a large one.
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
            throw WGPUError.outOfMemory("upload staging buffer creation failed (\(length)B)")
        }
        buffer.label = "webgpu.staging"
        return buffer
    }

    /// Power-of-two rounding with a 4KB floor.
    static func sizeClass(for length: Int) -> Int {
        var size = 4096
        while size < length { size <<= 1 }
        return size
    }

    // MARK: - Test observation

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
