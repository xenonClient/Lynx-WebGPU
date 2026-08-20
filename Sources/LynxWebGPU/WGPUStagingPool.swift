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
    ///
    /// ## Why this copies without `withUnsafeBytes`
    ///
    /// A field crash died in the batch loop retaining a "thrown error" that was not an error: the pointer
    /// it tried to retain was page-aligned and held half-float pixels — a **staging buffer**. Once the
    /// engine started recording which command was running, the op turned out to be `writeTexture`, and
    /// that lands here.
    ///
    /// On arm64 the Swift error return travels in **x21**, which is also an ordinary callee-saved register
    /// an optimizer may borrow. In the shipped `-Osize` binary the `withUnsafeBytes` closure this function
    /// used did exactly that: `mov x21, x0` right after `contents()`, twice. A caller that reads x21 as the
    /// thrown error then gets a buffer pointer, and retaining it reads pixel data as an object header —
    /// which is the crash, and why a debug build never showed it.
    ///
    /// `copyBytes(to:count:)` needs no closure, so that site is gone; the copy is the same memcpy.
    ///
    /// **This is a mitigation, not a proven fix.** Rebuilding the same source here (Xcode 17C52, iOS arm64
    /// and macOS arm64, `-Osize`) never reproduced that register choice — it appears only in the shipped
    /// build (Xcode 17F106, arm64e). So the site was removed rather than the bug being understood. Before
    /// shipping, check the built archive directly:
    ///
    /// ```zsh
    /// otool -arch arm64 -tV …/LynxWebGPU.framework/LynxWebGPU | grep -n "mov.*x21, x0"
    /// ```
    func acquire(_ data: Data, minimumLength: Int = 0) throws -> MTLBuffer {
        let buffer = try buffer(fitting: max(data.count, minimumLength, 1))
        guard !data.isEmpty else { return buffer }
        data.copyBytes(to: buffer.contents().assumingMemoryBound(to: UInt8.self), count: data.count)
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
