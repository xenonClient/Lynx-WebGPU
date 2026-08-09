import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// The upload staging pool — reuse, size classes and the total cap.
final class StagingPoolTests: XCTestCase {
    private var device: MTLDevice!

    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
        try XCTSkipIf(device == nil, "no Metal device")
    }

    func test_sizesRoundUpToPowerOfTwoClassesWithA4KBFloor() {
        XCTAssertEqual(WGPUStagingPool.sizeClass(for: 1), 4096)
        XCTAssertEqual(WGPUStagingPool.sizeClass(for: 4096), 4096)
        XCTAssertEqual(WGPUStagingPool.sizeClass(for: 4097), 8192)
        XCTAssertEqual(WGPUStagingPool.sizeClass(for: 100_000), 131_072)
    }

    func test_aRecycledBufferIsReusedAsTheSameInstance() throws {
        let pool = WGPUStagingPool(device: device)
        let first = try pool.acquire(Data([1, 2, 3, 4]))
        pool.recycle([first])
        XCTAssertEqual(pool.pooledBufferCount, 1)

        let second = try pool.acquire(Data([5, 6]))
        XCTAssertTrue(first === second, "a buffer of the right size must not be rebuilt")
        XCTAssertEqual(pool.pooledBufferCount, 0)
        // A reused buffer is refilled with the new data too.
        XCTAssertEqual(second.contents().load(as: UInt8.self), 5)
    }

    func test_choosesTheSmallestBufferThatFits() throws {
        let pool = WGPUStagingPool(device: device)
        let small = try pool.acquire(Data(count: 100))          // the 4096 class
        let large = try pool.acquire(Data(count: 50_000))       // the 65536 class
        pool.recycle([large, small])

        let picked = try pool.acquire(Data(count: 10))
        XCTAssertTrue(picked === small, "a small upload must not tie up a large buffer")
    }

    func test_buffersPastTheTotalCapAreNotRecycled() throws {
        let pool = WGPUStagingPool(device: device, maxPooledBytes: 8192)
        let buffers = try (0..<3).map { _ in try pool.acquire(Data(count: 4096)) }
        pool.recycle(buffers)

        XCTAssertEqual(pool.pooledBufferCount, 2, "an 8KB cap keeps at most two 4KB buffers")
        XCTAssertEqual(pool.pooledByteCount, 8192)
    }

    func test_aMinimumLengthLargerThanTheDataReservesThatMuch() throws {
        let pool = WGPUStagingPool(device: device)
        let buffer = try pool.acquire(Data([1]), minimumLength: 10_000)
        XCTAssertGreaterThanOrEqual(buffer.length, 10_000)
        XCTAssertEqual(buffer.contents().load(as: UInt8.self), 1)
    }

    // MARK: - Interpreter integration

    /// Whether staging is not rebuilt per frame (execute) but returns from a completed frame and is reused.
    func test_theStagingPoolStaysAtOneAcrossFrames() throws {
        let harness = try XCTUnwrap(RenderHarness.make(width: 8, height: 8))
        let payload: [Float] = [1, 2, 3, 4]
        harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.copyDst],
            ["op": "writeBuffer", "buffer": 1, "data": payload.base64],
        ])

        for frame in 0..<4 {
            // Wait until the previous frame's completion handler runs — recycling happens at GPU completion.
            XCTAssertTrue(
                waitUntil { harness.context!.stagingPool.pooledBufferCount == 1 },
                "frame \(frame): after completion the pool must hold 1 buffer"
            )
            harness.executeExpectingSuccess([
                ["op": "writeBuffer", "buffer": 1, "data": payload.base64],
            ])
            XCTAssertLessThanOrEqual(
                harness.context!.stagingPool.pooledBufferCount, 1,
                "frame \(frame): a pool growing with the frame count means reuse is not happening"
            )
        }
        XCTAssertTrue(waitUntil { harness.context!.stagingPool.pooledBufferCount == 1 })
        XCTAssertEqual(harness.context!.stagingPool.pooledByteCount, 4096, "a 16B upload needs one 4KB class")
    }

    private func waitUntil(timeout: TimeInterval = 2, _ condition: () -> Bool) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            Thread.sleep(forTimeInterval: 0.005)
        }
        return condition()
    }
}
