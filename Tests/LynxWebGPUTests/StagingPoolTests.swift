import XCTest
import Metal
import LynxWebGPUCore
@testable import LynxWebGPU

/// 업로드 스테이징 풀 — 재사용·크기 클래스·총량 상한 계약.
final class StagingPoolTests: XCTestCase {
    private var device: MTLDevice!

    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
        try XCTSkipIf(device == nil, "Metal 디바이스 없음")
    }

    func test_크기는_4KB_바닥의_2제곱_클래스로_반올림된다() {
        XCTAssertEqual(WGPUStagingPool.sizeClass(for: 1), 4096)
        XCTAssertEqual(WGPUStagingPool.sizeClass(for: 4096), 4096)
        XCTAssertEqual(WGPUStagingPool.sizeClass(for: 4097), 8192)
        XCTAssertEqual(WGPUStagingPool.sizeClass(for: 100_000), 131_072)
    }

    func test_회수한_버퍼를_같은_인스턴스로_재사용한다() throws {
        let pool = WGPUStagingPool(device: device)
        let first = try pool.acquire(Data([1, 2, 3, 4]))
        pool.recycle([first])
        XCTAssertEqual(pool.pooledBufferCount, 1)

        let second = try pool.acquire(Data([5, 6]))
        XCTAssertTrue(first === second, "크기가 맞는 버퍼는 새로 만들지 않아야 한다")
        XCTAssertEqual(pool.pooledBufferCount, 0)
        // 재사용 버퍼에도 데이터가 새로 채워진다.
        XCTAssertEqual(second.contents().load(as: UInt8.self), 5)
    }

    func test_맞는_것_중_가장_작은_버퍼를_고른다() throws {
        let pool = WGPUStagingPool(device: device)
        let small = try pool.acquire(Data(count: 100))          // 4096 클래스
        let large = try pool.acquire(Data(count: 50_000))       // 65536 클래스
        pool.recycle([large, small])

        let picked = try pool.acquire(Data(count: 10))
        XCTAssertTrue(picked === small, "작은 업로드가 큰 버퍼를 붙잡으면 안 된다")
    }

    func test_총량_상한을_넘는_버퍼는_회수하지_않는다() throws {
        let pool = WGPUStagingPool(device: device, maxPooledBytes: 8192)
        let buffers = try (0..<3).map { _ in try pool.acquire(Data(count: 4096)) }
        pool.recycle(buffers)

        XCTAssertEqual(pool.pooledBufferCount, 2, "8KB 상한이면 4KB 두 개까지만 남는다")
        XCTAssertEqual(pool.pooledByteCount, 8192)
    }

    func test_minimumLength가_데이터보다_크면_그만큼_잡는다() throws {
        let pool = WGPUStagingPool(device: device)
        let buffer = try pool.acquire(Data([1]), minimumLength: 10_000)
        XCTAssertGreaterThanOrEqual(buffer.length, 10_000)
        XCTAssertEqual(buffer.contents().load(as: UInt8.self), 1)
    }

    // MARK: - 해석기 통합

    /// 프레임(execute)마다 스테이징을 새로 만들지 않고, 완료된 프레임의 버퍼가 돌아와 재사용되는지.
    func test_프레임을_거듭해도_스테이징_풀이_1개로_유지된다() throws {
        let harness = try XCTUnwrap(RenderHarness.make(width: 8, height: 8))
        let payload: [Float] = [1, 2, 3, 4]
        harness.executeExpectingSuccess([
            ["op": "createBuffer", "id": 1, "size": 16, "usage": TestUsage.copyDst],
            ["op": "writeBuffer", "buffer": 1, "data": payload.base64],
        ])

        for frame in 0..<4 {
            // 직전 프레임의 완료 핸들러가 돌 때까지 기다린다 — 회수는 GPU 완료 시점이다.
            XCTAssertTrue(
                waitUntil { harness.context!.stagingPool.pooledBufferCount == 1 },
                "프레임 \(frame): 완료 후 풀에 버퍼 1개가 있어야 한다"
            )
            harness.executeExpectingSuccess([
                ["op": "writeBuffer", "buffer": 1, "data": payload.base64],
            ])
            XCTAssertLessThanOrEqual(
                harness.context!.stagingPool.pooledBufferCount, 1,
                "프레임 \(frame): 풀이 프레임 수만큼 자라면 재사용이 안 되는 것이다"
            )
        }
        XCTAssertTrue(waitUntil { harness.context!.stagingPool.pooledBufferCount == 1 })
        XCTAssertEqual(harness.context!.stagingPool.pooledByteCount, 4096, "16B 업로드는 4KB 클래스 하나면 된다")
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
