import XCTest
@testable import LynxWebGPUCore

/// Materializing the host payload at the boundary.
///
/// Receiving an `NSDictionary` as `[String: Any]` moves only **the top level** into native storage;
/// the arrays and dictionaries inside still point at the host's Objective-C objects. The engine reads
/// nested descriptors when the owning command runs, not when the batch starts, so a host that reuses
/// its conversion output mid-call would be read through that window. These tests close it.
final class WGPUPayloadTests: XCTestCase {
    func test_materializedPayloadSurvivesTheSourceBeingEmptied() throws {
        let command = NSMutableDictionary()
        command["op"] = "writeBuffer"
        command["buffer"] = 3
        let commands = NSMutableArray(array: [command])
        let source = NSMutableDictionary()
        source["commands"] = commands

        let owned = WGPUPayload.materialize(try XCTUnwrap(source as? [String: Any]))

        // The host reusing or releasing its conversion output.
        command.removeAllObjects()
        commands.removeAllObjects()
        source.removeAllObjects()

        let decoded = try WGPUValueReader(owned).requiredObjects("commands")
        XCTAssertEqual(decoded.count, 1)
        XCTAssertEqual(try decoded[0].requiredString("op"), "writeBuffer")
        XCTAssertEqual(try decoded[0].requiredInt("buffer"), 3)
    }

    /// The shape of the observed crash — nested descriptors are read well into the batch.
    func test_nestedContainersReadWhenACommandRunsAreMaterializedToo() throws {
        let attachments = NSMutableArray(array: [["view": 1, "loadOp": "clear"]])
        let pass = NSMutableDictionary()
        pass["op"] = "beginRenderPass"
        pass["colorAttachments"] = attachments
        let source = NSMutableDictionary()
        source["commands"] = NSMutableArray(array: [pass])

        let owned = WGPUPayload.materialize(try XCTUnwrap(source as? [String: Any]))
        // Same order the engine uses — the command list is walked when the batch starts.
        let commands = try WGPUValueReader(owned).requiredObjects("commands")

        // The host reuses the nested container before that command gets to run.
        attachments.removeAllObjects()

        XCTAssertEqual(try commands[0].requiredObjects("colorAttachments").count, 1)
    }

    /// Proof that the two tests above are not vacuous.
    ///
    /// If this assertion ever fails the Swift runtime started copying nested levels as well — the hazard
    /// `WGPUPayload` defends against would be gone, and it could be removed along with this test.
    func test_withoutMaterializationNestedContainersStillSeeTheSource() throws {
        let attachments = NSMutableArray(array: [["view": 1]])
        let source = NSMutableDictionary()
        source["colorAttachments"] = attachments

        let bridged = try XCTUnwrap(source as? [String: Any])
        attachments.removeAllObjects()

        XCTAssertEqual((bridged["colorAttachments"] as? [Any])?.count, 0)
    }

    func test_materializationPreservesStructureAndLeafTypes() throws {
        let bytes = Data([0xDE, 0xAD, 0xBE, 0xEF])
        let source: [String: Any] = [
            "commands": [
                ["op": "writeBuffer", "data": bytes, "bufferOffset": 16],
                ["op": "beginRenderPass", "colorAttachments": [
                    ["view": 7, "clearValue": ["r": 0.25, "g": 0.5, "b": 0.75, "a": 1.0]],
                ]],
            ],
            "label": NSNull(),
        ]

        let reader = WGPUValueReader(WGPUPayload.materialize(source))
        let commands = try reader.requiredObjects("commands")

        XCTAssertEqual(try commands[0].requiredData("data"), bytes)
        XCTAssertEqual(commands[0].int("bufferOffset", default: 0), 16)

        let attachment = try commands[1].requiredObjects("colorAttachments")[0]
        XCTAssertEqual(try attachment.requiredHandle("view"), WGPUHandle(7))
        XCTAssertEqual(try attachment.color("clearValue", default: .black).blue, 0.75)

        XCTAssertFalse(reader.has("label"), "NSNull must stay an absent value")
        XCTAssertEqual(try commands[1].requiredString("op"), "beginRenderPass")
    }

    /// Leaves move by reference — deep-copying a texture upload's `NSData` every frame would rewrite megabytes.
    func test_leafDataMovesByReferenceRatherThanBeingCopied() throws {
        let payload = NSMutableData(data: Data(repeating: 7, count: 4096))
        let source: [String: Any] = ["commands": [["op": "writeBuffer", "data": payload]]]

        let owned = WGPUPayload.materialize(source)
        let command = try XCTUnwrap((owned["commands"] as? [Any])?[0] as? [String: Any])

        XCTAssertTrue(command["data"] as AnyObject === payload, "a leaf must stay the same object")
    }

    func test_emptyPayloadsAndContainersPassThrough() {
        XCTAssertTrue(WGPUPayload.materialize([:]).isEmpty)

        let owned = WGPUPayload.materialize(["commands": [], "nested": ["a": [String: Any]()]])
        XCTAssertEqual((owned["commands"] as? [Any])?.count, 0)
        XCTAssertEqual(((owned["nested"] as? [String: Any])?["a"] as? [String: Any])?.count, 0)
    }
}
