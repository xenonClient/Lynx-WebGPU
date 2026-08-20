import Foundation

/// Materializes a host payload into **pure Swift containers**.
///
/// ## Why this is needed
///
/// Lynx converts JS values into `NSDictionary`/`NSArray` before handing them over. Receiving one as
/// `[String: Any]` does **not** copy it: Swift moves only the **top level** into native storage, and
/// every nested array or dictionary underneath stays a window onto the host-owned Objective-C object,
/// re-read through `objectForKey:` on every access.
///
/// The engine reads nested descriptors (`colorAttachments`, `entries`, `vertex.buffers`, …) not at the
/// start of a batch but **when the command that owns them runs**. So if the host reuses its conversion
/// output before the call returns, commands late in the batch read whatever now occupies that memory.
/// A crash of exactly this shape was observed: the batch loop copied a command reader on the JS thread
/// and `objc_retain` faulted reading a garbage isa — the bytes there were a float payload, not an
/// object header.
///
/// So the payload is walked **once** at the entry point and moved into native `Dictionary`/`Array`
/// storage. At that moment the call is still inside the host and the originals are alive; afterwards
/// the engine holds its own references and nothing it reads can be pulled out from under it.
///
/// ## Leaves are not copied
///
/// `NSString`, `NSNumber` and `NSData` are carried over **by reference**. Keeping a value alive takes
/// one strong reference, not a byte copy, and deep-copying a texture upload's `NSData` every frame
/// would rewrite megabytes for nothing. Leaves are also consumed within the command that reads them,
/// so they are never held across the batch the way containers are.
///
/// ## Cost
///
/// One extra walk of the payload, against which the read path gets **faster**: a lazily bridged
/// dictionary pays `objc_msgSend` plus value bridging per field, while materialized storage costs one
/// native hash lookup. A command reads several fields, so this usually more than pays for itself.
public enum WGPUPayload {
    /// Moves a dictionary into native containers, nested levels included.
    public static func materialize(_ payload: [String: Any]) -> [String: Any] {
        var result = [String: Any](minimumCapacity: payload.count)
        for (key, value) in payload {
            result[key] = materializeValue(value)
        }
        return result
    }

    /// Moves one value. Containers recurse; everything else is carried over as is.
    public static func materializeValue(_ value: Any) -> Any {
        // Order matters — `Data`, `NSNull` and `NSNumber` match neither cast and fall through untouched.
        if let dictionary = value as? [String: Any] {
            return materialize(dictionary)
        }
        if let array = value as? [Any] {
            var result = [Any]()
            result.reserveCapacity(array.count)
            for element in array {
                result.append(materializeValue(element))
            }
            return result
        }
        return value
    }
}
