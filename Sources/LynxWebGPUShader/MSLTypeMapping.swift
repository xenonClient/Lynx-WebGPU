import Foundation
import LynxWebGPUCore

/// Table mapping WGSL type and builtin names into Metal Shading Language.
///
/// Most of it is 1:1 (WGSL and MSL share vector/matrix syntax and many math function names).
/// Names absent here **pass through unchanged** — if the Metal compiler does not know a name it
/// raises a clear error at that point, so the translator need not know every name.
enum MSLTypeMapping {
    // MARK: - Types

    static func scalar(_ name: String) -> String {
        switch name {
        case "f32": return "float"
        case "i32": return "int"
        case "u32": return "uint"
        case "f16": return "half"
        case "bool": return "bool"
        default: return name
        }
    }

    /// The MSL type name used in value position. Runtime-sized arrays cannot be expressed here.
    static func type(_ type: WGSLType, module: WGSLModule) throws -> String {
        switch type {
        case .void:
            return "void"
        case .scalar(let name):
            return scalar(name)
        case .vector(let size, let element):
            guard case .scalar(let name) = element else {
                throw WGPUError.unsupported("WGSL: vector components must be scalars")
            }
            return "\(scalar(name))\(size)"
        case .matrix(let columns, let rows, let element):
            guard case .scalar(let name) = element else {
                throw WGPUError.unsupported("WGSL: matrix components must be scalars")
            }
            return "\(scalar(name))\(columns)x\(rows)"
        case .array(let element, let count):
            guard let count else {
                throw WGPUError.unsupported(
                    "WGSL: a runtime-sized array (array<T>) can only be used in a storage buffer binding"
                )
            }
            guard let length = WGSLLayout.constantValue(count, module: module) else {
                throw WGPUError.unsupported("WGSL: the array length cannot be computed at compile time")
            }
            return "array<\(try self.type(element, module: module)), \(length)>"
        case .atomic(let element):
            guard case .scalar(let name) = element else {
                throw WGPUError.unsupported("WGSL: atomic components must be i32 or u32")
            }
            return name == "u32" ? "atomic_uint" : "atomic_int"
        case .pointer(let space, let element, _):
            return "\(addressSpace(space)) \(try self.type(element, module: module))*"
        case .sampler(let comparison):
            return comparison ? "sampler" : "sampler"
        case .texture(let texture):
            return try textureType(texture, module: module)
        case .named(let name):
            return identifier(name)
        }
    }

    static func addressSpace(_ space: String) -> String {
        switch space {
        case "uniform": return "constant"
        case "storage": return "device"
        case "workgroup": return "threadgroup"
        case "private", "function": return "thread"
        default: return "thread"
        }
    }

    static func textureType(_ texture: WGSLTextureType, module: WGSLModule) throws -> String {
        let component: String
        switch texture.kind {
        case .storage:
            component = storageFormatComponent(texture.format)
        case .depth, .depthMultisampled:
            component = "float"
        default:
            if case .scalar(let name)? = texture.sampleType {
                component = scalar(name)
            } else {
                component = "float"
            }
        }

        let base: String
        switch (texture.kind, texture.dimension) {
        case (.depth, "2d"): base = "depth2d"
        case (.depth, "2d_array"): base = "depth2d_array"
        case (.depth, "cube"): base = "depthcube"
        case (.depth, "cube_array"): base = "depthcube_array"
        case (.depthMultisampled, _): base = "depth2d_ms"
        case (.multisampled, _): base = "texture2d_ms"
        case (_, "1d"): base = "texture1d"
        case (_, "2d"): base = "texture2d"
        case (_, "2d_array"): base = "texture2d_array"
        case (_, "3d"): base = "texture3d"
        case (_, "cube"): base = "texturecube"
        case (_, "cube_array"): base = "texturecube_array"
        default:
            throw WGPUError.unsupported("WGSL: unsupported texture dimension '\(texture.dimension)'")
        }

        if texture.kind == .storage {
            let access: String
            switch texture.access {
            case "read": access = "access::read"
            case "read_write": access = "access::read_write"
            default: access = "access::write"
            }
            return "\(base)<\(component), \(access)>"
        }
        return "\(base)<\(component)>"
    }

    private static func storageFormatComponent(_ format: String?) -> String {
        guard let format else { return "float" }
        if format.hasSuffix("uint") { return "uint" }
        if format.hasSuffix("sint") { return "int" }
        return "float"
    }

    // MARK: - Name collisions

    /// Ordinary identifiers in WGSL that are reserved words in MSL (C++14 plus Metal extensions).
    ///
    /// Names such as `texture`, `sampler`, `device` and `char` are extremely common in graphics
    /// shaders. Emitting them as-is either breaks the compile or (worse) changes meaning, so they all
    /// get a prefix.
    private static let reservedIdentifiers: Set<String> = [
        // Metal address spaces / function qualifiers
        "device", "constant", "threadgroup", "thread", "kernel", "vertex", "fragment",
        "texture", "sampler", "access", "ray_data", "object_data",
        // C++ keywords (only those WGSL does not already forbid)
        "char", "short", "long", "signed", "unsigned", "class", "union", "enum",
        "template", "typename", "namespace", "using", "public", "private", "protected",
        "virtual", "operator", "new", "delete", "this", "throw", "try", "catch", "friend",
        "inline", "static", "extern", "register", "volatile", "mutable", "explicit",
        "export", "typedef", "sizeof", "alignof", "decltype", "auto", "constexpr", "nullptr",
        "goto", "do", "typeid", "and", "or", "not", "xor", "compl", "bitand", "bitor",
        // MSL scalar/vector type names
        "half", "uchar", "ushort", "size_t", "ptrdiff_t",
    ]

    /// Names unusable as function names (`main` is not a C++ keyword but Metal rejects it).
    private static let reservedFunctionNames: Set<String> = ["main"]

    /// Renames an entry point or function to a name that is safe in MSL.
    ///
    /// The name the runtime passes to `MTLLibrary.makeFunction(name:)` must go through this function too.
    /// (`WGSLShaderModule.mslFunctionName(for:)`).
    static func functionName(_ name: String) -> String {
        if reservedFunctionNames.contains(name) { return "wgpu_fn_\(name)" }
        return identifier(name)
    }

    /// Renames a variable, parameter, struct or member to a name that is safe in MSL.
    /// **The declaration and every use must go through the same function** or the names diverge.
    static func identifier(_ name: String) -> String {
        reservedIdentifiers.contains(name) ? "wgpu_id_\(name)" : name
    }

    // MARK: - Builtin functions

    /// Those that differ only in name.
    static let renamedBuiltins: [String: String] = [
        "inverseSqrt": "rsqrt",
        "dpdx": "dfdx",
        "dpdxCoarse": "dfdx",
        "dpdxFine": "dfdx",
        "dpdy": "dfdy",
        "dpdyCoarse": "dfdy",
        "dpdyFine": "dfdy",
        "fwidthCoarse": "fwidth",
        "fwidthFine": "fwidth",
        "faceForward": "faceforward",
        "countLeadingZeros": "clz",
        "countTrailingZeros": "ctz",
        "countOneBits": "popcount",
        "reverseBits": "reverse_bits",
        "extractBits": "extract_bits",
        "insertBits": "insert_bits",
        "pack4x8snorm": "pack_float_to_snorm4x8",
        "pack4x8unorm": "pack_float_to_unorm4x8",
        "pack2x16snorm": "pack_float_to_snorm2x16",
        "pack2x16unorm": "pack_float_to_unorm2x16",
        "pack2x16float": "pack_float_to_half2x16",
        "unpack4x8snorm": "unpack_snorm4x8_to_float",
        "unpack4x8unorm": "unpack_unorm4x8_to_float",
        "unpack2x16snorm": "unpack_snorm2x16_to_float",
        "unpack2x16unorm": "unpack_unorm2x16_to_float",
        "unpack2x16float": "unpack_half2x16_to_float",
    ]

    /// Scalar conversion constructors (`f32(x)` → `float(x)`).
    static let scalarConstructors: Set<String> = ["f32", "i32", "u32", "bool", "f16"]

    /// The `atomicAdd` → `atomic_fetch_add_explicit` family.
    static let atomicFetchOperations: [String: String] = [
        "atomicAdd": "atomic_fetch_add_explicit",
        "atomicSub": "atomic_fetch_sub_explicit",
        "atomicMax": "atomic_fetch_max_explicit",
        "atomicMin": "atomic_fetch_min_explicit",
        "atomicAnd": "atomic_fetch_and_explicit",
        "atomicOr": "atomic_fetch_or_explicit",
        "atomicXor": "atomic_fetch_xor_explicit",
    ]

    /// Builtins this implementation cannot translate yet — rejected explicitly rather than translated silently wrong.
    static let unsupportedBuiltins: [String: String] = [
        "atomicCompareExchangeWeak": "the return struct (__atomic_compare_exchange_result) cannot be translated",
        "modf": "the return struct cannot be translated. Split it into floor/fract",
        "frexp": "the return struct cannot be translated",
        "workgroupUniformLoad": "not supported",
    ]

    /// WGSL `@builtin(x)` → the MSL attribute and type.
    static func builtin(_ name: String, stage: WGSLStage, isInput: Bool) throws -> (attribute: String, type: String) {
        switch name {
        case "vertex_index": return ("[[vertex_id]]", "uint")
        case "instance_index": return ("[[instance_id]]", "uint")
        case "position":
            return (isInput && stage == .fragment ? "[[position]]" : "[[position]]", "float4")
        case "front_facing": return ("[[front_facing]]", "bool")
        case "frag_depth": return ("[[depth(any)]]", "float")
        case "sample_index": return ("[[sample_id]]", "uint")
        case "primitive_index": return ("[[primitive_id]]", "uint")
        case "sample_mask": return ("[[sample_mask]]", "uint")
        case "local_invocation_id": return ("[[thread_position_in_threadgroup]]", "uint3")
        case "local_invocation_index": return ("[[thread_index_in_threadgroup]]", "uint")
        case "global_invocation_id": return ("[[thread_position_in_grid]]", "uint3")
        case "workgroup_id": return ("[[threadgroup_position_in_grid]]", "uint3")
        case "num_workgroups": return ("[[threadgroups_per_grid]]", "uint3")
        default:
            throw WGPUError.unsupported("WGSL: unsupported @builtin(\(name))")
        }
    }

    /// `@interpolate(...)` → the MSL interpolation attribute.
    static func interpolation(_ attribute: WGSLAttribute?) -> String {
        guard let attribute, let kind = attribute.arguments.first else { return "" }
        switch kind {
        case "flat": return " [[flat]]"
        case "linear": return " [[center_no_perspective]]"
        default: return ""
        }
    }
}
