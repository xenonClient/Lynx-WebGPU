import Foundation
import LynxWebGPUCore

/// WGSL 타입·내장 함수 이름을 Metal Shading Language로 옮기는 표.
///
/// 대부분은 1:1이다 (WGSL과 MSL 모두 벡터/행렬 문법이 같고 수학 함수 이름도 겹친다).
/// 여기 없는 이름은 **그대로 통과시킨다** — Metal 컴파일러가 모르는 이름이면 그 시점에
/// 명확한 오류를 내므로, 번역기가 이름을 다 알아야 할 필요는 없다.
enum MSLTypeMapping {
    // MARK: - 타입

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

    /// 값 위치에서 쓰는 MSL 타입 이름. 런타임 크기 배열은 여기서 표현할 수 없다.
    static func type(_ type: WGSLType, module: WGSLModule) throws -> String {
        switch type {
        case .void:
            return "void"
        case .scalar(let name):
            return scalar(name)
        case .vector(let size, let element):
            guard case .scalar(let name) = element else {
                throw WGPUError.unsupported("WGSL: 벡터 성분은 스칼라여야 한다")
            }
            return "\(scalar(name))\(size)"
        case .matrix(let columns, let rows, let element):
            guard case .scalar(let name) = element else {
                throw WGPUError.unsupported("WGSL: 행렬 성분은 스칼라여야 한다")
            }
            return "\(scalar(name))\(columns)x\(rows)"
        case .array(let element, let count):
            guard let count else {
                throw WGPUError.unsupported(
                    "WGSL: 런타임 크기 배열(array<T>)은 storage 버퍼 바인딩에서만 쓸 수 있다"
                )
            }
            guard let length = WGSLLayout.constantValue(count, module: module) else {
                throw WGPUError.unsupported("WGSL: 배열 길이를 컴파일 타임에 계산할 수 없다")
            }
            return "array<\(try self.type(element, module: module)), \(length)>"
        case .atomic(let element):
            guard case .scalar(let name) = element else {
                throw WGPUError.unsupported("WGSL: atomic 성분은 i32/u32여야 한다")
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
            throw WGPUError.unsupported("WGSL: 지원하지 않는 텍스처 차원 '\(texture.dimension)'")
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

    // MARK: - 이름 충돌

    /// WGSL에서는 평범한 식별자지만 MSL(C++14 + Metal 확장)에서는 예약어인 것들.
    ///
    /// 그래픽스 셰이더에서 `texture` `sampler` `device` `char` 같은 이름은 아주 흔하다.
    /// 그대로 내보내면 컴파일이 깨지거나(운 나쁘면) 다른 의미로 해석되므로 전부 접두사를 붙인다.
    private static let reservedIdentifiers: Set<String> = [
        // Metal 주소 공간 / 함수 한정자
        "device", "constant", "threadgroup", "thread", "kernel", "vertex", "fragment",
        "texture", "sampler", "access", "ray_data", "object_data",
        // C++ 키워드 (WGSL이 금지하지 않는 것만)
        "char", "short", "long", "signed", "unsigned", "class", "union", "enum",
        "template", "typename", "namespace", "using", "public", "private", "protected",
        "virtual", "operator", "new", "delete", "this", "throw", "try", "catch", "friend",
        "inline", "static", "extern", "register", "volatile", "mutable", "explicit",
        "export", "typedef", "sizeof", "alignof", "decltype", "auto", "constexpr", "nullptr",
        "goto", "do", "typeid", "and", "or", "not", "xor", "compl", "bitand", "bitor",
        // MSL 스칼라/벡터 타입 이름
        "half", "uchar", "ushort", "size_t", "ptrdiff_t",
    ]

    /// 함수 이름으로 쓸 수 없는 것 (`main`은 C++ 키워드는 아니지만 Metal이 거부한다).
    private static let reservedFunctionNames: Set<String> = ["main"]

    /// 진입점·함수 이름을 MSL에서 안전한 이름으로 바꾼다.
    ///
    /// 런타임이 `MTLLibrary.makeFunction(name:)`에 넘길 이름도 이 함수를 거쳐야 한다
    /// (`WGSLShaderModule.mslFunctionName(for:)`).
    static func functionName(_ name: String) -> String {
        if reservedFunctionNames.contains(name) { return "wgpu_fn_\(name)" }
        return identifier(name)
    }

    /// 변수·매개변수·구조체·멤버 이름을 MSL에서 안전한 이름으로 바꾼다.
    /// **선언과 모든 사용처가 같은 함수를 거쳐야** 이름이 어긋나지 않는다.
    static func identifier(_ name: String) -> String {
        reservedIdentifiers.contains(name) ? "wgpu_id_\(name)" : name
    }

    // MARK: - 내장 함수

    /// 이름만 다른 것들.
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

    /// 스칼라 변환 생성자 (`f32(x)` → `float(x)`).
    static let scalarConstructors: Set<String> = ["f32", "i32", "u32", "bool", "f16"]

    /// `atomicAdd` → `atomic_fetch_add_explicit` 류.
    static let atomicFetchOperations: [String: String] = [
        "atomicAdd": "atomic_fetch_add_explicit",
        "atomicSub": "atomic_fetch_sub_explicit",
        "atomicMax": "atomic_fetch_max_explicit",
        "atomicMin": "atomic_fetch_min_explicit",
        "atomicAnd": "atomic_fetch_and_explicit",
        "atomicOr": "atomic_fetch_or_explicit",
        "atomicXor": "atomic_fetch_xor_explicit",
    ]

    /// 이 구현이 아직 옮기지 못하는 내장 함수 — 조용히 틀리게 번역하지 않고 명시적으로 거부한다.
    static let unsupportedBuiltins: [String: String] = [
        "arrayLength": "런타임 배열 길이는 버퍼 크기를 셰이더가 알아야 해서 지원하지 않는다. 길이를 유니폼으로 넘길 것",
        "atomicCompareExchangeWeak": "반환 구조체(__atomic_compare_exchange_result)를 옮기지 못한다",
        "modf": "반환 구조체를 옮기지 못한다. floor/fract로 나눠 쓸 것",
        "frexp": "반환 구조체를 옮기지 못한다",
        "workgroupUniformLoad": "지원하지 않는다",
    ]

    /// WGSL `@builtin(x)` → MSL 속성과 타입.
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
            throw WGPUError.unsupported("WGSL: 지원하지 않는 @builtin(\(name))")
        }
    }

    /// `@interpolate(...)` → MSL 보간 속성.
    static func interpolation(_ attribute: WGSLAttribute?) -> String {
        guard let attribute, let kind = attribute.arguments.first else { return "" }
        switch kind {
        case "flat": return " [[flat]]"
        case "linear": return " [[center_no_perspective]]"
        default: return ""
        }
    }
}
