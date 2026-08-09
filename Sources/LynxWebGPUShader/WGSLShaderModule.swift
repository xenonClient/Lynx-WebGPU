import Foundation
import LynxWebGPUCore

/// A parsed WGSL shader module.
///
/// At `createShaderModule` time it does **parsing and reflection only.** MSL emission is deferred to
/// pipeline creation, because binding indices can only be assigned once the pipeline layout is known
/// (`translateToMSL(entryPoints:bindings:)`).
public final class WGSLShaderModule {
    /// The original WGSL.
    public let source: String
    /// Entry point and binding information needed to create a pipeline.
    public let reflection: WGSLShaderReflection

    private let ast: WGSLModule

    public init(source: String) throws {
        self.source = source
        self.ast = try WGSLParser.parse(source)
        self.reflection = WGSLReflectionBuilder.build(ast)
    }

    /// Builds MSL source containing the requested entry points.
    ///
    /// - Parameters:
    ///   - entryPoints: the entry point names this pipeline uses (vertex/fragment, or one compute).
    ///   - bindings: `@group/@binding` → Metal index assignment.
    ///   - constants: pipeline constant (`override`) values. They override the shader's defaults.
    public func translateToMSL(
        entryPoints: [String],
        bindings: WGSLBindingAssignment,
        constants: [String: Double] = [:]
    ) throws -> String {
        var emitter = MSLEmitter(
            module: Self.applying(constants, to: ast), reflection: reflection, bindings: bindings
        )
        return try emitter.emit(entryPoints: entryPoints)
    }

    /// Folds pipeline constants into the AST up front.
    ///
    /// To make the emitter and the layout calculator read the same constant table, **planting the
    /// values in the AST** is safer than carrying them separately (it naturally covers the case where
    /// an array length depends on an override).
    private static func applying(_ constants: [String: Double], to module: WGSLModule) -> WGSLModule {
        guard !constants.isEmpty else { return module }
        var updated = module
        updated.constants = module.constants.map { constant in
            guard constant.isOverride, let supplied = constants[constant.name] else { return constant }
            var replaced = constant
            replaced.value = literal(supplied, type: constant.type)
            return replaced
        }
        return updated
    }

    private static func literal(_ value: Double, type: WGSLType?) -> WGSLExpression {
        let isFloatType: Bool
        if case .scalar(let name)? = type { isFloatType = name == "f32" || name == "f16" } else { isFloatType = false }
        if !isFloatType, value == value.rounded() {
            return .intLiteral(String(Int(value)))
        }
        return .floatLiteral(String(value))
    }

    /// Derives the bind group layouts a `layout: "auto"` pipeline uses from the shader declarations.
    ///
    /// Returns an array in group index order, with an empty array where a group is unused (Metal index
    /// assignment depends on group order, so the slot has to stay).
    public func autoBindGroupLayouts(entryPoints: [String]) -> [[WGPUBindGroupLayoutEntry]] {
        let used = reflection.resources(usedBy: entryPoints)
        guard let maximumGroup = used.map(\.group).max() else { return [] }

        var groups: [[WGPUBindGroupLayoutEntry]] = Array(repeating: [], count: maximumGroup + 1)
        for resource in used {
            groups[resource.group].append(WGPUBindGroupLayoutEntry(
                binding: resource.binding,
                visibility: reflection.visibility(of: resource.name),
                layout: resource.bindingLayout
            ))
        }
        return groups.map { $0.sorted { $0.binding < $1.binding } }
    }

    /// The **MSL function name** corresponding to a WGSL entry point name.
    ///
    /// Names MSL rejects, such as `main`, are renamed during emission, so this value is what must be
    /// passed to `MTLLibrary.makeFunction(name:)`.
    public static func mslFunctionName(for entryPoint: String) -> String {
        MSLTypeMapping.functionName(entryPoint)
    }

    /// Whether these entry points need the buffer size table.
    ///
    /// Needed when they use `arrayLength()` or **index a runtime-sized array** — the latter because
    /// clamping a range requires knowing the bound (robustness). When needed, the runtime must plug
    /// the table into the reserved index (`WGSLMetalLimits.bufferSizesIndex`).
    ///
    /// **This must compute the same thing as the emitter** — a mismatch makes the shader read an unbound buffer.
    public func usesArrayLength(entryPoints: [String]) -> Bool {
        let users = WGSLReflectionBuilder.functionsNeedingBufferSizes(in: ast)
        return entryPoints.contains(where: users.contains)
    }

    /// The entry point's `@workgroup_size` — used as threadsPerThreadgroup in a compute dispatch.
    public func workgroupSize(of entryPoint: String) -> (x: Int, y: Int, z: Int)? {
        reflection.entryPoint(named: entryPoint)?.workgroupSize
    }

    /// The spec's **"get the entry point"** — with no name, use the stage's **only** entry point.
    ///
    /// `entryPoint` is not required by the spec. Omitted, it uses the entry point when exactly one
    /// exists for that stage, and is an error when there is none or more than one. Guessing `"main"`
    /// would reject entire shaders whose entry points are named differently — three.js's mipmap
    /// shaders (`mainVS` + `main_2d`, …) really did break that way.
    public func resolveEntryPoint(_ requested: String?, stage: WGSLStage) throws -> String {
        if let requested {
            _ = try requireEntryPoint(requested, stage: stage)
            return requested
        }
        let candidates = reflection.entryPoints.filter { $0.stage == stage }
        guard let only = candidates.first, candidates.count == 1 else {
            let available = reflection.entryPoints.map { "\($0.name)(\($0.stage.rawValue))" }
            throw WGPUError.validation(
                candidates.isEmpty
                    ? "the shader has no \(stage.rawValue) entry point (available: \(available.joined(separator: ", ")))"
                    : "the shader has \(candidates.count) \(stage.rawValue) entry points, so one cannot be chosen — "
                        + "specify entryPoint (\(candidates.map(\.name).joined(separator: ", ")))"
            )
        }
        return only.name
    }

    /// Checks the entry point exists and belongs to the expected stage.
    public func requireEntryPoint(_ name: String, stage: WGSLStage) throws -> WGSLEntryPointInfo {
        guard let entryPoint = reflection.entryPoint(named: name) else {
            let available = reflection.entryPoints.map { "\($0.name)(\($0.stage.rawValue))" }
            throw WGPUError.validation(
                "the shader has no entry point '\(name)' (available: \(available.joined(separator: ", ")))"
            )
        }
        guard entryPoint.stage == stage else {
            throw WGPUError.validation(
                "entry point '\(name)' is a \(entryPoint.stage.rawValue) shader but was used as \(stage.rawValue)"
            )
        }
        return entryPoint
    }
}
