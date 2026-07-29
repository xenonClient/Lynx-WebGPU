import Foundation
import LynxWebGPUCore

/// 파싱된 WGSL 셰이더 모듈.
///
/// `createShaderModule` 시점에는 **파싱과 리플렉션만** 한다. MSL 방출은
/// 파이프라인 레이아웃이 정해져야 바인딩 인덱스를 붙일 수 있으므로 파이프라인 생성 시점으로 미룬다
/// (`translateToMSL(entryPoints:bindings:)`).
public final class WGSLShaderModule {
    /// 원본 WGSL.
    public let source: String
    /// 파이프라인 생성에 필요한 진입점·바인딩 정보.
    public let reflection: WGSLShaderReflection

    private let ast: WGSLModule

    public init(source: String) throws {
        self.source = source
        self.ast = try WGSLParser.parse(source)
        self.reflection = WGSLReflectionBuilder.build(ast)
    }

    /// 지정한 진입점들을 담은 MSL 소스를 만든다.
    ///
    /// - Parameters:
    ///   - entryPoints: 이 파이프라인이 쓰는 진입점 이름 (버텍스/프래그먼트 또는 컴퓨트 1개).
    ///   - bindings: `@group/@binding` → Metal 인덱스 배정.
    public func translateToMSL(entryPoints: [String], bindings: WGSLBindingAssignment) throws -> String {
        var emitter = MSLEmitter(module: ast, reflection: reflection, bindings: bindings)
        return try emitter.emit(entryPoints: entryPoints)
    }

    /// `layout: "auto"` 파이프라인이 쓸 바인드 그룹 레이아웃을 셰이더 선언에서 유도한다.
    ///
    /// 그룹 인덱스 순서의 배열을 돌려주며, 쓰이지 않는 그룹 자리는 빈 배열이다
    /// (Metal 인덱스 배정이 그룹 순서에 의존하므로 자리를 비워 둬야 한다).
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

    /// WGSL 진입점 이름에 대응하는 **MSL 함수 이름**.
    ///
    /// `main`처럼 MSL이 거부하는 이름은 방출 시 바뀌므로, `MTLLibrary.makeFunction(name:)`에는
    /// 반드시 이 값을 넘겨야 한다.
    public static func mslFunctionName(for entryPoint: String) -> String {
        MSLTypeMapping.functionName(entryPoint)
    }

    /// 진입점의 `@workgroup_size` — 컴퓨트 디스패치에서 threadsPerThreadgroup으로 쓴다.
    public func workgroupSize(of entryPoint: String) -> (x: Int, y: Int, z: Int)? {
        reflection.entryPoint(named: entryPoint)?.workgroupSize
    }

    /// 진입점이 존재하고 기대한 스테이지인지 확인한다.
    public func requireEntryPoint(_ name: String, stage: WGSLStage) throws -> WGSLEntryPointInfo {
        guard let entryPoint = reflection.entryPoint(named: name) else {
            let available = reflection.entryPoints.map { "\($0.name)(\($0.stage.rawValue))" }
            throw WGPUError.validation(
                "셰이더에 진입점 '\(name)'이(가) 없다 (있는 것: \(available.joined(separator: ", ")))"
            )
        }
        guard entryPoint.stage == stage else {
            throw WGPUError.validation(
                "진입점 '\(name)'은(는) \(entryPoint.stage.rawValue) 셰이더인데 \(stage.rawValue)로 쓰였다"
            )
        }
        return entryPoint
    }
}
