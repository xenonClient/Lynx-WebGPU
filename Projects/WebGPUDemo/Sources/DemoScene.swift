import Foundation

/// 번들에 들어 있는 데모 씬. 각각 `Resources/<rawValue>.lynx.bundle` 로 번들된다.
///
/// 오프스크린 하네스가 자동으로 검증하는 기능들을 **눈으로도 확인**할 수 있게 1:1로 맞춰 두었다
/// (`Tests/LynxWebGPUTests/RenderPipelineTests.swift` 참고).
enum DemoScene: String, CaseIterable {
    case triangle
    case cube
    case particles
    case texture
    case blending
    case readback
    case constants
    case msl

    var title: String {
        switch self {
        case .triangle: return "회전 삼각형"
        case .cube: return "3D 큐브"
        case .particles: return "입자 4096개"
        case .texture: return "텍스처 · 샘플러"
        case .blending: return "알파 블렌딩"
        case .readback: return "컴퓨트 · 리드백"
        case .constants: return "파이프라인 상수"
        case .msl: return "MSL 탈출구"
        }
    }

    /// 이 씬이 실제로 밟는 WebGPU 경로.
    var subtitle: String {
        switch self {
        case .triangle: return "정점 버퍼 + 유니폼 + 매 프레임 writeBuffer"
        case .cube: return "인덱스 드로우 + 깊이 테스트 + 백페이스 컬링 + MVP"
        case .particles: return "컴퓨트 셰이더 + 스토리지 버퍼 + 인스턴싱 + 가산 블렌딩"
        case .texture: return "createTexture + writeTexture + 샘플러 + textureSample"
        case .blending: return "src-alpha 블렌딩 + 겹치는 반투명 도형"
        case .readback: return "컴퓨트 결과를 mapAsync로 CPU에서 읽어 표시"
        case .constants: return "같은 셰이더를 override 값만 바꿔 여러 파이프라인으로"
        case .msl: return "language: 'msl' — 트랜스파일러 우회 + 명시적 레이아웃"
        }
    }
}
