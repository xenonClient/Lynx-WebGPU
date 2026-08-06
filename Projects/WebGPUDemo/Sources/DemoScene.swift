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
    case dynamic
    case blending
    case stencil
    case gpudriven
    case bundle
    case query
    case readback
    case constants
    case msl
    case interactive
    case wgsl
    case arraybuffer
    case bench
    case hdr
    case scrollpass
    case three
    case spec
    case images
    case contracts
    case threelab

    /// 화면 전체를 덮어 띄울 씬 — 네비게이션 컨트롤러가 아니라 **모달로 표시**한다.
    ///
    /// 밀어서 뒤로 가기(`interactivePopGestureRecognizer`)가 화면 왼쪽에서 시작하는 드래그를
    /// 가로채기 때문에, 캔버스를 잡고 끄는 씬은 그 제스처와 경쟁한다. 모달 전체 화면에는
    /// 그런 제스처가 없어서 터치가 온전히 Lynx로 간다 (뒤로가기는 직접 붙인 버튼으로).
    var coversFullScreen: Bool { self == .interactive || self == .hdr || self == .scrollpass }

    var title: String {
        switch self {
        case .triangle: return "회전 삼각형"
        case .cube: return "3D 큐브"
        case .particles: return "입자 4096개"
        case .texture: return "텍스처 · 샘플러"
        case .dynamic: return "동적 텍스처"
        case .blending: return "알파 블렌딩"
        case .stencil: return "스텐실 마스크"
        case .gpudriven: return "GPU-driven 렌더링"
        case .bundle: return "렌더 번들"
        case .query: return "쿼리 · 오류 스코프"
        case .readback: return "컴퓨트 · 리드백"
        case .constants: return "파이프라인 상수"
        case .msl: return "MSL 탈출구"
        case .interactive: return "홀로그래픽 카드"
        case .wgsl: return "WGSL 호환성"
        case .arraybuffer: return "바이너리 브리징"
        case .bench: return "브리지 비용 측정"
        case .hdr: return "HDR 게인맵 재구성"
        case .scrollpass: return "스크롤 통과"
        case .three: return "three.js 렌더러"
        case .spec: return "명세 표면 점검"
        case .images: return "이미지 · 압축 텍스처"
        case .contracts: return "계약 점검"
        case .threelab: return "three.js 고난도 조합"
        }
    }

    /// 이 씬이 실제로 밟는 WebGPU 경로.
    var subtitle: String {
        switch self {
        case .triangle: return "정점 버퍼 + 유니폼 + 매 프레임 writeBuffer"
        case .cube: return "인덱스 드로우 + 깊이 테스트 + 백페이스 컬링 + MVP"
        case .particles: return "컴퓨트 셰이더 + 스토리지 버퍼 + 인스턴싱 + 가산 블렌딩"
        case .texture: return "createTexture + writeTexture + 샘플러 + textureSample"
        case .dynamic: return "CPU 플라스마를 매 프레임 writeTexture로 — 큐 순서 업로드 검증"
        case .blending: return "미리 곱해진 알파 합성 + 겹치는 반투명 도형"
        case .stencil: return "stencil8 단독 포맷 — 같은 삼각형 3번, 갈리는 이유는 스텐실뿐"
        case .gpudriven: return "컴퓨트가 개수를 정하고 간접 디스패치·드로우가 그 버퍼를 읽는다"
        case .bundle: return "드로우 120개를 한 번만 기록 — 프레임당 커맨드 수를 HUD에 표시"
        case .query: return "occlusion 샘플 수 + 타임스탬프 + pushErrorScope 대비 실험"
        case .readback: return "컴퓨트 결과를 mapAsync로 CPU에서 읽어 표시"
        case .constants: return "같은 셰이더를 override 값만 바꿔 여러 파이프라인으로"
        case .msl: return "language: 'msl' — 트랜스파일러 우회 + 명시적 레이아웃"
        case .interactive: return "잡고 기울이면 포일이 흐른다 — Lynx 터치 → 3D 자세 → 셰이더"
        case .wgsl: return "arrayLength + 외부 텍스처 + 타입 없는 상수식"
        case .arraybuffer: return "ArrayBuffer로 양방향 왕복 — Lynx 값 변환 검증"
        case .bench: return "base64 문자열 vs ArrayBuffer — 인코딩·제출 비용 비교"
        case .hdr: return "loadAsset + 게인맵 컴퓨트 → rgba16float 중간 텍스처 → 톤매핑"
        case .scrollpass: return "<scroll-view> 위 캔버스 — passthrough-touches 히트 테스트 검증"
        case .three: return "체크리스트 16종 + ASTC 압축 텍스처를 입은 회전 큐브·디코딩한 PNG 배경"
        case .spec: return "디버그 마커·진단·부분 매핑 등 명세 표면 14종을 값으로 확인"
        case .images: return "ASTC·BC 블록과 PNG 디코딩을 되읽은 픽셀 색으로 확인"
        case .contracts: return "버퍼 복사 기본값·범위, 정수 vec3 배치, 번들 격리, 포맷 역방향"
        case .threelab: return "TSL 절차적 머티리얼 · 그림자 맵 · 컴퓨트 파티클 · 인스턴싱 · bloom"
        }
    }
}
