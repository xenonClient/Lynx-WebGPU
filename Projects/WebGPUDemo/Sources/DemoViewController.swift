import UIKit
import Lynx
import LynxWebGPU
import LynxWebGPUBridge

/// `.lynx.bundle` 파일을 앱 번들에서 읽어 LynxView에 넘긴다.
final class BundleTemplateProvider: NSObject, LynxTemplateProvider {
    func loadTemplate(withUrl url: String!, onComplete callback: LynxTemplateLoadBlock!) {
        do {
            callback(try Data(contentsOf: URL(fileURLWithPath: url)), nil)
        } catch {
            callback(nil, error)
        }
    }
}

/// 데모 씬 하나를 띄우는 호스트 화면.
///
/// `docs/LYNX-INTEGRATION.md` §2의 3단계가 그대로 들어 있다:
/// `LynxWebGPUHost()` → `LynxWebGPU.register(in:host:)` → `host.attach(to:)`
final class DemoViewController: UIViewController {
    private let scene: DemoScene
    private var host: LynxWebGPUHost?
    private var lynxView: LynxView?

    init(scene: DemoScene) {
        self.scene = scene
        super.init(nibName: nil, bundle: nil)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    override func viewDidLoad() {
        super.viewDidLoad()
        title = scene.title
        navigationItem.largeTitleDisplayMode = .never
        view.backgroundColor = UIColor(red: 0.05, green: 0.06, blue: 0.09, alpha: 1)

        do {
            host = try LynxWebGPUHost()
        } catch {
            showError("WebGPU 런타임을 만들 수 없다: \(error)")
            return
        }
        guard let host else { return }

        guard let templatePath = Bundle.main.path(forResource: scene.rawValue, ofType: "lynx.bundle") else {
            showError(
                "\(scene.rawValue).lynx.bundle 이 앱 번들에 없다.\n"
                    + "Projects/WebGPUDemo/DemoSrc 에서 `mise exec -- npm run build` 후 "
                    + "산출물을 Resources/ 로 복사할 것."
            )
            return
        }

        let screenSize = UIScreen.main.bounds.size
        let lynxView = LynxView { builder in
            let config = LynxConfig(provider: BundleTemplateProvider())
            LynxWebGPU.register(in: config, host: host)   // NativeModules.WebGPU + <webgpu-canvas>
            builder.config = config
            builder.screenSize = screenSize
            builder.fontScale = 1.0
        }
        host.attach(to: lynxView)                        // 전역 이벤트/캔버스 연결

        lynxView.preferredLayoutWidth = screenSize.width
        lynxView.preferredLayoutHeight = screenSize.height
        lynxView.layoutWidthMode = .exact
        lynxView.layoutHeightMode = .exact
        lynxView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(lynxView)
        NSLayoutConstraint.activate([
            lynxView.topAnchor.constraint(equalTo: view.topAnchor),
            lynxView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            lynxView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            lynxView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])
        self.lynxView = lynxView

        // 모달로 올라온 씬은 네비게이션 바가 없으니 닫기 버튼을 직접 얹는다.
        // LynxView보다 **나중에** 붙여야 UIKit 히트 테스트에서 위로 온다.
        if scene.coversFullScreen { addDismissButton() }

        lynxView.loadTemplate(fromURL: templatePath, initData: initialData)
    }

    private func addDismissButton() {
        var configuration = UIButton.Configuration.filled()
        configuration.image = UIImage(systemName: "xmark")
        configuration.baseBackgroundColor = UIColor(white: 1, alpha: 0.14)
        configuration.baseForegroundColor = .white
        configuration.cornerStyle = .capsule
        configuration.contentInsets = NSDirectionalEdgeInsets(top: 12, leading: 12, bottom: 12, trailing: 12)

        let button = UIButton(configuration: configuration)
        button.accessibilityLabel = "닫기"
        button.addTarget(self, action: #selector(dismissScene), for: .touchUpInside)
        button.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(button)
        NSLayoutConstraint.activate([
            button.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 8),
            button.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 16),
        ])
    }

    @objc private func dismissScene() {
        if presentingViewController != nil {
            dismiss(animated: true)
        } else {
            navigationController?.popViewController(animated: true)
        }
    }

    /// 자동화 하네스 — 시뮬레이터에는 터치를 주입할 방법이 없어, **손으로 눌러야만 보이는 상태를
    /// 회귀 확인하려면** 런치 인자로 고정하는 이 경로가 필요하다.
    ///
    ///   xcrun simctl launch <device> org.lynxwebgpu.demo -demo interactive -cardTilt 0.45
    ///   xcrun simctl launch <device> org.lynxwebgpu.demo -demo bundle -altMode 1
    ///
    /// `-altMode 1`은 토글이 있는 씬(`stencil`·`gpudriven`·`bundle`)을 **기본이 아닌 쪽**으로
    /// 시작시킨다 — 버튼을 누른 화면을 캡처하려고 둔 것이다.
    private var initialData: LynxTemplateData? {
        var data: [String: Any] = [:]
        let tilt = UserDefaults.standard.double(forKey: "cardTilt")
        if tilt != 0 { data["forceTilt"] = tilt }
        if UserDefaults.standard.bool(forKey: "altMode") { data["altMode"] = true }
        return data.isEmpty ? nil : LynxTemplateData(dictionary: data)
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        guard let lynxView, view.bounds.width > 0, view.bounds.height > 0 else { return }
        guard lynxView.preferredLayoutWidth != view.bounds.width
            || lynxView.preferredLayoutHeight != view.bounds.height else { return }
        lynxView.preferredLayoutWidth = view.bounds.width
        lynxView.preferredLayoutHeight = view.bounds.height
        lynxView.triggerLayout()
    }

    deinit {
        host?.detach()   // 디스플레이 링크 정지 + GPU 객체 해제
    }

    private func showError(_ message: String) {
        let label = UILabel()
        label.text = message
        label.textColor = .white
        label.numberOfLines = 0
        label.font = .monospacedSystemFont(ofSize: 13, weight: .regular)
        label.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(label)
        NSLayoutConstraint.activate([
            label.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            label.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 24),
            label.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -24),
        ])
    }
}
