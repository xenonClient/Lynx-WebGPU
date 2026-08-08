import UIKit
import Lynx
import LynxWebGPUBridge

// WebGPUDemo의 DemoViewController에서 **딱 한 곳**만 다르다 — 런타임 주입 줄.
// `import LynxWebGPU`(Metal 엔진)가 없다는 것이 이 파일의 요점이다: 브리지도, 씬 목록도,
// `.lynx.bundle`(JS)도 데모의 것을 그대로 쓰면서 백엔드만 Dawn이다.

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

/// 데모 씬 하나를 Dawn 런타임으로 띄우는 호스트 화면.
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
            // 런타임(백엔드)은 **앱이 고른다** — 여기서는 Dawn이다. 이 한 줄이 전부다.
            host = LynxWebGPUHost(runtime: try DawnWebGPURuntime())
        } catch {
            showError("Dawn 런타임을 만들 수 없다: \(error)")
            return
        }
        guard let host else { return }

        guard let templatePath = Bundle.main.path(forResource: scene.rawValue, ofType: "lynx.bundle") else {
            showError("\(scene.rawValue).lynx.bundle 이 앱 번들에 없다.")
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
        host.attach(to: lynxView)

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

    /// 자동화 하네스 — 데모와 같은 런치 인자 (`-cardTilt`, `-altMode`).
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
        host?.detach()
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
