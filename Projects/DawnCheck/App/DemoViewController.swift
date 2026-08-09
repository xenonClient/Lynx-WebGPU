import UIKit
import Lynx
import LynxWebGPUBridge

// It differs from WebGPUDemo's DemoViewController in **exactly one place** — the runtime injection line.
// The point of this file is the absence of `import LynxWebGPU` (the Metal engine): the bridge, the scene
// list and the `.lynx.bundle` (JS) are all the demo's, unchanged — only the backend is Dawn.

/// Reads a `.lynx.bundle` file out of the app bundle and hands it to LynxView.
final class BundleTemplateProvider: NSObject, LynxTemplateProvider {
    func loadTemplate(withUrl url: String!, onComplete callback: LynxTemplateLoadBlock!) {
        do {
            callback(try Data(contentsOf: URL(fileURLWithPath: url)), nil)
        } catch {
            callback(nil, error)
        }
    }
}

/// The host screen that brings up one demo scene on the Dawn runtime.
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
            // **The app picks the runtime (backend)** — here it is Dawn. This one line is all of it.
            host = LynxWebGPUHost(runtime: try DawnWebGPURuntime())
        } catch {
            showError("could not create the Dawn runtime: \(error)")
            return
        }
        guard let host else { return }

        guard let templatePath = Bundle.main.path(forResource: scene.rawValue, ofType: "lynx.bundle") else {
            showError("\(scene.rawValue).lynx.bundle is not in the app bundle.")
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
        button.accessibilityLabel = "Close"
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

    /// The automation harness — the same launch arguments as the demo (`-cardTilt`, `-altMode`).
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
