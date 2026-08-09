import UIKit
import Lynx
import LynxWebGPU
import LynxWebGPUBridge

/// Reads a `.lynx.bundle` file from the app bundle and hands it to the LynxView.
final class BundleTemplateProvider: NSObject, LynxTemplateProvider {
    func loadTemplate(withUrl url: String!, onComplete callback: LynxTemplateLoadBlock!) {
        do {
            callback(try Data(contentsOf: URL(fileURLWithPath: url)), nil)
        } catch {
            callback(nil, error)
        }
    }
}

/// The host screen showing one demo scene.
///
/// It contains the three steps of `docs/LYNX-INTEGRATION.md` §2 verbatim:
/// `LynxWebGPUHost(runtime:)` → `LynxWebGPU.register(in:host:)` → `host.attach(to:)`
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
            // **The app picks** the runtime (backend) — the default is the Metal engine.
            host = LynxWebGPUHost(runtime: try LynxWebGPUContext())
        } catch {
            showError("could not create the WebGPU runtime: \(error)")
            return
        }
        guard let host else { return }

        guard let templatePath = Bundle.main.path(forResource: scene.rawValue, ofType: "lynx.bundle") else {
            showError(
                "\(scene.rawValue).lynx.bundle is not in the app bundle.\n"
                    + "Run `mise exec -- npm run build` in Projects/WebGPUDemo/DemoSrc and "
                    + "copy the output into Resources/."
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
        host.attach(to: lynxView)                        // wires global events and the canvas

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

        // A modally presented scene has no navigation bar, so a close button is placed directly.
        // It must be attached **after** the LynxView to sit above it in UIKit hit testing.
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

    /// Automation harness — the simulator offers no way to inject touches, so **regression-checking a
    /// state only reachable by hand** needs this path, pinned through a launch argument.
    ///
    ///   xcrun simctl launch <device> org.lynxwebgpu.demo -demo interactive -cardTilt 0.45
    ///   xcrun simctl launch <device> org.lynxwebgpu.demo -demo bundle -altMode 1
    ///
    /// `-altMode 1` starts a scene with a toggle (`stencil`, `gpudriven`, `bundle`) on **the non-default**
    /// side — it exists to capture the screen after the button has been pressed.
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
        host?.detach()   // stops the display link and releases GPU objects
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
