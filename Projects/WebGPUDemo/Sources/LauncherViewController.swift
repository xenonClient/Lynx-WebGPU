import UIKit
import Metal

/// The demo scene list. Tapping a row shows that scene's Lynx bundle.
///
/// Each scene **creates a fresh LynxView and WebGPU runtime and releases them on leaving**, so moving
/// between the list and a scene exercises the creation and teardown paths too.
final class LauncherViewController: UITableViewController {
    private let scenes = DemoScene.allCases

    init() {
        super.init(style: .insetGrouped)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    override func viewDidLoad() {
        super.viewDidLoad()
        title = "WebGPU on Lynx"
        navigationItem.largeTitleDisplayMode = .always
        navigationController?.navigationBar.prefersLargeTitles = true
        tableView.register(UITableViewCell.self, forCellReuseIdentifier: "scene")
    }

    override func numberOfSections(in tableView: UITableView) -> Int { 1 }

    override func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        scenes.count
    }

    override func tableView(_ tableView: UITableView, titleForFooterInSection section: Int) -> String? {
        let device = MTLCreateSystemDefaultDevice()
        return "\(device?.name ?? "no Metal device") · unified memory \(device?.hasUnifiedMemory == true ? "yes" : "no")"
    }

    override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "scene", for: indexPath)
        let scene = scenes[indexPath.row]

        var content = cell.defaultContentConfiguration()
        content.text = scene.title
        content.secondaryText = scene.subtitle
        content.secondaryTextProperties.numberOfLines = 0
        content.secondaryTextProperties.color = .secondaryLabel
        content.secondaryTextProperties.font = .preferredFont(forTextStyle: .caption1)
        cell.contentConfiguration = content
        cell.accessoryType = .disclosureIndicator
        return cell
    }

    override func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        tableView.deselectRow(at: indexPath, animated: true)
        let scene = scenes[indexPath.row]
        let controller = DemoViewController(scene: scene)

        // Scenes whose gestures must not overlap go modal full-screen (see `DemoScene.coversFullScreen`).
        guard scene.coversFullScreen else {
            navigationController?.pushViewController(controller, animated: true)
            return
        }
        controller.modalPresentationStyle = .fullScreen
        present(controller, animated: true)
    }
}
