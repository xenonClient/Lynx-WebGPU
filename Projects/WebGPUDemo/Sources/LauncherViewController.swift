import UIKit
import Metal

/// 데모 씬 목록. 각 행을 누르면 그 씬의 Lynx 번들을 띄운다.
///
/// 씬마다 **LynxView와 WebGPU 런타임을 새로 만들고 화면을 떠날 때 해제**하므로,
/// 목록 ↔ 씬을 오가는 것만으로 생성/해제 경로까지 함께 확인된다.
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
        return "\(device?.name ?? "Metal 디바이스 없음") · 통합 메모리 \(device?.hasUnifiedMemory == true ? "O" : "X")"
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

        // 제스처가 겹치면 안 되는 씬은 모달 전체 화면으로 (`DemoScene.coversFullScreen` 참고).
        guard scene.coversFullScreen else {
            navigationController?.pushViewController(controller, animated: true)
            return
        }
        controller.modalPresentationStyle = .fullScreen
        present(controller, animated: true)
    }
}
