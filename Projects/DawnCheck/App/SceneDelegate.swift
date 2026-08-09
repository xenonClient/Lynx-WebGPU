import UIKit

final class SceneDelegate: UIResponder, UIWindowSceneDelegate {
    var window: UIWindow?

    func scene(
        _ scene: UIScene,
        willConnectTo session: UISceneSession,
        options connectionOptions: UIScene.ConnectionOptions
    ) {
        guard let windowScene = scene as? UIWindowScene else { return }

        let navigation = UINavigationController(rootViewController: LauncherViewController())

        // The automation harness — the same launch arguments as the demo (`-demo <name>`), the same scene list.
        //   xcrun simctl launch <device> org.lynxwebgpu.dawndemo -demo triangle
        let requested = UserDefaults.standard.string(forKey: "demo").flatMap(DemoScene.init(rawValue:))
        if let requested, !requested.coversFullScreen {
            navigation.pushViewController(DemoViewController(scene: requested), animated: false)
        }

        let window = UIWindow(windowScene: windowScene)
        window.rootViewController = navigation
        window.makeKeyAndVisible()
        self.window = window

        if let requested, requested.coversFullScreen {
            let controller = DemoViewController(scene: requested)
            controller.modalPresentationStyle = .fullScreen
            navigation.present(controller, animated: false)
        }
    }
}
