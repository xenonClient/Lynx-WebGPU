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

        // Automation harness: passing `-demo <name>` skips the list and enters that scene directly
        // (for screenshot capture — `docs/TESTING.md` §8).
        //   xcrun simctl launch <device> org.lynxwebgpu.demo -demo cube
        let requested = UserDefaults.standard.string(forKey: "demo").flatMap(DemoScene.init(rawValue:))
        if let requested, !requested.coversFullScreen {
            navigation.pushViewController(DemoViewController(scene: requested), animated: false)
        }

        let window = UIWindow(windowScene: windowScene)
        window.rootViewController = navigation
        window.makeKeyAndVisible()
        self.window = window

        // A scene shown modally is presented after the window is up (the list stays behind so Back feels natural).
        if let requested, requested.coversFullScreen {
            let controller = DemoViewController(scene: requested)
            controller.modalPresentationStyle = .fullScreen
            navigation.present(controller, animated: false)
        }
    }
}
