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

        // 자동화 하네스: `-demo <name>` 을 주면 목록을 건너뛰고 그 씬으로 바로 들어간다
        // (스크린샷 캡처용 — `docs/TESTING.md` §8).
        //   xcrun simctl launch <device> org.lynxwebgpu.demo -demo cube
        if let requested = UserDefaults.standard.string(forKey: "demo"),
           let target = DemoScene(rawValue: requested) {
            navigation.pushViewController(DemoViewController(scene: target), animated: false)
        }

        let window = UIWindow(windowScene: windowScene)
        window.rootViewController = navigation
        window.makeKeyAndVisible()
        self.window = window
    }
}
