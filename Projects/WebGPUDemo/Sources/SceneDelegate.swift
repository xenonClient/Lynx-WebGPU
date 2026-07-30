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
        let requested = UserDefaults.standard.string(forKey: "demo").flatMap(DemoScene.init(rawValue:))
        if let requested, !requested.coversFullScreen {
            navigation.pushViewController(DemoViewController(scene: requested), animated: false)
        }

        let window = UIWindow(windowScene: windowScene)
        window.rootViewController = navigation
        window.makeKeyAndVisible()
        self.window = window

        // 모달로 띄우는 씬은 창이 올라온 뒤에 present한다 (목록이 뒤에 남아 뒤로가기가 자연스럽다).
        if let requested, requested.coversFullScreen {
            let controller = DemoViewController(scene: requested)
            controller.modalPresentationStyle = .fullScreen
            navigation.present(controller, animated: false)
        }
    }
}
