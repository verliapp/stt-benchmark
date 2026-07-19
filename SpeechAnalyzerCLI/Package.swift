// swift-tools-version: 6.0
import PackageDescription
let package = Package(
    name: "sacli",
    platforms: [.macOS("26.0")],
    targets: [ .executableTarget(name: "sacli", path: "Sources/sacli") ]
)
