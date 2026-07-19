import AVFAudio
import Foundation
import Speech

private struct Transcript: Encodable {
    let text: String
}

private struct Arguments {
    let audioFiles: [URL]
    let outputDirectory: URL

    static func parse(_ arguments: [String]) throws -> Arguments {
        var explicitFiles: [URL] = []
        var audioFolder: URL?
        var outputDirectory: URL?
        var index = 0

        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "--audio":
                index += 1
                guard index < arguments.count else {
                    throw CLIError.usage("Missing file after --audio")
                }
                explicitFiles.append(URL(fileURLWithPath: arguments[index]))
            case "--audio-folder":
                index += 1
                guard index < arguments.count else {
                    throw CLIError.usage("Missing directory after --audio-folder")
                }
                guard audioFolder == nil else {
                    throw CLIError.usage("--audio-folder may only be specified once")
                }
                audioFolder = URL(fileURLWithPath: arguments[index], isDirectory: true)
            case "--out":
                index += 1
                guard index < arguments.count else {
                    throw CLIError.usage("Missing directory after --out")
                }
                guard outputDirectory == nil else {
                    throw CLIError.usage("--out may only be specified once")
                }
                outputDirectory = URL(fileURLWithPath: arguments[index], isDirectory: true)
            case "--help", "-h":
                throw CLIError.usage(nil)
            default:
                throw CLIError.usage("Unknown argument: \(argument)")
            }
            index += 1
        }

        guard let outputDirectory else {
            throw CLIError.usage("--out is required")
        }
        guard audioFolder == nil || explicitFiles.isEmpty else {
            throw CLIError.usage("Use either --audio-folder or --audio, not both")
        }

        let audioFiles: [URL]
        if let audioFolder {
            let contents = try FileManager.default.contentsOfDirectory(
                at: audioFolder,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )
            audioFiles = contents
                .filter { ["flac", "wav"].contains($0.pathExtension.lowercased()) }
                .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
        } else {
            guard !explicitFiles.isEmpty else {
                throw CLIError.usage("At least one --audio or --audio-folder is required")
            }
            audioFiles = explicitFiles
        }

        return Arguments(audioFiles: audioFiles, outputDirectory: outputDirectory)
    }
}

private enum CLIError: LocalizedError {
    case usage(String?)
    case message(String)

    var errorDescription: String? {
        switch self {
        case .usage(let message):
            let usage = """
            Usage:
              sacli --audio-folder <DIR> --out <OUTDIR>
              sacli --audio <FILE> [--audio <FILE> ...] --out <OUTDIR>
            """
            return [message, usage].compactMap { $0 }.joined(separator: "\n")
        case .message(let message):
            return message
        }
    }
}

@main
private struct SpeechAnalyzerCLI {
    private static let locale = Locale(identifier: "en-US")

    static func main() async {
        do {
            let arguments = try Arguments.parse(Array(CommandLine.arguments.dropFirst()))
            try FileManager.default.createDirectory(
                at: arguments.outputDirectory,
                withIntermediateDirectories: true
            )

            let selectedLocale = try await prepareTranscriptionAssets()
            for audioURL in arguments.audioFiles {
                let text = try await transcribe(audioURL, locale: selectedLocale)
                try writeTranscript(text, for: audioURL, to: arguments.outputDirectory)

                let summary = String(text.prefix(60))
                    .replacingOccurrences(of: "\t", with: " ")
                    .replacingOccurrences(of: "\n", with: " ")
                    .replacingOccurrences(of: "\r", with: " ")
                let name = audioURL.deletingPathExtension().lastPathComponent
                print("\(name)\t\(summary)")
            }
        } catch {
            FileHandle.standardError.write(Data("sacli: \(error.localizedDescription)\n".utf8))
            Foundation.exit(EXIT_FAILURE)
        }
    }

    private static func prepareTranscriptionAssets() async throws -> Locale {
        guard SpeechTranscriber.isAvailable else {
            throw CLIError.message("SpeechTranscriber is unavailable on this Mac")
        }
        let selectedLocale = locale
        let assetProbe = SpeechTranscriber(locale: selectedLocale, preset: .transcription)
        let modules: [any SpeechModule] = [assetProbe]
        let installed = await SpeechTranscriber.installedLocales.contains {
            $0.identifier == selectedLocale.identifier
        }

        if !installed {
            let request: AssetInstallationRequest?
            do {
                request = try await AssetInventory.assetInstallationRequest(supporting: modules)
            } catch {
                throw CLIError.message("Failed to create the en-US speech asset installation request: \(error)")
            }
            guard let request else {
                let status = await AssetInventory.status(forModules: modules)
                guard status == .installed else {
                    throw CLIError.message("No en-US speech asset installation request is available (status: \(status))")
                }
                return selectedLocale
            }
            do {
                try await request.downloadAndInstall()
            } catch {
                throw CLIError.message("Failed to download and install the en-US speech asset: \(error)")
            }
        }

        let finalStatus = await AssetInventory.status(forModules: modules)
        guard finalStatus == .installed else {
            throw CLIError.message("The en-US speech asset is not installed (status: \(finalStatus))")
        }

        do {
            _ = try await AssetInventory.reserve(locale: selectedLocale)
        } catch {
            throw CLIError.message("Failed to reserve the installed en-US speech asset: \(error)")
        }
        return selectedLocale
    }

    private static func transcribe(_ audioURL: URL, locale: Locale) async throws -> String {
        let audioFile: AVAudioFile
        do {
            audioFile = try AVAudioFile(forReading: audioURL)
        } catch {
            throw CLIError.message("Failed to open \(audioURL.path): \(error)")
        }
        let transcriber = SpeechTranscriber(locale: locale, preset: .transcription)
        let analyzer = SpeechAnalyzer(modules: [transcriber])

        let analysisTask = Task {
            try await analyzer.start(inputAudioFile: audioFile, finishAfterFile: true)
        }

        do {
            var segments: [String] = []
            for try await result in transcriber.results {
                if result.isFinal {
                    segments.append(String(result.text.characters))
                }
            }
            try await analysisTask.value
            // Join final segments with a space. On short clips there is usually one
            // final segment, so this matches earlier single-segment runs; on longer
            // out-of-domain clips there can be several, and joining with "" would
            // merge the last word of one segment into the first word of the next and
            // inflate WER. The scorer normalizes whitespace, so an extra space is
            // harmless while a missing one is not.
            return segments.joined(separator: " ")
        } catch {
            analysisTask.cancel()
            throw CLIError.message("Failed to transcribe \(audioURL.path): \(error)")
        }
    }

    private static func writeTranscript(_ text: String, for audioURL: URL, to outputDirectory: URL) throws {
        let baseName = audioURL.deletingPathExtension().lastPathComponent
        let outputURL = outputDirectory.appendingPathComponent(baseName).appendingPathExtension("json")
        let data = try JSONEncoder().encode(Transcript(text: text))
        try data.write(to: outputURL, options: .atomic)
    }
}
