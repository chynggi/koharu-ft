#!/usr/bin/env bun

// The installed tauri-cli must be built from the same tauri revision the app links.
// A stale CLI still builds and bundles successfully, but detects the webview runtime
// with the rules of its own revision, so it can silently ship an installer without
// the CEF distribution the app needs to start.

import { readFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import path from 'node:path'

const root = path.resolve(__dirname, '..')
const cargoHome = process.env.CARGO_HOME ?? path.join(homedir(), '.cargo')

async function main() {
  const cargoToml = await readFile(path.join(root, 'Cargo.toml'), 'utf8')
  const expected = cargoToml.match(/^tauri\s*=\s*\{[^\n]*?\brev\s*=\s*"([0-9a-f]{7,40})"/m)?.[1]

  if (!expected) {
    throw new Error('could not read the pinned tauri revision from Cargo.toml')
  }

  const cratesTomlPath = path.join(cargoHome, '.crates.toml')
  let cratesToml: string
  try {
    cratesToml = await readFile(cratesTomlPath, 'utf8')
  } catch {
    throw new Error(`tauri-cli is not installed (no ${cratesTomlPath})\n${installHint(expected)}`)
  }

  const entry = cratesToml.match(/^"tauri-cli\s[^"]*"/m)?.[0]

  if (!entry) {
    throw new Error(`tauri-cli is not installed\n${installHint(expected)}`)
  }

  const installed = entry.match(/\?rev=([0-9a-f]{7,40})/)?.[1]

  if (!installed) {
    throw new Error(
      `installed tauri-cli does not come from the tauri git repository: ${entry}\n${installHint(expected)}`,
    )
  }

  if (installed !== expected) {
    throw new Error(
      `tauri-cli revision mismatch: installed ${installed}, Cargo.toml pins ${expected}\n${installHint(expected)}`,
    )
  }

  console.log(`tauri-cli matches the pinned revision ${expected}`)
}

function installHint(rev: string) {
  return `Run \`bun install\`, or install it directly:\n  cargo install --git https://github.com/tauri-apps/tauri --rev ${rev} --locked tauri-cli`
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error)
  process.exit(1)
})
