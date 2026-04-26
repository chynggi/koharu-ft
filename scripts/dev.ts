// @ts-nocheck
import os from 'node:os'
import path from 'node:path'
import { readdir, access } from 'node:fs/promises'
import { exec as execCallback, spawn } from 'node:child_process'
import { promisify } from 'node:util'
import * as readline from 'node:readline/promises'

const exec = promisify(execCallback)

async function pathExists(target: string) {
  try {
    await access(target)
    return true
  } catch {
    return false
  }
}

async function checkNvcc() {
  try {
    await exec('nvcc --version', { env: process.env })
  } catch {
    throw new Error('nvcc not found')
  }
}

function sortVersionsDesc(versions: string[]) {
  return versions.sort((a, b) => {
    const verA = parseInt(a.replace('v', '').replace('.', ''))
    const verB = parseInt(b.replace('v', '').replace('.', ''))
    return verB - verA
  })
}

async function setupCuda() {
  const cudaPath = process.env.CUDA_PATH
  if (cudaPath) {
    const binPath = path.join(cudaPath, 'bin')
    process.env.PATH = `${binPath}${path.delimiter}${process.env.PATH}`
    return
  }

  const cudaRoot = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA'
  const versions = await readdir(cudaRoot).catch(() => [])

  if (versions.length === 0) {
    throw new Error(
      'NVCC not found. Please install the CUDA Toolkit from https://developer.nvidia.com/cuda-downloads',
    )
  }

  sortVersionsDesc(versions)

  let selectedVersion = versions[0]
  if (versions.length > 1) {
    console.log('\\nMultiple CUDA toolkits detected:')
    for (let i = 0; i < versions.length; i++) {
      console.log(`  ${i + 1}. ${versions[i]}`)
    }
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    })
    const answer = await rl.question(`Select CUDA version (default: 1): `)
    rl.close()

    const index = parseInt(answer.trim(), 10) - 1
    if (!isNaN(index) && index >= 0 && index < versions.length) {
      selectedVersion = versions[index]
    }
    console.log(`Using CUDA Toolkit version: ${selectedVersion}\\n`)
  }

  if (selectedVersion.startsWith('v')) {
    const binPath = path.join(cudaRoot, selectedVersion, 'bin')
    if (await pathExists(binPath)) {
      process.env.PATH = `${binPath}${path.delimiter}${process.env.PATH}`
      process.env.CUDA_PATH = path.join(cudaRoot, selectedVersion)
      return
    }
  }

  throw new Error(
    'NVCC not found. Please install the CUDA Toolkit from https://developer.nvidia.com/cuda-downloads',
  )
}

async function setupCl() {
  const vsRoots = [
    'C:/Program Files/Microsoft Visual Studio',
    'C:/Program Files (x86)/Microsoft Visual Studio',
  ]
  const editions = ['Community', 'Professional', 'Enterprise', 'BuildTools']

  for (const vsRoot of vsRoots) {
    const vsVersions = await readdir(vsRoot).catch(() => [])

    for (const vsVersion of vsVersions) {
      for (const edition of editions) {
        const vcPath = path.join(vsRoot, vsVersion, edition, 'VC/Tools/MSVC')
        if (await pathExists(vcPath)) {
          const msvcVersions = await readdir(vcPath)
          for (const msvcVersion of msvcVersions) {
            const binPath = path.join(vcPath, msvcVersion, 'bin/Hostx64/x64')
            if (await pathExists(binPath)) {
              process.env.PATH = `${binPath}${path.delimiter}${process.env.PATH}`
              return
            }
          }
        }
      }
    }
  }

  throw new Error(
    'cl.exe not found. Please install Visual Studio with C++ build tools from https://visualstudio.microsoft.com/downloads/',
  )
}

async function dev() {
  if (os.type() === 'Windows_NT') {
    await setupCuda().catch(console.error)
    
    // Setup cl.exe path
    await setupCl()
  }

  const args = process.argv.slice(2)
  if (args.length === 0) {
    throw new Error('No command provided')
  }

  const proc = spawn(args[0], args.slice(1), {
    stdio: 'inherit',
    shell: false,
    env: process.env,
  })

  proc.on('error', (err) => {
    throw err
  })

  proc.on('exit', (code) => {
    process.exit(code)
  })
}

dev().catch((err) => {
  process.stderr.write(`Error: ${err.message} \n`)
  process.exit(1)
})
