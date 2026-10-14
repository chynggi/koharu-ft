# SDD workspace, task brief, and review package helpers for this repo.
#
# The bundled SDD shell scripts require bash, and this machine has no WSL
# distribution, so this script reproduces their behaviour. The formats match
# the originals: briefs are the verbatim task text, review packages carry the
# commit list, stat summary, and a -U10 diff.
#
# Usage:
#   .\scripts\sdd.ps1 workspace PLAN_FILE
#   .\scripts\sdd.ps1 brief PLAN_FILE TASK_NUMBER
#   .\scripts\sdd.ps1 package PLAN_FILE BASE HEAD

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet('workspace', 'brief', 'package')]
    [string]$Command,

    [Parameter(Mandatory = $true, Position = 1)]
    [string]$Plan,

    [Parameter(Position = 2)]
    [string]$A,

    [Parameter(Position = 3)]
    [string]$B
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $Plan -PathType Leaf)) {
    Write-Error "no such plan file: $Plan"
    exit 2
}

$root = (& git rev-parse --show-toplevel).Trim()
$slug = [System.IO.Path]::GetFileNameWithoutExtension($Plan)
if ([string]::IsNullOrWhiteSpace($slug)) {
    Write-Error "cannot derive a workspace name from: $Plan"
    exit 2
}

$planDir = (& git -C (Split-Path -Parent (Resolve-Path -LiteralPath $Plan)) rev-parse --show-toplevel 2>$null)
$planResolved = (Resolve-Path -LiteralPath $Plan).Path
$rootFull = (Resolve-Path -LiteralPath $root).Path
$planId = $planResolved
if ($planResolved.StartsWith($rootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
    $planId = $planResolved.Substring($rootFull.Length).TrimStart('\', '/')
}

$base = Join-Path $rootFull '.superpowers\sdd'
New-Item -ItemType Directory -Force -Path $base | Out-Null
# A self-ignoring .gitignore keeps every plan's workspace out of git status.
Set-Content -LiteralPath (Join-Path $base '.gitignore') -Value '*' -NoNewline

function Test-Owns([string]$Dir) {
    $marker = Join-Path $Dir 'plan-path'
    if (Test-Path -LiteralPath $marker) {
        return ((Get-Content -LiteralPath $marker -Raw).Trim() -eq $planId)
    }
    New-Item -ItemType Directory -Force -Path $Dir | Out-Null
    Set-Content -LiteralPath $marker -Value $planId
    return $true
}

function Get-Workspace {
    $dir = Join-Path $base $slug
    if (Test-Owns $dir) { return $dir }
    $parent = Split-Path -Leaf (Split-Path -Parent $planResolved)
    $dir = Join-Path $base "$slug-$parent"
    if (Test-Owns $dir) { return $dir }
    $n = 2
    while ($true) {
        $candidate = Join-Path $base "$slug-$parent-$n"
        if (Test-Owns $candidate) { return $candidate }
        $n++
    }
}

$workspace = Get-Workspace

switch ($Command) {
    'workspace' {
        Write-Output $workspace
    }

    'brief' {
        if ([string]::IsNullOrWhiteSpace($A)) {
            Write-Error 'brief requires TASK_NUMBER'
            exit 2
        }
        $n = $A
        $out = Join-Path $workspace "task-$n-brief.md"

        # A task runs until the next '### Task' heading, but code fences may
        # contain lines that look like headings, so track fence state.
        #
        # The walk resumes the fence state from the whole plan up to this
        # task's heading rather than assuming the task starts outside a
        # fence. A marker before the heading is a plan defect worth knowing
        # about, because it makes every later brief swallow its neighbours.
        $lines = Get-Content -LiteralPath $Plan
        $headingIndex = -1
        $fenceBefore = $false
        $tempFence = $false
        for ($i = 0; $i -lt $lines.Count; $i++) {
            if ($lines[$i] -match '^```') { $tempFence = -not $tempFence }
            if ($lines[$i] -match '^#+[ \t]+Task[ \t]+[0-9]+' -and $lines[$i] -match "^#+[ \t]+Task[ \t]+$n([^0-9]|$)") {
                $headingIndex = $i
                $fenceBefore = $tempFence
                break
            }
        }
        if ($headingIndex -lt 0) {
            Write-Error "task $n not found in $Plan (no heading matching 'Task $n')"
            exit 3
        }
        if ($fenceBefore) {
            Write-Warning "plan has an unbalanced code fence before Task $n; later briefs will over-capture"
        }

        $collected = New-Object System.Collections.Generic.List[string]
        $inFence = $fenceBefore
        $inTask = $true
        for ($i = $headingIndex; $i -lt $lines.Count; $i++) {
            $line = $lines[$i]
            if ($i -gt $headingIndex -and -not $inFence -and $line -match '^#+[ \t]+Task[ \t]+[0-9]+') {
                break
            }
            if ($line -match '^```') { $inFence = -not $inFence }
            $collected.Add($line)
        }
        if ($inFence) {
            Write-Warning "task $n brief ends inside a code fence; its section may be truncated"
        }

        if ($collected.Count -eq 0) {
            Write-Error "task $n not found in $Plan (no heading matching 'Task $n')"
            exit 3
        }
        Set-Content -LiteralPath $out -Value $collected
        Write-Output "wrote ${out}: $($collected.Count) lines"
    }

    'package' {
        if ([string]::IsNullOrWhiteSpace($A) -or [string]::IsNullOrWhiteSpace($B)) {
            Write-Error 'package requires BASE and HEAD'
            exit 2
        }
        $baseRev = $A
        $headRev = $B
        & git rev-parse --verify --quiet $baseRev *> $null
        if ($LASTEXITCODE -ne 0) { Write-Error "bad BASE: $baseRev"; exit 2 }
        & git rev-parse --verify --quiet $headRev *> $null
        if ($LASTEXITCODE -ne 0) { Write-Error "bad HEAD: $headRev"; exit 2 }

        & git merge-base --is-ancestor $baseRev $headRev
        if ($LASTEXITCODE -ne 0) {
            Write-Error "HEAD is not a descendant of BASE: ${baseRev}..${headRev}"
            exit 3
        }
        $count = (& git rev-list --count "${baseRev}..${headRev}").Trim()
        if ([int]$count -eq 0) {
            Write-Error "empty commit range: ${baseRev}..${headRev}"
            exit 3
        }

        $shortBase = (& git rev-parse --short $baseRev).Trim()
        $shortHead = (& git rev-parse --short $headRev).Trim()
        $out = Join-Path $workspace "review-$shortBase..$shortHead.diff"

        $body = New-Object System.Collections.Generic.List[string]
        $body.Add("# Review package: ${baseRev}..${headRev}")
        $body.Add('')
        $body.Add('## Commits')
        $body.AddRange([string[]](& git log --oneline "${baseRev}..${headRev}"))
        $body.Add('')
        $body.Add('## Files changed')
        $body.AddRange([string[]](& git diff --stat "${baseRev}..${headRev}"))
        $body.Add('')
        $body.Add('## Diff')
        $body.AddRange([string[]](& git diff -U10 "${baseRev}..${headRev}"))
        Set-Content -LiteralPath $out -Value $body
        $bytes = (Get-Item -LiteralPath $out).Length
        Write-Output "wrote ${out}: $count commit(s), $bytes bytes"
    }
}
