# syntax=docker/dockerfile:1
#
# Builds koharu (this fork) and packages a .deb, then runs it headlessly
# behind Xvfb — the desktop CEF window is not meant to be viewed remotely;
# access is over HTTP via koharu-rpc (see PORT/KOHARU_API_TOKEN below),
# which the same window also loads locally once it's up.
#
# NOT build-tested in this environment (no local Docker daemon available
# when this was written, and concurrent heavy builds were off-limits here).
# Expect to iterate against real build output on the vast.ai instance —
# see the notes at the bottom for the failure modes most likely to show up
# first (torch-sys/cmake, CEF sandbox-as-root, CEF's own binary download).
FROM vastai/linux-desktop:cuda-13.2-ubuntu24.04-2026-06-16

ENV DEBIAN_FRONTEND=noninteractive

# Tauri/CEF linux bundle deps (matches .github/workflows/release.yml's
# ubuntu step) plus native-FFI build tooling this base image may not carry:
# cmake (koharu-torch-sys), clang/libclang-dev (bindgen, used by every
# *-sys crate), pkg-config, and xvfb for the headless display CEF needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgtk-3-dev \
    build-essential \
    curl \
    wget \
    file \
    git \
    libxdo-dev \
    libssl-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev \
    cmake \
    clang \
    libclang-dev \
    pkg-config \
    xvfb \
    ca-certificates \
    sudo \
    xdg-user-dirs \
    dbus-x11 \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04's apt-provided nodejs is 18.x; Next.js requires >=20.9 (only
# `npm prefix`, used by this repo's build scripts to resolve workspace-root
# paths, and Next's own CLI need real Node — everything else runs on bun).
# curl|bash swallows a failed download silently (apt then falls back to
# Ubuntu's own old/npm-less nodejs, which fails much later in a confusing
# way) — download to a file first so a broken fetch fails loudly here, and
# assert both binaries actually work before moving on.
RUN curl -fsSL https://deb.nodesource.com/setup_22.x -o /tmp/nodesource_setup.sh \
    && bash /tmp/nodesource_setup.sh \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -f /tmp/nodesource_setup.sh \
    && rm -rf /var/lib/apt/lists/* \
    && node --version \
    && npm --version

# Rust toolchain
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal
RUN cargo install tauri-cli --version "^2" --locked

# Bun (this repo's JS package manager/runtime — see beforeBuildCommand in
# crates/koharu/tauri.conf.json)
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH=/root/.bun/bin:$PATH

WORKDIR /koharu
COPY . .

RUN bun install

# The API token is NOT baked into the image: koharu-rpc injects
# `window.__KOHARU_API_TOKEN__` into the served index.html at startup from
# the runtime KOHARU_API_TOKEN environment variable, so the same image can
# be reused with a fresh secret per instance and no secret lands in a layer.
# The frontend now uses a relative `/api/v1` base URL, so it works both from
# the local CEF window (127.0.0.1) and a remote browser (vast.ai port map).
ARG KOHARU_RPC_PORT=47823

# tauri.conf.json has createUpdaterArtifacts=true (for real desktop
# releases), which makes the CLI also try to sign an updater artifact and
# fail without TAURI_SIGNING_PRIVATE_KEY. This deployment doesn't ship
# through the updater, so turn that off for this build only via a config
# override rather than touching the checked-in release config.
#
# Cache mounts speed up iteration: registry across builds always, target/
# too, but --mount cache contents don't survive into the image layer, so
# copy the artifacts this RUN actually needs out to a plain path first.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/koharu/target \
    cargo tauri build --bundles deb --config '{"bundle":{"createUpdaterArtifacts":false}}' \
    && mkdir -p /koharu/dist \
    && cp target/release/bundle/deb/*.deb /koharu/dist/

# Install the produced .deb like a real user would (also surfaces any
# missing runtime dependency the package declares) instead of just running
# the raw target/release binary.
RUN dpkg -i /koharu/dist/*.deb || (apt-get update && apt-get install -f -y --no-install-recommends)

# Chromium/CEF's sandbox refuses to run as root (the same class of problem
# this branch hit on Windows under an elevated shell — see HANDOFF.md), so
# run as an unprivileged user. Note that this is *not* on its own enough to
# make the sandbox work here: the namespace sandbox still needs either
# --cap-add=SYS_ADMIN or a host that permits unprivileged user namespaces,
# and vast.ai templates can express neither. The build therefore turns the
# sandbox off outright — see the note above ENTRYPOINT.
RUN useradd --create-home --shell /bin/bash koharu \
    && usermod -aG video,render koharu 2>/dev/null || true
# Created here, as root, because Xvfb otherwise makes it itself and warns
# "Owner of /tmp/.X11-unix should be set to root" on every start.
RUN mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix
USER koharu
WORKDIR /home/koharu

# dirs::document_dir() (used to locate the project library root, see
# crates/koharu-app/src/commands/project.rs) reads XDG user-dirs config
# that doesn't exist in a bare container — generate it so it resolves to
# ~/Documents instead of failing app setup entirely.
RUN xdg-user-dirs-update

ENV KOHARU_RPC_HOST=0.0.0.0
ENV KOHARU_RPC_PORT=${KOHARU_RPC_PORT}
EXPOSE 47823

# Xvfb gives CEF a display to initialize against without any real desktop/
# VNC infrastructure — nothing needs to actually render for the HTTP UI
# (the point of this branch's work) to work from a remote browser.
# dbus-run-session avoids the repeated "Failed to connect to the bus"
# errors CEF logs (non-fatal on their own, but noisy and occasionally
# needed by Chromium subsystems) since there's no system/session bus
# running in a bare container otherwise.
COPY --chown=koharu:koharu <<'EOF' /home/koharu/entrypoint.sh
#!/bin/bash
set -euo pipefail

display=99

# The base image exports XDG_RUNTIME_DIR for its own user, and the uid we get
# is not necessarily that one — vast.ai reported
# 'XDG_RUNTIME_DIR "/run/user/1001" is owned by uid 1001, not our uid 1002'
# and dbus then refused to set up its transient directory. Point it at
# somewhere this process definitely owns instead of guessing the uid.
XDG_RUNTIME_DIR="$(mktemp -d /tmp/koharu-runtime-XXXXXX)"
chmod 700 "$XDG_RUNTIME_DIR"
export XDG_RUNTIME_DIR

# A restarted container keeps its filesystem, so the previous run's X lock is
# still here while the server that made it is gone. Xvfb then refuses to start
# with "Server is already active for display 99" — which is what turned a
# single startup failure into an endless restart loop. The lock holds the X
# server's pid, so a dead pid means the lock is stale and ours to clear.
lock="/tmp/.X${display}-lock"
start_server=1
if [ -e "$lock" ]; then
  owner="$(tr -dc '0-9' < "$lock" 2>/dev/null || true)"
  if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; then
    start_server=0
  else
    rm -f "$lock" "/tmp/.X11-unix/X${display}"
  fi
fi

if [ "$start_server" = 1 ]; then
  Xvfb ":${display}" -screen 0 1920x1080x24 &
fi
export DISPLAY=":${display}"

# Xvfb is backgrounded, so `set -e` cannot see it fail. Without this wait a
# failure surfaces later and much less legibly, as CEF being unable to open a
# display it was told exists.
for _ in $(seq 100); do
  [ -S "/tmp/.X11-unix/X${display}" ] && break
  sleep 0.1
done
if [ ! -S "/tmp/.X11-unix/X${display}" ]; then
  echo "entrypoint: Xvfb did not come up on ${DISPLAY}" >&2
  exit 1
fi

exec dbus-run-session -- koharu
EOF
RUN sed -i 's/\r$//' /home/koharu/entrypoint.sh && chmod +x /home/koharu/entrypoint.sh

# The base image ships its own ENTRYPOINT (vast.ai's instance-portal/SSH/
# provisioning bootstrap) that parses its own flags rather than `exec "$@"`
# — confirmed by hand: `docker run <base image> bash -c "..."` logged
# "Warning: Unknown flag: bash" / "Warning: Unknown flag: -c" and ran its
# own default routine instead, ignoring the command entirely. A plain CMD
# here would silently never run for the same reason; ENTRYPOINT must be
# reset explicitly to actually replace it.
ENTRYPOINT ["/home/koharu/entrypoint.sh"]

# --- build-verified locally via Docker Desktop; run notes below are from
#     actually starting the resulting image ---
# 1. No `--cap-add=SYS_ADMIN` is needed, because the CEF sandbox is off.
#    That takes *two* changes, and an earlier revision of this note was wrong
#    to claim the first one was enough:
#      a. vendor/tauri-runtime-cef's `default` feature is patched to `[]`,
#         which sets `CefSettings::no_sandbox`.
#      b. the same crate appends `--no-sandbox` to the command line. (a) only
#         reaches `cef::initialize`, which only the browser process runs, so
#         with (a) alone the zygote still tried to create namespaces and the
#         container died at
#         `zygote_host_impl_linux.cc: Check failed: . : Operation not permitted`.
#    vast.ai templates can't express docker capabilities, and the host may
#    also forbid unprivileged user namespaces, so turning the sandbox off is
#    the only viable path on that marketplace.
#    Confirmed fixed on vast.ai: the zygote FATAL is gone from the log.
# 2. `KOHARU_API_TOKEN` is REQUIRED at run time. The image binds
#    KOHARU_RPC_HOST=0.0.0.0, and crates/koharu/src/main.rs refuses to start
#    without a token on a non-loopback bind rather than expose the API
#    unauthenticated — so with it unset the container panics on every boot:
#      docker run -e KOHARU_API_TOKEN="$(openssl rand -hex 32)" ...
#    On vast.ai put it in the template's environment. It is deliberately not
#    defaulted or auto-generated: a token the operator never chose is one
#    nobody rotates.
# 3. `KOHARU_RPC_PORT` must match `devUrl` in crates/koharu/tauri.conf.json
#    (currently 47823) for the desktop window's *initial* navigation to
#    succeed without a race (see HANDOFF.md's Phase 3c section) — if you
#    override the port here, update tauri.conf.json's devUrl to match
#    before building, or the CEF window (not the HTTP API, which is
#    unaffected) will show a connection-refused page.
# 4. Xvfb has no GPU/compositor behind it; CEF fell back to software
#    rendering without complaint in local testing, but if a future CEF/
#    driver combination insists on real GPU access, this is the first
#    place to check.
# 5. Xvfb here is a bare virtual framebuffer with no compositor/GPU
#    acceleration wired to it; if CEF's GPU process insists on a real
#    display/DRM node and fails hard instead of falling back to software
#    rendering, this is the first place to look.
