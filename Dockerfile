# Koharu's processing server, for a Linux machine with an NVIDIA GPU: the
# desktop app sends it runs (Settings > Pipeline > Processing server). The
# runtime downloads CUDA, libtorch, llama.cpp, and stable-diffusion.cpp into
# its store on first use, so the image carries only the server; keep the
# store on a volume so that happens once.
#
#   docker build -t koharu-server .
#   docker run --gpus all -p 47823:47823 -e KOHARU_SERVER_TOKEN=... \
#     -v koharu-store:/root/.local/share koharu-server

FROM rust:1-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
        clang libclang-dev cmake ninja-build pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /koharu
COPY . .
RUN cargo build --release --locked -p koharu-server -j 8

FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libssl3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /koharu/target/release/koharu-server /usr/local/bin/koharu-server
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
VOLUME ["/root/.local/share", "/root/.koharu"]
EXPOSE 47823
ENTRYPOINT ["koharu-server", "--bind", "0.0.0.0:47823"]
