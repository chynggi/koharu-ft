# koharu-server

Runs Koharu's models for the desktop app on another machine, such as a GPU
server, while editing stays in the app.

## How a run goes

1. The app sends the pages a run works on, each as a snapshot that makes the
   page again under its own ids (`Session::snapshot`), and the request with
   the app's pipeline settings.
2. The server makes a project of them in a temporary folder, records every
   edit from then on (`Session::record`), and runs the request with its own
   models.
3. It answers with the run's events and edits, in order. The app replays
   each edit on its project (`Session::replay`), which makes the same ids, so
   a remote run writes what a local one would: one undo step per stage and
   page, under the run's id in the app.

The wire is `koharu_pipeline::remote`'s: `POST /v1/runs` with a body of
zstd-compressed JSON, answered by frames of zstd-compressed JSON, each after
its length as four little-endian bytes. `GET /v1/health` answers `ok`.

- **One run at a time.** The models share one GPU; a second run is refused
  with 409 until the first ends.
- **A run whose app hangs up stops**, as when the app's Stop is pressed.
- **The project must hold still.** An edit that no longer replays, because the
  page changed under the run, fails the run there; what it wrote before stays.
- **A translation conversation lasts one run**, since each run is its own
  project on the server.

## Running it

```sh
KOHARU_SERVER_TOKEN=$(openssl rand -hex 32) koharu-server --bind 0.0.0.0:47823
```

- `--bind` defaults to `127.0.0.1:47823`. Any address but this machine's needs
  `KOHARU_SERVER_TOKEN`, which every request brings as its bearer token.
- Secrets come from the environment, never a file: a hosted translator's key
  from `KOHARU_SECRET_` and its name in capitals, any other character `_`,
  such as `KOHARU_SECRET_OPENAI` for `openai`. Signing in to a vendor isn't
  possible here.
- The runtime downloads what the GPU needs on first use into its store, under
  the local data folder (`~/.local/share/Koharu/store` on Linux).

In the app, set Settings > Pipeline > Processing server to the server's
address, such as `http://192.168.0.10:47823`, and its token. Leaving the
address empty runs here again.

## Docker

The repository's `Dockerfile` builds the server for Linux with an NVIDIA GPU:

```sh
docker build -t koharu-server .
docker run --gpus all -p 47823:47823 \
  -e KOHARU_SERVER_TOKEN=... -e KOHARU_SECRET_OPENAI=... \
  -v koharu-store:/root/.local/share koharu-server
```

Keep the store on a volume, or every new container downloads CUDA, libtorch,
and the models again. Pass secrets when the container starts, never as build
arguments, which stay in the image.
