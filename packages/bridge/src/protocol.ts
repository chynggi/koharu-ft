// Hand-written fetch/SSE client for koharu-rpc. Replaces the generated
// Tauri IPC bridge; keeps the same `commands` shape and types so callers
// need zero changes beyond the transport underneath.

// Spelled as a literal `process.env.NEXT_PUBLIC_...` member expression on
// purpose. Next substitutes the build-time value by matching exactly that
// shape in the source, so reaching it through a variable
// (`globalThis.process?.env`) is never substituted, leaves `process`
// undefined in the browser, and silently pins every deployment to the
// default below. There is no `typeof process` guard for the same reason: it
// survives substitution and would evaluate false in the browser, discarding
// the value that was just inlined next to it.
// Declared locally rather than by depending on @types/node: this module runs
// in the browser, and only this one substituted value is ever read from here.
declare const process: { env: Record<string, string | undefined> };
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "/api/v1";
// Injected into the served index.html by koharu-rpc at startup when the API
// is bound to a non-loopback host (see KOHARU_API_TOKEN in
// crates/koharu/src/main.rs), so the token never has to be baked into the
// static bundle or the Docker image. Unset for local loopback use.
const API_TOKEN = (globalThis as { __KOHARU_API_TOKEN__?: string }).__KOHARU_API_TOKEN__;

function authHeaders(headers?: Record<string, string>): Record<string, string> | undefined {
	if (!API_TOKEN) return headers;
	return { ...headers, authorization: `Bearer ${API_TOKEN}` };
}

async function errorMessage(response: Response): Promise<string> {
	try {
		const body = await response.json();
		return typeof body?.error === "string" ? body.error : "request failed";
	} catch {
		return `request failed (HTTP ${response.status})`;
	}
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
	const response = await fetch(`${API_BASE_URL}${path}`, {
		...init,
		headers: authHeaders(init?.headers as Record<string, string> | undefined),
	});
	if (!response.ok) throw new Error(await errorMessage(response));
	const text = await response.text();
	if (!text) return null as T;
	try {
		return JSON.parse(text) as T;
	} catch {
		// A 2xx body that is not JSON means the request never reached the API:
		// something in front of it — a dev server, a proxy, a static-file
		// fallback — answered with its own page. Say that, rather than letting
		// `Unexpected token '<'` stand in for it.
		const type = response.headers.get("content-type") ?? "an unknown content type";
		throw new Error(`${path} returned ${type} instead of JSON; the Koharu API did not answer`);
	}
}

function get<T>(path: string): Promise<T> {
	return request<T>(path);
}

function post<T>(path: string, body?: unknown): Promise<T> {
	return request<T>(path, {
		method: "POST",
		headers: body === undefined ? undefined : { "content-type": "application/json" },
		body: body === undefined ? undefined : JSON.stringify(body),
	});
}

function put<T>(path: string, body: unknown): Promise<T> {
	return request<T>(path, {
		method: "PUT",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(body),
	});
}

// No content-type: the browser has to set it, because only it knows the
// multipart boundary it generated.
function upload<T>(path: string, form: FormData): Promise<T> {
	return request<T>(path, { method: "POST", body: form });
}

/** 이진 응답을 Blob으로 가져온다. `bytes`와 달리 파일로 저장할 것에 쓴다. */
async function downloadGet(path: string): Promise<Blob> {
	const response = await fetch(`${API_BASE_URL}${path}`, { headers: authHeaders() });
	if (!response.ok) throw new Error(await errorMessage(response));
	return response.blob();
}

function del<T>(path: string): Promise<T> {
	return request<T>(path, { method: "DELETE" });
}

async function bytes(path: string): Promise<ArrayBuffer> {
	const response = await fetch(`${API_BASE_URL}${path}`, { headers: authHeaders() });
	if (!response.ok) throw new Error(await errorMessage(response));
	return response.arrayBuffer();
}

async function readSse<T>(response: Response, onEvent: (event: T) => void): Promise<void> {
	const reader = response.body?.getReader();
	if (!reader) return;
	const decoder = new TextDecoder();
	let buffer = "";
	for (;;) {
		const { done, value } = await reader.read();
		if (done) break;
		buffer += decoder.decode(value, { stream: true });
		let boundary: number;
		while ((boundary = buffer.indexOf("\n\n")) !== -1) {
			const chunk = buffer.slice(0, boundary);
			buffer = buffer.slice(boundary + 2);
			const line = chunk.split("\n").find((entry) => entry.startsWith("data:"));
			const payload = line?.slice(5).trim();
			if (payload) onEvent(JSON.parse(payload) as T);
		}
	}
}

/** Commands */
export const commands = {
	getAgentStatus: () => get<AgentStatus>("/agent/status"),
	loginAgent: (onEvent: (event: LoginEvent) => void) =>
		fetch(`${API_BASE_URL}/agent/login`, { method: "POST", headers: authHeaders() }).then(
			async (response) => {
				if (!response.ok) throw new Error(await errorMessage(response));
				await readSse<LoginEvent>(response, onEvent);
				return get<AgentStatus>("/agent/status");
			},
		),
	logoutAgent: () => post<AgentStatus>("/agent/logout"),
	saveAgentConfig: (config: Config) => put<Config>("/agent/config", config),
	runAgent: (prompt: string, onEvent: (event: Event) => void) =>
		new Promise<RunId>((resolve, reject) => {
			fetch(`${API_BASE_URL}/agent/run`, {
				method: "POST",
				headers: authHeaders({ "content-type": "application/json" }),
				body: JSON.stringify({ prompt }),
			})
				.then(async (response) => {
					if (!response.ok) {
						reject(new Error(await errorMessage(response)));
						return;
					}
					let started = false;
					await readSse<Event>(response, (event) => {
						onEvent(event);
						if (!started && event.type === "started") {
							started = true;
							resolve(event.run);
						}
					});
					if (!started) reject(new Error("the agent run ended before it started"));
				})
				.catch(reject);
		}),
	cancelAgent: (run: RunId) => post<null>(`/agent/run/${run}/cancel`),
	subscribe: () =>
		Promise.all([get<Preferences>("/config"), get<Job[]>("/jobs")]).then(
			([preferences, jobs]): StartupState => ({
				preferences,
				jobs,
				canvas: {
					page: null,
					revision: null,
					generation: 0,
					size: [0, 0],
					element_frames: [],
				},
			}),
		),
	getProject: () => get<ProjectInfo | null>("/project"),
	getPages: () => get<PageSummary[]>("/pages"),
	getPage: () => get<Page | null>("/page"),
	listProjects: () => get<ProjectSummary[]>("/projects"),
	createProject: (name: string) => post<null>("/projects", { name }),
	openProject: (name: string) => post<null>(`/projects/${encodeURIComponent(name)}/open`),
	deleteProject: (name: string) => del<null>(`/projects/${encodeURIComponent(name)}`),
	closeProject: () => post<null>("/project/close"),
	// Linux desktop only: hands an interactive move (no direction) or resize of
	// the window to the window manager for the pointer button being held.
	startWindowMoveResize: (direction?: ResizeDirection) =>
		post<null>("/window/move-resize", { direction: direction ?? null }),
	// Server-side paths. Only meaningful when the caller can name files the
	// server can open — that is, the desktop window. A browser cannot, and must
	// use `importPagesUpload` instead; see the note there.
	importPages: (files: string[]) => post<PageSummary[]>("/pages/import", { files }),
	// The remote path. Source images live on the user's machine, so there is no
	// path the server could be given that would refer to the same file — the
	// bytes have to travel. Accepts archives (cbz/zip/rar) and PDFs as well as
	// images, exactly like the path-based route.
	importPagesUpload: (files: File[]) => {
		const form = new FormData();
		for (const file of files) form.append("files", file, file.name);
		return upload<PageSummary[]>("/pages/import/upload", form);
	},
	// The desktop path: the dialog opens in the server process, which is the
	// user's own machine there. Rejected for a remote caller — see the route.
	importPagesDialog: (source: PageImportSource) =>
		post<PageSummary[]>("/pages/import/dialog", { source }),
	selectPage: (page: EntityId) => post<PageSelection>("/page/select", { page }),
	renamePage: (page: EntityId, label: string) => post<null>("/page/rename", { page, label }),
	// 챕터 경계는 페이지 하나의 불리언이라 전용 라우트를 쓴다.
	setPageChapterStart: (page: EntityId, chapterStart: boolean) =>
		post<null>("/page/chapter-start", { page, chapter_start: chapterStart }),
	deletePages: (pages: EntityId[]) => post<null>("/pages/delete", { pages }),
	movePage: (page: EntityId, index: number) => post<null>("/page/move", { page, index }),
	// 인접한 페이지들을 하나의 넓은 스프레드로 합치고 새 페이지 id를 돌려준다.
	// `delete_originals`는 라우트 구조체가 snake_case 그대로 읽으므로 JSON 키도 snake_case다.
	mergePages: (pages: EntityId[], deleteOriginals: boolean) =>
		post<EntityId>("/pages/merge", { pages, delete_originals: deleteOriginals }),
	setSourceText: (layer: EntityId, text: string) =>
		post<null>("/layers/source-text", { layer, text }),
	setTranslation: (layer: EntityId, text: string | null) =>
		post<null>("/layers/translation", { layer, text }),
	setTypography: (updates: TypographyUpdate[]) =>
		post<null>("/layers/typography", { updates }),
	setGeometry: (updates: GeometryUpdate[]) => post<null>("/layers/geometry", { updates }),
	setVisibility: (layers: EntityId[], visible: boolean | null, opacity: number | null) =>
		post<null>("/layers/visibility", { layers, visible, opacity }),
	deleteLayers: (layers: EntityId[]) => post<null>("/layers/delete", { layers }),
	moveLayer: (layer: EntityId, parent: EntityId, index: number) =>
		post<Page>("/layers/move", { layer, parent, index }),
	// 클립보드에 담아 둔 텍스트 레이어들을 다른 페이지로 복제하고 새 id를 돌려준다.
	pasteTextLayers: (layers: EntityId[], page: EntityId) =>
		post<EntityId[]>("/layers/paste", { layers, page }),
	undo: () => post<null>("/project/undo"),
	redo: () => post<null>("/project/redo"),
	process: (scope: Scope, operation: Operation) => post<JobId>("/process", { scope, operation }),
	stopJob: (job: JobId) => post<null>("/process/stop", { job }),
	// A directory on the server. Same caveat as `importPages`: only a caller
	// that can name the server's filesystem has any use for this.
	// 셋 다 즉시 `JobId`를 돌려주고 렌더링은 백그라운드에서 돈다. 진행률은
	// `openEventStream`의 job 이벤트로 오고, 중단은 `stopJob`이다.
	exportPages: (pages: EntityId[], options: ExportOptions, directory: string) =>
		post<JobId>("/pages/export", { pages, options, directory }),
	// The desktop path: native folder picker, server-side, loopback only.
	// 사용자가 선택창을 닫으면 `null`이다.
	exportPagesDialog: (pages: EntityId[], options: ExportOptions) =>
		post<JobId | null>("/pages/export/dialog", { pages, options }),
	// The remote path. Rendered output has to reach the user's machine, so it
	// comes back as one ZIP rather than being written somewhere they cannot see.
	// 렌더링과 전송이 나뉘어 있다: 이것이 job을 시작하고,
	// `getExportArchive`가 끝난 뒤의 ZIP을 가져온다.
	exportPagesDownload: (pages: EntityId[], options: ExportOptions) =>
		post<JobId>("/pages/export/download", { pages, options }),
	getExportArchive: (job: JobId) => downloadGet(`/pages/export/download/${job}`),
	getThumbnail: (page: EntityId) => bytes(`/pages/${page}/thumbnail`),
	getFonts: () => get<FontFamily[]>("/fonts"),
	getFontPreview: (familyName: string) =>
		bytes(`/fonts/${encodeURIComponent(familyName)}/preview`),
	savePreferences: (
		pipeline: PipelineConfig,
		providers: ProviderPreferences,
		typesetting: TypesettingConfig,
	) => put<Preferences>("/config", { pipeline, providers, typesetting }),
	getPreferences: () => get<Preferences>("/config"),
	getTranslationModels: () => get<Model[]>("/translation-models"),
	getLlmCapabilities: () => get<LlmCapabilities>("/llm/capabilities"),
	// The picker runs in the server process, so it only makes sense for a
	// loopback caller (the desktop window). A remote browser gets `null` and
	// the caller is expected to let the user type the server-side path
	// instead — the path has to be one koharu-rpc can read either way.
	pickGgufFile: () => post<string | null>("/llm/gguf-file"),
	getCanvasManifest: (generation: CanvasGeneration) => bytes(`/canvas/manifest/${generation}`),
	getCanvasResource: (generation: CanvasGeneration, resource: string) =>
		bytes(`/canvas/resource/${generation}/${encodeURIComponent(resource)}`),
	prepareCanvasPage: (page: EntityId) =>
		post<CanvasPagePreparation | null>(`/pages/${page}/canvas/prepare`),
	getCanvasPageManifest: (page: EntityId, revision: Revision) =>
		bytes(`/pages/${page}/canvas/manifest/${revision}`),
	getCanvasPageResource: (page: EntityId, revision: Revision, resource: string) =>
		bytes(`/pages/${page}/canvas/resource/${revision}/${encodeURIComponent(resource)}`),
	addPointText: (point: Point) => post<LayerCommit>("/canvas/text/point", { point }),
	addTextBox: (frame: Frame) => post<LayerCommit>("/canvas/text/box", { frame }),
	addShape: (
		expectedRevision: Revision,
		page: EntityId,
		geometry: Geometry,
		shape: ShapeGeometry,
		style: ShapeStyle,
		angleDegrees: number,
	) =>
		post<LayerCommit>("/canvas/shape", {
			expected_revision: expectedRevision,
			page,
			geometry,
			shape,
			style,
			angle_degrees: angleDegrees,
		}),
	setShapeStyle: (expectedRevision: Revision, updates: ShapeStyleUpdate[]) =>
		post<Revision>("/canvas/shape/style", { expected_revision: expectedRevision, updates }),
	/**
	 * Sets or clears a shape's pattern fill. `null` restores a solid fill.
	 *
	 * Separate from `setShapePatternAsset` because uploading an image is large
	 * and can fail, while changing a parameter is immediate; one command would
	 * let a slow upload block a parameter change.
	 */
	setShapePattern: (expectedRevision: Revision, layer: EntityId, pattern: PatternSpec | null) =>
		post<LayerCommit>("/canvas/shape/pattern", {
			expected_revision: expectedRevision,
			layer,
			pattern,
		}),
	/** Uploads the `pattern` asset of a shape layer. Shape layers only. */
	setShapePatternAsset: (
		expectedRevision: Revision,
		layer: EntityId,
		bytes: number[],
		mediaType: string,
	) =>
		post<LayerCommit>("/canvas/shape/pattern/asset", {
			expected_revision: expectedRevision,
			layer,
			bytes,
			media_type: mediaType,
		}),
	rasterizeShapes: (expectedRevision: Revision, page: EntityId, shapes: EntityId[]) =>
		post<Revision>("/canvas/shape/rasterize", {
			expected_revision: expectedRevision,
			page,
			shapes,
		}),
	manualOcr: (points: Point[]) => post<ManualOcrCommit>("/canvas/ocr/manual", { points }),
	commitPaint: (
		expectedRevision: Revision,
		layer: string | null,
		points: Point[],
		brush: PaintBrush,
	) =>
		post<LayerCommit>("/canvas/commit/paint", {
			expected_revision: expectedRevision,
			layer,
			points,
			brush,
		}),
	commitErase: (expectedRevision: Revision, layer: EntityId, points: Point[], diameter: number) =>
		post<LayerCommit>("/canvas/commit/erase", {
			expected_revision: expectedRevision,
			layer,
			points,
			diameter,
		}),
	commitTransform: (expectedRevision: Revision, elements: TransformFrame[]) =>
		post<number | null>("/canvas/commit/transform", {
			expected_revision: expectedRevision,
			elements,
		}),
	commitInpaint: (expectedRevision: Revision, points: Point[], diameter: number) =>
		post<string | null>("/canvas/commit/inpaint", {
			expected_revision: expectedRevision,
			points,
			diameter,
		}),
	adjustRaster: (expectedRevision: Revision, layer: EntityId, adjust: ColorAdjust) =>
		post<LayerCommit>("/canvas/raster/adjust", {
			expected_revision: expectedRevision,
			layer,
			adjust,
		}),
	fillRaster: (
		expectedRevision: Revision,
		layer: EntityId,
		point: Point,
		color: [number, number, number, number],
	) =>
		post<LayerCommit>("/canvas/raster/fill", {
			expected_revision: expectedRevision,
			layer,
			point,
			color,
		}),
	getMeta: () => get<{ name: string; version: string }>("/meta"),
};

/** Live event stream (canvas/job/download/resource/project), replacing the
 *  Tauri `subscribe` Channels. Caller owns the returned EventSource and
 *  must `.close()` it when done. */
export function openEventStream(handlers: {
	onCanvas: (value: CanvasState) => void;
	onJob: (value: Job) => void;
	onDownload: (value: Download) => void;
	onResource: (value: ModelResources) => void;
	onProject: (value: ProjectInfo | null) => void;
}): EventSource {
	const eventsUrl = API_TOKEN
		? `${API_BASE_URL}/events?token=${encodeURIComponent(API_TOKEN)}`
		: `${API_BASE_URL}/events`;
	const source = new EventSource(eventsUrl);
	source.onmessage = (message) => {
		const envelope = JSON.parse(message.data) as { type: string; data: unknown };
		switch (envelope.type) {
			case "canvas":
				handlers.onCanvas(envelope.data as CanvasState);
				break;
			case "job":
				handlers.onJob(envelope.data as Job);
				break;
			case "download":
				handlers.onDownload(envelope.data as Download);
				break;
			case "resource":
				handlers.onResource(envelope.data as ModelResources);
				break;
			case "project":
				handlers.onProject(envelope.data as ProjectInfo | null);
				break;
		}
	};
	source.onerror = (error) => {
		console.error("koharu event stream error", error);
	};
	return source;
}

/* Types */
export type ResizeDirection =
	| "North"
	| "NorthEast"
	| "East"
	| "SouthEast"
	| "South"
	| "SouthWest"
	| "West"
	| "NorthWest";

export type Account = {
	id: string,
	email: string | null,
	plan: string | null,
};

export type AgentStatus = {
	account: Account | null,
	models: CodexModel[],
	config: Config,
	running: RunId | null,
};

export type AnalysisRegion = {
	id: EntityId,
	parent: EntityId | null,
	geometry: Geometry,
	kind: string,
	label: string | null,
};

export type ArrowHead = {
	length: number,
	width: number,
};

export type Bounds = {
	x: number,
	y: number,
	width: number,
	height: number,
};

export type CaiyunConfig = Record<string, never>;

export type CanvasBytes = number[];

export type CanvasGeneration = number;

export type CanvasPagePreparation = {
	revision: Revision,
	page: Page,
};

export type CanvasState = {
	page: EntityId | null,
	revision: Revision | null,
	generation: number,
	size: [number, number],
	element_frames: TransformFrame[],
};

export type ClaudeConfig = Record<string, never>;

export type CodexModel = {
	id: string,
	name: string,
	reasoning: Reasoning[],
};

/**
 *  Where one FLUX.2 Klein checkpoint is loaded from.
 * 
 *  Tagged with `kind` so that `koharu_config`'s merge treats a changed variant
 *  as a replaced subtree instead of blending two shapes.
 */
export type ComponentSourceConfig = 
/**  The repository Koharu pins for this component. */
{ kind: "builtin" } | 
/**  A checkpoint already on disk. Nothing is downloaded and the file is only read. */
{ kind: "local_file"; path: string } | 
/**  Any Hugging Face repository. Without a revision the repository head is used. */
{ kind: "hugging_face"; repository: string; revision?: string | null; filename: string } |
/**  An arbitrary URL. `digest` is a 64-character BLAKE3 hex digest (case-insensitive). */
{ kind: "url"; url: string; digest: string };

export type Config = {
	model: string | null,
	reasoning: Reasoning,
};

/**
 *  How the llama.cpp context is sized.
 * 
 *  `Dynamic` is the default and reproduces Koharu's per-inference sizing: the
 *  context is exactly what the prepared prompt plus the requested output needs.
 */
export type ContextMode = 
/**  Size the context from the prompt on every call. */
{ kind: "dynamic" } | 
/**  Always allocate `size` positions. Smaller than a prompt requires is an error. */
{ kind: "fixed"; size: number } | 
/**  Size dynamically, then clamp into the given range. */
{ kind: "bounded"; minimum?: number | null; maximum?: number | null };

export type CredentialInput = {
	configured: boolean,
	editable: boolean,
	environment_variable: string | null,
	value: string | null,
	clear: boolean,
};

/**  A GGUF file the user registered from their own filesystem. */
export type CustomModel = {
	/**  Stable identifier stored in the pipeline's model selection. */
	id: string,
	/**  Display name shown in the model picker. */
	name: string,
	/**  Absolute path to the `.gguf` weights. */
	path: string,
	/**  Optional MTMD projector; present means the model is used with vision. */
	projector?: string | null,
};

export type DeepLConfig = {
	base_url?: string | null,
};

export type DeepSeekConfig = Record<string, never>;

export type DeferredCapability = {
	/**  Field name in the runtime settings, e.g. `flash_attention`. */
	setting: string,
	reason: string,
};

export type DetectionModel = {
	model: "koharu-layout-rfdetr-seg-2xl",
} & KoharuLayoutRFDetrSeg2XLConfig;

export type DeviceResources = {
	name: string,
	selected: boolean,
	memory_budget: number | null,
	memory_used: number | null,
	/**
	 *  How much of the machine the two figures above cover. Windows reports
	 *  this process alone, the Linux providers report the whole device, so the
	 *  UI has to say which rather than labelling both "VRAM in use".
	 */
	memory_scope: MemoryScope | null,
	utilization: number | null,
};

export type Download = {
	id: number,
	state: DownloadState,
	name: string | null,
	completed: number,
	total: number,
	error: string | null,
};

export type DownloadState = "running" | "finished" | "failed";

export type EntityId = string;

export type Error = string;

export type Event = { type: "started"; run: RunId } | { type: "text_delta"; run: RunId; delta: string } | { type: "reasoning_delta"; run: RunId; delta: string } | { type: "tool_started"; run: RunId; call_id: string; name: string } | { type: "tool_finished"; run: RunId; call_id: string; name: string; changed: boolean; output: string } | { type: "completed"; run: RunId; message: string } | { type: "failed"; run: RunId; message: string } | { type: "cancelled"; run: RunId };

export type ExportFormat = "png" | "psd" | "cbz";

export type ExportOptions = {
	formats: ExportFormat[],
	pattern: string,
	subfolders: boolean,
};

export type FlashAttentionMode = 
/**  Let llama.cpp decide per backend and model. */
"auto" | "on" | "off";

export type Flux2KleinConfig = {
	prompt?: string,
	/**
	 *  Which checkpoints the context is assembled from. Defaults to the pinned
	 *  FLUX.2 Klein 4B repositories.
	 */
	source?: Flux2KleinSourceConfig,
	steps?: number,
	strength?: number,
	/**  `-1` draws a fresh seed for every call. */
	seed?: number,
	padding_mask_crop?: number | null,
	/**  The working area every tile is shrunk to before denoising. */
	max_pixels?: number,
};

export type Flux2KleinSourceConfig = {
	transformer?: ComponentSourceConfig,
	text_encoder?: ComponentSourceConfig,
	vae?: ComponentSourceConfig,
};

export type FontFace = {
	postscript_name: string,
	weight: number,
	weight_range: FontRange | null,
	style: FontStyle,
};

export type FontFamily = {
	name: string,
	metadata: FontMetadata,
	sources: FontSource[],
	faces: FontFace[],
};

export type FontMetadata = {
	primary_script: string | null,
	scripts: string[],
	languages: string[],
	category: string | null,
	classifications: string[],
	use_cases: string[],
};

export type FontPreviewBytes = number[];

export type FontRange = {
	minimum: number,
	maximum: number,
};

export type FontSource = "system" | "bundled";

export type FontStyle = "normal" | "italic" | "oblique";

export type Frame = {
	x: number,
	y: number,
	width: number,
	height: number,
	angle_degrees: number,
};

export type GeminiConfig = Record<string, never>;

export type GenerationConfig = {
	temperature?: number | null,
	top_k?: number | null,
	top_p?: number | null,
	min_p?: number | null,
	max_tokens?: number | null,
	repeat_penalty?: number | null,
	frequency_penalty?: number | null,
	presence_penalty?: number | null,
	reasoning?: boolean | null,
	vision?: boolean | null,
};

export type Geometry = {
	points: Point[],
};

export type GeometryUpdate = {
	layer: EntityId,
	points: Point[] | null,
};

export type GoogleCloudConfig = Record<string, never>;

/**
 *  How many model layers to offload to the accelerator.
 *
 *  `All` is the default and matches Koharu's existing behaviour. A layer count
 *  larger than the model has is clamped by llama.cpp. When the selected device
 *  is the CPU, no layers are offloaded regardless of this setting.
 */
export type GpuLayers = { kind: "all" } | { kind: "custom"; layers: number };

export type GrokConfig = Record<string, never>;

export type GroupRole = "text";

export type InpaintingModel = {
	model: "lama",
} & LaMaConfig | { model: "mi-gan" } & MiGanConfig | { model: "manga-inpaintor" } & MangaInpaintorConfig | { model: "aot-inpainting" } | {
	model: "flux2-klein",
} & Flux2KleinConfig | { model: "powerpaint" } & PowerPaintConfig | {
	model: "rorem-mixed",
} & RoremMixedConfig;

export type Job = {
	id: JobId,
	kind: JobKind,
	state: JobState,
	completed: number,
	total: number,
	page: EntityId | null,
	stage: Stage | null,
	model: string | null,
	error: string | null,
};

export type JobId = string;

export type JobKind = "processing" | "export";

export type JobState = "running" | "finished" | "failed" | "stopped";

export type KoharuLayoutRFDetrSeg2XLConfig = {
	text_threshold?: number | null,
	bubble_threshold?: number | null,
	panel_threshold?: number | null,
};

/**
 *  KV cache element types llama.cpp accepts for `type_k` / `type_v`.
 * 
 *  This is deliberately narrower than [`KvCacheType`]: the k-quant and most
 *  i-quant types exist in ggml but are not valid KV cache types, and offering
 *  them would only produce runtime failures.
 */
export type KvCacheChoice = "f32" | "f16" | "bf16" | "q8_0" | "q5_1" | "q5_0" | "q4_1" | "q4_0" | "iq4_nl";

/**
 *  LaMa checkpoint selection. Defaults to the `mayocream/lama-manga`
 *  safetensors checkpoint, matching the behavior before this field existed.
 */
export type LaMaConfig = {
	source?: ComponentSourceConfig,
	format?: WeightsFormatConfig,
};

export type LanguageChoice = {
	tag: string,
	name: string,
};

export type Layer = { type: "group"; id: EntityId; parent: EntityId | null; visibility: LayerVisibility; name: string; role: GroupRole | null } | { type: "text"; id: EntityId; parent: EntityId | null; geometry: Geometry | null; angle_degrees: number | null; visibility: LayerVisibility; content: TextContent; typography: Typography | null; layout: TextLayoutKind; automatic_region: EntityId | null } | { type: "raster"; id: EntityId; parent: EntityId | null; visibility: LayerVisibility; image: string | null; name: string; kind: RasterLayerKind } | { type: "shape"; id: EntityId; parent: EntityId | null; geometry: Geometry | null; angle_degrees: number | null; visibility: LayerVisibility; name: string; shape: ShapeGeometry; style: ShapeStyle; rasterized: boolean } | { type: "image"; id: EntityId; parent: EntityId | null; geometry: Geometry; visibility: LayerVisibility; image: string } | { type: "artwork"; id: EntityId; parent: EntityId | null; geometry: Geometry; visibility: LayerVisibility; image: string };

export type LayerCommit = {
	revision: Revision,
	layer: EntityId,
};

/**
 * A bucket fill. `point` is the click location in page pixels; `color` is the
 * colour to flood the connected region with. No tolerance is configurable: the
 * server fills exactly-matching connected pixels.
 */
export type FillRequest = {
	point: Point,
	color: [number, number, number, number],
};

/**
 * A raster colour adjustment. Command arguments, so this uses internal tagging
 * (`{ type, ... }`) like `Layer` rather than the external tagging that the
 * persisted scene components use.
 */
export type ColorAdjust =
	| { type: "brightness"; delta: number }
	| { type: "contrast"; percent: number }
	| { type: "levels"; low: number; high: number }
	| { type: "hue"; degrees: number }
	| { type: "grayscale" };

export type LayerVisibility = {
	visible: boolean,
	opacity: number,
};

export type LineCap = "butt" | "round" | "square";

export type LineJoin = "miter" | "round" | "bevel";

/**
 *  What the current build and device actually support.
 * 
 *  Anything llama.cpp only validates while creating a context is reported in
 *  `deferred` rather than guessed at: the UI keeps those controls enabled and
 *  surfaces llama.cpp's own error if a value is rejected.
 */
export type LlmCapabilities = {
	/**  Human-readable accelerator description, or `CPU`. */
	device: string,
	backend: string,
	/**
	 *  Whether layers can be offloaded at all. `false` means the GPU Layers
	 *  control has no effect.
	 */
	gpu_offload: boolean,
	/**  Total device memory in bytes, when the driver reports it. */
	total_memory: number | null,
	/**  Settings that are only validated when a context is created. */
	deferred: DeferredCapability[],
};

/**
 *  llama.cpp settings a power user can override. Every field is optional so the
 *  layers above can be merged without inventing values for what they do not set.
 */
export type LlmRuntimeConfig = {
	context?: ContextMode | null,
	/**  Upper bound on generated tokens. The per-run generation settings win. */
	max_output_tokens?: number | null,
	n_batch?: number | null,
	n_ubatch?: number | null,
	gpu_layers?: GpuLayers | null,
	n_threads?: number | null,
	n_threads_batch?: number | null,
	kv_cache_type_k?: KvCacheChoice | null,
	kv_cache_type_v?: KvCacheChoice | null,
	flash_attention?: FlashAttentionMode | null,
};

export type LmStudioConfig = {
	base_url?: string | null,
};

/**  `[providers.local]`. */
export type LocalConfig = {
	/**  Runtime settings applied to every local model. */
	runtime?: LlmRuntimeConfig,
	/**  Per-model overrides keyed by model id, layered over `runtime`. */
	profiles?: { [key in string]: LlmRuntimeConfig },
	/**  GGUF files registered by the user. */
	models?: CustomModel[],
};

export type LoginEvent = { type: "progress"; message: string } | { type: "device_code"; verification_url: string; user_code: string };

/**
 *  Manga inpainter checkpoint selection. The pipeline is assembled from an
 *  inpaintor and a line model.
 */
export type MangaInpaintorConfig = {
	inpaintor?: ComponentSourceConfig,
	line?: ComponentSourceConfig,
};

/**  Mirror of [`koharu_pipeline::MemoryScope`] for the frontend. */
export type MemoryScope = "process" | "device" | "system";

/**  Result of a manual OCR request: the region/layer added for the drawn area
 *  and the pipeline job that recognizes text inside it. */
export type ManualOcrCommit = {
	revision: Revision,
	region: EntityId,
	layer: EntityId,
	job: JobId,
};

export type MiniMaxConfig = Record<string, never>;

/**
 *  MI-GAN checkpoint selection. A prompt-free erase-only model, so it only
 *  carries a source.
 */
export type MiGanConfig = {
	source?: ComponentSourceConfig,
};

export type Model = {
	provider: Provider,
	model: string | null,
	name: string,
	quantizations: Quantization[],
	vision: boolean,
	reasoning: boolean,
};

export type ModelResources = {
	process_memory: number,
	system_memory: number,
	process_cpu: number,
	devices: DeviceResources[],
};

export type ModelSelection = {
	provider: Provider,
	model?: string | null,
	quantization?: string | null,
	vision?: boolean,
	reasoning?: boolean,
};

export type OcrModel = { model: "paddleocr-vl-1.6" } | {
	model: "paddleocr-vl-1.6-manga-lora",
} & PaddleOcrVl1_6MangaLoraConfig | { model: "manga-ocr" } | { model: "baberu-ocr" } | { model: "hayai-ocr" };

/**
 *  The manga reader variant of quantized PaddleOCR-VL-1.6: the fine-tuned
 *  vision tower projector plus the peft LoRA, both converted to GGUF (see
 *  `scripts/convert_paddleocr_vl_manga_lora.py`).
 */
export type PaddleOcrVl1_6MangaLoraConfig = {
	/**  The LoRA adapter GGUF. Defaults to the converted `manga-lora.gguf`. */
	lora?: ComponentSourceConfig,
	/**
	 *  The projector built from the fine-tuned vision tower. Defaults to the
	 *  converted `manga-mmproj.gguf`.
	 */
	projector?: ComponentSourceConfig,
	/**  Multiplier on the adapter's contribution. `1.0` is the llama.cpp default. */
	scale?: number,
};

export type OpenAiCompatibleConfig = {
	base_url?: string | null,
};

export type OpenAiConfig = Record<string, never>;

export type OpenRouterConfig = Record<string, never>;

export type Operation = { operation: "full" } | { operation: "through"; stage: Stage } | { operation: "only"; stage: Stage } | { operation: "stages"; stages: Stage[] };

export type Page = {
	id: EntityId,
	label: string,
	size: PageSize,
	layers: Layer[],
	regions: AnalysisRegion[],
};

export type PageImportSource = "files" | "folder";

export type PageSelection = {
	project: ProjectInfo,
	page: Page,
};

export type PageSize = {
	width: number,
	height: number,
};

export type PageSummary = {
	id: EntityId,
	label: string,
	size: PageSize,
	source_asset: string | null,
	chapter_start: boolean,
	layer_count: number,
};

export type RampAxis = "horizontal" | "vertical";

/**
 * A two-colour ramp for the brush. Command arguments, so this is internally
 * tagged (`kind`) like `ColorAdjust` rather than externally tagged like the
 * persisted scene components.
 */
export type BrushRamp =
	| {
			kind: "along_stroke",
			from: [number, number, number, number],
			to: [number, number, number, number],
	  }
	| {
			kind: "fixed_axis",
			from: [number, number, number, number],
			to: [number, number, number, number],
			axis: RampAxis,
	  };

export type PaintBrush = {
	diameter: number,
	color: [number, number, number, number],
	/** Absent (`null`) means a flat `color` stroke. */
	ramp: BrushRamp | null,
};

export type PipelineConfig = {
	detection: DetectionModel,
	ocr: OcrModel,
	translation: TranslationConfig,
	inpainting: InpaintingModel,
	/**
	 *  Settings for every model are kept independently of the active model.
	 *  The active stage fields above only select which profile is used.
	 */
	processor: ProcessorConfig,
};

export type Point = {
	x: number,
	y: number,
};

export type Preferences = {
	pipeline: PipelineConfig,
	providers: ProviderPreferences,
	typesetting: TypesettingConfig,
	languages: LanguageChoice[],
};

/**
 *  PowerPaint V1 inpainting. The converted GGUF and its task embeddings are
 *  produced locally by `scripts/convert_powerpaint.py`, so both paths are
 *  required and there is no pinned repository to fall back to.
 */
export type PowerPaintConfig = {
	model_path?: string,
	embeddings_dir?: string,
	mask_dilation?: number,
	steps?: number,
	guidance_scale?: number,
	seed?: number,
	strength?: number,
};

export type ProcessorConfig = {
	"koharu-layout-rfdetr-seg-2xl"?: KoharuLayoutRFDetrSeg2XLConfig | null,
	"paddleocr-vl-1.6-manga-lora"?: PaddleOcrVl1_6MangaLoraConfig | null,
	lama?: LaMaConfig | null,
	"mi-gan"?: MiGanConfig | null,
	"manga-inpaintor"?: MangaInpaintorConfig | null,
	"flux2-klein"?: Flux2KleinConfig | null,
	"rorem-mixed"?: RoremMixedConfig | null,
	powerpaint?: PowerPaintConfig | null,
};

export type ProjectInfo = {
	name: string,
	revision: Revision,
	active_page: EntityId | null,
	can_undo: boolean,
	can_redo: boolean,
};

export type ProjectSummary = {
	name: string,
};

export type Provider = "local" | "openai" | "gemini" | "claude" | "grok" | "minimax" | "deepseek" | "openai-compatible" | "openrouter" | "lm-studio" | "deepl" | "google-cloud-translation" | "caiyun";

export type ProviderConfig = { provider: "local"; settings: LocalConfig } | { provider: "openai"; settings: OpenAiConfig } | { provider: "gemini"; settings: GeminiConfig } | { provider: "claude"; settings: ClaudeConfig } | { provider: "grok"; settings: GrokConfig } | { provider: "minimax"; settings: MiniMaxConfig } | { provider: "deepseek"; settings: DeepSeekConfig } | { provider: "openai-compatible"; settings: OpenAiCompatibleConfig } | { provider: "openrouter"; settings: OpenRouterConfig } | { provider: "lm-studio"; settings: LmStudioConfig } | { provider: "deepl"; settings: DeepLConfig } | { provider: "google-cloud-translation"; settings: GoogleCloudConfig } | { provider: "caiyun"; settings: CaiyunConfig };

export type ProviderPreference = {
	name: string,
	config: ProviderConfig,
	credential: CredentialInput | null,
};

export type ProviderPreferences = {
	entries: ProviderPreference[],
};

export type Quantization = {
	id: string,
	name: string,
	downloaded: boolean,
};

export type RasterLayerKind = "cleanup" | "paint";

/**
 *  Regenerable shape parameters. The outline polygon is the entity's separate
 *  `Geometry`, not part of this type (decision D8).
 *
 *  These match serde's external tagging exactly — `Rectangle`/`Line` are
 *  struct variants and serialize as `{ rectangle: { corner_radius } }`, while
 *  `Ellipse`/`Polygon`/`Freeform` are unit variants and serialize as bare
 *  strings. `ShapeFill::Solid` is a newtype variant, so it is
 *  `{ solid: [r,g,b,a] }` with no `kind` key, while `ShapeFill::Gradient` is a
 *  newtype variant over a struct, so it is a one-key `{ gradient: {...} }`.
 *  `GradientKind` is likewise externally tagged and `GradientExtend` is a unit
 *  enum, so it stays a bare lowercase string.
 *  `the_shape_wire_shape_matches_the_typescript_client` and
 *  `the_gradient_wire_shape_matches_the_typescript_client` in
 *  `crates/koharu-app/src/commands/project.rs` pin every one of these strings;
 *  change those tests and these types together.
 */
export type ShapeGeometry = { rectangle: { corner_radius: number } } | "ellipse" | { line: {
	start: [number, number],
	end: [number, number],
	head: ArrowHead | null,
} } | "polygon" | "freeform";

/**
 *  Gradient control points are **normalized frame-local** values: the frame's
 *  top-left is `(0,0)` and its bottom-right is `(1,1)`. They are never page
 *  coordinates, so moving or resizing the shape does not invalidate them.
 *
 *  Stops may be stored in any order; the renderer sorts them stably by offset.
 *  Only duplicate offsets are rejected, because which color wins would
 *  otherwise be undefined.
 */
export type GradientSpec = {
	kind: GradientKind,
	stops: GradientStop[],
	extend: GradientExtend,
};

/**
 *  `radial`'s radius is a fraction of the frame's **width**, not of its height
 *  or of either diagonal. `sweep`'s angles are absolute degrees rather than
 *  normalized values, so no frame factor applies to them.
 */
export type GradientKind = { linear: { start: [number, number], end: [number, number] } } | { radial: {
	center: [number, number],
	radius: number,
} } | { sweep: {
	center: [number, number],
	start_degrees: number,
	end_degrees: number,
} };

export type GradientStop = {
	offset: number,
	color: [number, number, number, number],
};

export type GradientExtend = "pad" | "repeat" | "reflect";

export type PatternRepeat = "tile" | "stretch";

/**
 * An image-pattern fill for a shape.
 *
 * The pattern image is owned by the shape's own layer under the `pattern` asset
 * role and is uploaded separately by `setShapePatternAsset`, so this type
 * references nothing. `origin` and `scale` are ignored when `repeat` is
 * `"stretch"`, which fits one image to the shape's box instead.
 */
export type PatternSpec = {
	repeat: PatternRepeat,
	origin: [number, number],
	scale: number,
	rotation_degrees: number,
	fallback: [number, number, number, number],
};

export type ShapeFill =
	| { solid: [number, number, number, number] }
	| { gradient: GradientSpec }
	| { pattern: PatternSpec };

export type StrokeStyle = {
	color: [number, number, number, number],
	width: number,
	dash: number[] | null,
	cap: LineCap,
	join: LineJoin,
};

export type ShapeStyle = {
	fill: ShapeFill | null,
	stroke: StrokeStyle | null,
};

export type ShapeStyleUpdate = {
	element: EntityId,
	style: ShapeStyle,
};

export type Reasoning = "low" | "medium" | "high" | "xhigh" | "max" | "ultra";

export type Revision = number;

export type RoremMixedConfig = {
	prompt?: string,
	negative_prompt?: string,
};

export type RunId = string;

export type Scope = { scope: "project" } | { scope: "pages"; value: EntityId[] } | { scope: "region"; value: {
	page: EntityId,
	bounds: Bounds,
} } | { scope: "entities"; value: EntityId[] };

export type SourceText = {
	text: string,
	language: string | null,
};

export type Stage = "detection" | "ocr" | "translation" | "inpainting";

export type StartupState = {
	preferences: Preferences,
	jobs: Job[],
	canvas: CanvasState,
};

export type TextAlignment = "Start" | "Center" | "End" | "Justify";

export type TextContent = {
	id: EntityId,
	source: SourceText | null,
	translation: Translation | null,
	role: string | null,
	source_region: EntityId | null,
};

export type TextLayoutKind = "point" | "paragraph";

export type ThumbnailBytes = number[];

export type TransformFrame = {
	element: EntityId,
	frame: Frame,
};

export type Translation = {
	text: string,
	language: string | null,
};

export type TranslationConfig = {
	model: ModelSelection,
	generation: GenerationConfig,
	target_language: string,
	instructions: string | null,
	combine_pages?: boolean,
	max_pages_per_request?: number,
};

export type TypesettingConfig = {
	font_families?: string[],
};

export type Typography = {
	preferred_font: string | null,
	font_weight: number | null,
	font_style: FontStyle | null,
	size: number | null,
	auto_fit: boolean,
	color: [number, number, number, number] | null,
	stroke_color: [number, number, number, number] | null,
	stroke_width: number | null,
	alignment: TextAlignment | null,
	writing_mode: WritingMode | null,
};

export type TypographyUpdate = {
	layer: EntityId,
	typography: Typography,
};

/**
 *  The format of the LaMa weights file. The config representation of
 *  `koharu_ml::lama::WeightsFormat`.
 */
export type WeightsFormatConfig = "safe_tensors" | "torch_script";

export type WritingMode = "Horizontal" | "Vertical";
