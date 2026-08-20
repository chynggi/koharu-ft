// Hand-written fetch/SSE client for koharu-rpc. Replaces the generated
// Tauri IPC bridge; keeps the same `commands` shape and types so callers
// need zero changes beyond the transport underneath.

const env = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process
	?.env;
const API_BASE_URL = env?.NEXT_PUBLIC_API_BASE_URL || "/api/v1";
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
		return "request failed";
	}
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
	const response = await fetch(`${API_BASE_URL}${path}`, {
		...init,
		headers: authHeaders(init?.headers as Record<string, string> | undefined),
	});
	if (!response.ok) throw new Error(await errorMessage(response));
	const text = await response.text();
	return (text ? JSON.parse(text) : null) as T;
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
	// NOTE: the desktop command picked files via a native dialog server-side;
	// the HTTP route instead expects explicit file paths, which this browser
	// client has no way to obtain from a bare `PageImportSource`. Kept for
	// call-site compatibility — see Phase 3b report for the tracked gap.
	importPages: (_source: PageImportSource) => post<null>("/pages/import", { files: [] }),
	selectPage: (page: EntityId) => post<PageSelection>("/page/select", { page }),
	renamePage: (page: EntityId, label: string) => post<null>("/page/rename", { page, label }),
	deletePages: (pages: EntityId[]) => post<null>("/pages/delete", { pages }),
	movePage: (page: EntityId, index: number) => post<null>("/page/move", { page, index }),
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
	undo: () => post<null>("/project/undo"),
	redo: () => post<null>("/project/redo"),
	process: (scope: Scope, operation: Operation) => post<JobId>("/process", { scope, operation }),
	stopJob: (job: JobId) => post<null>("/process/stop", { job }),
	// NOTE: the desktop command picked the destination via a native save
	// dialog; the HTTP route instead expects an explicit target directory.
	// `directory` is optional here only to keep existing 2-arg call sites
	// compiling — see Phase 3b report for the tracked gap.
	exportPages: (pages: EntityId[], format: ExportFormat, directory = "") =>
		post<null>("/pages/export", { pages, format, directory }),
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

export type AtlasCloudConfig = Record<string, never>;

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

export type Config = {
	model: string | null,
	reasoning: Reasoning,
};

export type CredentialInput = {
	configured: boolean,
	value: string | null,
	clear: boolean,
};

export type DeepLConfig = {
	base_url?: string | null,
};

export type DeepSeekConfig = Record<string, never>;

export type DetectionModel = {
	model: "koharu-layout-rfdetr-seg-2xl",
} & KoharuLayoutRFDetrSeg2XLConfig;

export type DeviceResources = {
	name: string,
	selected: boolean,
	memory_budget: number | null,
	memory_used: number | null,
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

export type ExportFormat = "png" | "psd";

export type Flux2KleinConfig = {
	prompt?: string,
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
	thinking?: boolean,
};

export type Geometry = {
	points: Point[],
};

export type GeometryUpdate = {
	layer: EntityId,
	points: Point[] | null,
};

export type GoogleCloudConfig = Record<string, never>;

export type GrokConfig = Record<string, never>;

export type GroupRole = "text";

export type InpaintingModel = { model: "lama" } | { model: "aot-inpainting" } | {
	model: "flux2-klein",
} & Flux2KleinConfig | {
	model: "rorem-mixed",
} & RoremMixedConfig;

export type Job = {
	id: JobId,
	state: JobState,
	completed: number,
	total: number,
	page: EntityId | null,
	stage: Stage | null,
	model: string | null,
	error: string | null,
};

export type JobId = string;

export type JobState = "running" | "finished" | "failed" | "stopped";

export type KoharuLayoutRFDetrSeg2XLConfig = {
	text_threshold?: number | null,
	bubble_threshold?: number | null,
	panel_threshold?: number | null,
};

export type LanguageChoice = {
	tag: string,
	name: string,
};

export type Layer = { type: "group"; id: EntityId; parent: EntityId | null; visibility: LayerVisibility; name: string; role: GroupRole | null } | { type: "text"; id: EntityId; parent: EntityId | null; geometry: Geometry | null; visibility: LayerVisibility; content: TextContent; typography: Typography | null; layout: TextLayoutKind; automatic_region: EntityId | null } | { type: "raster"; id: EntityId; parent: EntityId | null; visibility: LayerVisibility; image: string | null; name: string; kind: RasterLayerKind } | { type: "image"; id: EntityId; parent: EntityId | null; geometry: Geometry; visibility: LayerVisibility; image: string } | { type: "artwork"; id: EntityId; parent: EntityId | null; geometry: Geometry; visibility: LayerVisibility; image: string };

export type LayerCommit = {
	revision: Revision,
	layer: EntityId,
};

export type LayerVisibility = {
	visible: boolean,
	opacity: number,
};

export type LmStudioConfig = {
	base_url?: string | null,
};

export type LocalConfig = Record<string, never>;

export type LoginEvent = { type: "progress"; message: string } | { type: "device_code"; verification_url: string; user_code: string };

export type MiniMaxConfig = Record<string, never>;

export type Model = {
	provider: Provider,
	model: string | null,
	name: string,
	quantizations: Quantization[],
	vision: boolean,
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
	vision: boolean,
};

export type OcrModel = { model: "paddleocr-vl-1.6" } | { model: "manga-ocr" } | { model: "baberu-ocr" };

export type OpenAiCompatibleConfig = {
	base_url?: string | null,
	vision?: boolean,
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
	layer_count: number,
};

export type PaintBrush = {
	diameter: number,
	color: [number, number, number, number],
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

export type ProcessorConfig = {
	"koharu-layout-rfdetr-seg-2xl"?: KoharuLayoutRFDetrSeg2XLConfig | null,
	"flux2-klein"?: Flux2KleinConfig | null,
	"rorem-mixed"?: RoremMixedConfig | null,
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

export type Provider = "local" | "atlas-cloud" | "openai" | "gemini" | "claude" | "grok" | "minimax" | "deepseek" | "openai-compatible" | "openrouter" | "lm-studio" | "deepl" | "google-cloud-translation" | "caiyun";

export type ProviderConfig = { provider: "local"; settings: LocalConfig } | { provider: "atlas-cloud"; settings: AtlasCloudConfig } | { provider: "openai"; settings: OpenAiConfig } | { provider: "gemini"; settings: GeminiConfig } | { provider: "claude"; settings: ClaudeConfig } | { provider: "grok"; settings: GrokConfig } | { provider: "minimax"; settings: MiniMaxConfig } | { provider: "deepseek"; settings: DeepSeekConfig } | { provider: "openai-compatible"; settings: OpenAiCompatibleConfig } | { provider: "openrouter"; settings: OpenRouterConfig } | { provider: "lm-studio"; settings: LmStudioConfig } | { provider: "deepl"; settings: DeepLConfig } | { provider: "google-cloud-translation"; settings: GoogleCloudConfig } | { provider: "caiyun"; settings: CaiyunConfig };

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
};

export type RasterLayerKind = "cleanup" | "paint";

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

export type WritingMode = "Horizontal" | "Vertical";
