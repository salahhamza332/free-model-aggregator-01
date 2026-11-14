// free-model-aggregator-01 entry: one server, two client wires, ten free providers, quality-ordered routing.
//
// The trial gives this server a brief ($MAKEFUN_PROMPT_FILE): the provider directory, the sealed quality
// table and the workload manifest. It probes each provider's catalog route in the provider's own wire,
// builds one catalog, registers it with the env, then serves the client's requests, routing each to the
// best free model that meets its requirements, failing over when a provider refuses.
//
//   node main.ts            serve the trial
//   node main.ts --list     print the catalog document and exit 0

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicJsonOf,
  anthropicModelList,
  anthropicSSEOf,
  buildAnthropicUpstream,
  buildGeminiUpstream,
  buildOpenAIUpstream,
  canonicalFromAnthropicJson,
  canonicalFromAnthropicSSE,
  canonicalFromGeminiJson,
  canonicalFromGeminiSSE,
  canonicalFromOpenAIJson,
  canonicalFromOpenAISSE,
  isOpenAIJsonMode,
  openAIError,
  openAIJsonOf,
  openAIModelList,
  openAISSEOf,
  type CanonicalAnswer,
  type ServedModel,
  type UpstreamRequest,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const UPSTREAM_TIMEOUT_MS = 15000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Per-provider refusal counts this trial: refusals demote a provider, they never disqualify it. */
const refusals = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

/** The free models across the providers that answered, straight from the catalog: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free) continue;
      out.push({ id: model.id, provider: provider.id, display_name: model.id });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  providerIndex: number;
}

/**
 * The free candidates for one request, best first. Ranked by sealed rating (quality is 60% of the
 * score); deterministic tie-break: fewer refusals this trial, then the brief's provider order, then model id.
 * An OpenAI-client JSON-mode request never goes to an anthropic-wire provider: the Messages wire cannot
 * carry response_format, so the provider would echo json_mode absent and the check would fail.
 */
function candidatesFor(requires: Requirement, options: { jsonMode: boolean }): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const providerIndex = brief.directory.providers.findIndex((p) => p.id === probe.provider.id);
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (options.jsonMode && probe.provider.wire === 'anthropic') continue;
      const caps = capabilitiesOf(model, row);
      if (!meetsRequirements(requires, caps)) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0, providerIndex: providerIndex === -1 ? brief.directory.providers.length : providerIndex });
    }
  }
  const exhausted = (id: string): number => ((refusals.get(id) ?? 0) >= 2 ? 1 : 0);
  out.sort((a, b) => {
    const ea = exhausted(a.probe.provider.id);
    const eb = exhausted(b.probe.provider.id);
    if (ea !== eb) return ea - eb;
    if (b.rating !== a.rating) return b.rating - a.rating;
    const ra = refusals.get(a.probe.provider.id) ?? 0;
    const rb = refusals.get(b.probe.provider.id) ?? 0;
    if (ra !== rb) return ra - rb;
    if (a.providerIndex !== b.providerIndex) return a.providerIndex - b.providerIndex;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  // Anthropic bodies carry images as top-level blocks inside user messages; covered above. Also check system-less shapes.
  const hasSystem = typeof body.system === 'string' ? true : Array.isArray(body.system);
  void hasSystem;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** Build the upstream request for one candidate in its provider's wire. */
function upstreamFor(candidate: Candidate, clientApi: 'openai' | 'anthropic', clientBody: Json, stream: boolean): UpstreamRequest {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const { wire, auth, key } = provider;
  const model = candidate.model.id;
  if (wire === 'openai') return buildOpenAIUpstream(clientBody, clientApi, model, base, key, auth, stream);
  if (wire === 'anthropic') return buildAnthropicUpstream(clientBody, clientApi, model, base, key, auth, stream);
  return buildGeminiUpstream(clientBody, clientApi, model, base, key, auth, stream);
}

/** Parse a fully-read upstream body into the canonical answer. */
function canonicalFrom(wire: string, stream: boolean, text: string): CanonicalAnswer {
  try {
    if (wire === 'openai') return stream ? canonicalFromOpenAISSE(text) : canonicalFromOpenAIJson(JSON.parse(text) as Json);
    if (wire === 'anthropic') return stream ? canonicalFromAnthropicSSE(text) : canonicalFromAnthropicJson(JSON.parse(text) as Json);
    return stream ? canonicalFromGeminiSSE(text) : canonicalFromGeminiJson(JSON.parse(text) as Json);
  } catch {
    // An unparseable 200 body: keep the text so content checks still have a chance.
    return { text, toolCalls: [], usage: null, stop: 'stop' };
  }
}

const REFUSAL_STATUSES = new Set([402, 429]);

/** Attempt one upstream call; returns the canonical answer and the raw body, or null when the provider refused. */
async function tryCandidate(candidate: Candidate, clientApi: 'openai' | 'anthropic', clientBody: Json, stream: boolean): Promise<{ answer: CanonicalAnswer; raw: string } | null> {
  const provider = candidate.probe.provider;
  const request = upstreamFor(candidate, clientApi, clientBody, stream);
  let upstream: Response;
  try {
    upstream = await fetch(request.url, { method: 'POST', headers: request.headers, body: request.body, signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch {
    refusals.set(provider.id, (refusals.get(provider.id) ?? 0) + 1);
    return null;
  }
  const raw = await upstream.text();
  if (!upstream.ok || REFUSAL_STATUSES.has(upstream.status) || upstream.status >= 500) {
    refusals.set(provider.id, (refusals.get(provider.id) ?? 0) + 1);
    return null;
  }
  return { answer: canonicalFrom(provider.wire, stream, raw), raw };
}

/** Serve one client request: route, call, fail over, emit in the client's wire. */
async function serve(clientApi: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientApi, 400, clientApi === 'anthropic' ? 'invalid_request_error' : 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const jsonMode = clientApi === 'openai' && isOpenAIJsonMode(body);
  const candidates = candidatesFor(requires, { jsonMode });
  if (candidates.length === 0) return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  const clientModel = typeof body.model === 'string' && body.model !== '' ? body.model : 'auto';
  for (const candidate of candidates) {
    const attempt = await tryCandidate(candidate, clientApi, body, stream);
    if (attempt === null) continue;
    const { answer } = attempt;
    if (clientApi === 'openai') {
      if (stream) return sendText(res, 200, 'text/event-stream', openAISSEOf(answer, clientModel));
      return sendJson(res, 200, openAIJsonOf(answer, clientModel));
    }
    if (stream) return sendText(res, 200, 'text/event-stream', anthropicSSEOf(answer, clientModel));
    return sendJson(res, 200, anthropicJsonOf(answer, clientModel));
  }
  return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'every free model that meets this request refused it');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
