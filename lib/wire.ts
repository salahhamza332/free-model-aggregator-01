// Client-wire translation and upstream-wire plumbing for the aggregator.
//
// The client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), JSON or SSE. Any of the ten providers (openai, anthropic or gemini wire) can
// serve, and this file converts a client request into the provider's wire, and the provider's answer
// back into the client's wire. Everything here is a pure function of its arguments or of the provider's
// answer: no clock, no randomness.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A data: URL's media type and raw base64 payload, or null when it is not one. */
export function splitDataUrl(url: string): { media_type: string; data: string } | null {
  const match = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (match === null) return null;
  return { media_type: match[1] !== '' ? match[1] : 'image/png', data: match[2] };
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
export function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** Canonical stop family: 0 stop, 1 length, 2 tool_calls, 3 content_filter. */
export type StopFamily = 'stop' | 'length' | 'tool_calls' | 'content_filter';

export function stopFamilyOfOpenAI(finish: unknown): StopFamily {
  if (finish === 'length') return 'length';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_calls';
  if (finish === 'content_filter') return 'content_filter';
  return 'stop';
}

export function stopFamilyOfAnthropic(reason: unknown): StopFamily {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

export function stopFamilyOfGemini(reason: unknown): StopFamily {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY' || reason === 'PROHIBITED_CONTENT' || reason === 'BLOCKLIST') return 'content_filter';
  return 'stop';
}

export function openAIFinishOf(family: StopFamily): string {
  return family === 'stop' ? 'stop' : family === 'length' ? 'length' : family === 'tool_calls' ? 'tool_calls' : 'content_filter';
}

export function anthropicStopOf(family: StopFamily): string {
  return family === 'stop' ? 'end_turn' : family === 'length' ? 'max_tokens' : family === 'tool_calls' ? 'tool_use' : 'refusal';
}

/** What the entry reports back about an upstream answer, in one neutral shape. */
export interface CanonicalAnswer {
  text: string;
  toolCalls: { id: string; name: string; arguments: Json }[];
  usage: { input: number; output: number } | null;
  stop: StopFamily;
}

export interface CanonicalChunk {
  text?: string;
  toolCalls?: { index: number; id?: string; name?: string; argumentsDelta?: string }[];
  usage?: { input: number; output: number } | null;
  stop?: StopFamily | null;
}

function openAIToolCallsToCanonical(calls: unknown): CanonicalAnswer['toolCalls'] {
  const out: CanonicalAnswer['toolCalls'] = [];
  if (!Array.isArray(calls)) return out;
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    out.push({
      id: typeof call.id === 'string' && call.id !== '' ? call.id : `call_${out.length}`,
      name: stringOf(call.function.name),
      arguments: argumentsOf(call.function.arguments),
    });
  }
  return out;
}

/** One Chat Completions JSON answer as the canonical answer. */
export function canonicalFromOpenAIJson(body: Json): CanonicalAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const usage = isRecord(body.usage) ? { input: numberOrZero(body.usage.prompt_tokens), output: numberOrZero(body.usage.completion_tokens) } : null;
  return { text, toolCalls: openAIToolCallsToCanonical(message.tool_calls), usage, stop: stopFamilyOfOpenAI(choice.finish_reason) };
}

/** One Messages JSON answer as the canonical answer. */
export function canonicalFromAnthropicJson(body: Json): CanonicalAnswer {
  const content = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: CanonicalAnswer['toolCalls'] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') {
      toolCalls.push({ id: stringOf(block.id) || `toolu_${toolCalls.length}`, name: stringOf(block.name), arguments: isRecord(block.input) ? block.input : {} });
    }
  }
  const usage = isRecord(body.usage) ? { input: numberOrZero(body.usage.input_tokens), output: numberOrZero(body.usage.output_tokens) } : null;
  return { text, toolCalls, usage, stop: stopFamilyOfAnthropic(body.stop_reason) };
}

/** One Gemini generateContent JSON answer as the canonical answer. */
export function canonicalFromGeminiJson(body: Json): CanonicalAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const toolCalls: CanonicalAnswer['toolCalls'] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    else if (isRecord(part.functionCall)) {
      toolCalls.push({ id: `call_${toolCalls.length}`, name: stringOf(part.functionCall.name), arguments: isRecord(part.functionCall.args) ? part.functionCall.args : {} });
    }
  }
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : null;
  const usage = meta === null ? null : { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) };
  return { text, toolCalls, usage, stop: stopFamilyOfGemini(candidate.finishReason) };
}

/** Parse one SSE body into its frames, tolerating CRLF and blank frames. */
export function parseSseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  for (const chunk of text.replace(/\r\n/g, '\n').split('\n\n')) {
    const trimmed = chunk.trim();
    if (trimmed === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of chunk.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** A Chat Completions SSE answer as the canonical answer. */
export function canonicalFromOpenAISSE(text: string): CanonicalAnswer {
  let out = '';
  const toolCalls: { index: number; id: string; name: string; arguments: string }[] = [];
  let usage: CanonicalAnswer['usage'] = null;
  let stop: StopFamily = 'stop';
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) usage = { input: numberOrZero(chunk.usage.prompt_tokens), output: numberOrZero(chunk.usage.completion_tokens) };
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const index = typeof call.index === 'number' ? call.index : toolCalls.length;
      let slot = toolCalls.find((t) => t.index === index);
      if (slot === undefined) {
        slot = { index, id: '', name: '', arguments: '' };
        toolCalls.push(slot);
      }
      if (typeof call.id === 'string' && call.id !== '') slot.id = call.id;
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.name === 'string' && fn.name !== '') slot.name = fn.name;
      if (typeof fn.arguments === 'string') slot.arguments += fn.arguments;
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) stop = stopFamilyOfOpenAI(choice.finish_reason);
  }
  return {
    text: out,
    toolCalls: toolCalls.map((t, i) => ({ id: t.id !== '' ? t.id : `call_${i}`, name: t.name, arguments: argumentsOf(t.arguments) })),
    usage,
    stop,
  };
}

/** An Anthropic Messages SSE answer as the canonical answer. */
export function canonicalFromAnthropicSSE(text: string): CanonicalAnswer {
  let out = '';
  const toolCalls: { index: number; id: string; name: string; arguments: string }[] = [];
  let usage: CanonicalAnswer['usage'] = null;
  let stop: StopFamily = 'stop';
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    switch (frame.event) {
      case 'message_start': {
        const message = isRecord(data.message) ? data.message : {};
        const u = isRecord(message.usage) ? message.usage : {};
        usage = { input: numberOrZero(u.input_tokens), output: usage?.output ?? 0 };
        break;
      }
      case 'content_block_start': {
        const index = numberOrZero(data.index);
        const block = isRecord(data.content_block) ? data.content_block : {};
        if (block.type === 'tool_use') {
          toolCalls.push({ index, id: stringOf(block.id) || `toolu_${index}`, name: stringOf(block.name), arguments: '' });
        }
        break;
      }
      case 'content_block_delta': {
        const index = numberOrZero(data.index);
        const delta = isRecord(data.delta) ? data.delta : {};
        if (delta.type === 'text_delta' && typeof delta.text === 'string') out += delta.text;
        else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string') {
          const slot = toolCalls.find((t) => t.index === index);
          if (slot !== undefined) slot.arguments += delta.partial_json;
        }
        break;
      }
      case 'message_delta': {
        const delta = isRecord(data.delta) ? data.delta : {};
        if (typeof delta.stop_reason === 'string') stop = stopFamilyOfAnthropic(delta.stop_reason);
        const u = isRecord(data.usage) ? data.usage : null;
        if (u !== null) usage = { input: usage?.input ?? 0, output: numberOrZero(u.output_tokens) };
        break;
      }
    }
  }
  return {
    text: out,
    toolCalls: toolCalls.map((t) => ({ id: t.id, name: t.name, arguments: argumentsOf(t.arguments) })),
    usage,
    stop,
  };
}

/** A Gemini streamGenerateContent SSE answer as the canonical answer. */
export function canonicalFromGeminiSSE(text: string): CanonicalAnswer {
  let out = '';
  const toolCalls: CanonicalAnswer['toolCalls'] = [];
  let usage: CanonicalAnswer['usage'] = null;
  let stop: StopFamily = 'stop';
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      const content = isRecord(candidate.content) ? candidate.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const part of parts) {
        if (!isRecord(part)) continue;
        if (typeof part.text === 'string') out += part.text;
        else if (isRecord(part.functionCall)) {
          toolCalls.push({ id: `call_${toolCalls.length}`, name: stringOf(part.functionCall.name), arguments: isRecord(part.functionCall.args) ? part.functionCall.args : {} });
        }
      }
      if (candidate.finishReason !== undefined) stop = stopFamilyOfGemini(candidate.finishReason);
    }
    const meta = isRecord(data.usageMetadata) ? data.usageMetadata : null;
    if (meta !== null) usage = { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) };
  }
  return { text: out, toolCalls, usage, stop };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function toolCallsForOpenAI(answer: CanonicalAnswer): Json[] {
  return answer.toolCalls.map((call) => ({
    id: call.id,
    type: 'function',
    function: { name: call.name, arguments: JSON.stringify(call.arguments) },
  }));
}

/** The canonical answer as a Chat Completions JSON object. */
export function openAIJsonOf(answer: CanonicalAnswer, model: string): Json {
  const toolCalls = toolCallsForOpenAI(answer);
  return {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
        finish_reason: openAIFinishOf(answer.stop),
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: answer.usage?.input ?? 0,
      completion_tokens: answer.usage?.output ?? 0,
      total_tokens: (answer.usage?.input ?? 0) + (answer.usage?.output ?? 0),
    },
  };
}

/** The canonical answer as a Chat Completions SSE stream. */
export function openAISSEOf(answer: CanonicalAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({ id: 'chatcmpl-aggregator', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
  const frames: string[] = [];
  frames.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  if (answer.text !== '') frames.push(sseEvent(null, chunk({ content: answer.text }, null)));
  answer.toolCalls.forEach((call, i) => {
    frames.push(sseEvent(null, chunk({ tool_calls: [{ index: i, id: call.id, type: 'function', function: { name: call.name, arguments: '' } }] }, null)));
    frames.push(sseEvent(null, chunk({ tool_calls: [{ index: i, function: { arguments: JSON.stringify(call.arguments) } }] }, null)));
  });
  frames.push(sseEvent(null, chunk({}, openAIFinishOf(answer.stop))));
  frames.push(sseEvent(null, {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [],
    usage: {
      prompt_tokens: answer.usage?.input ?? 0,
      completion_tokens: answer.usage?.output ?? 0,
      total_tokens: (answer.usage?.input ?? 0) + (answer.usage?.output ?? 0),
    },
  }));
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** The canonical answer as a Messages JSON object. */
export function anthropicJsonOf(answer: CanonicalAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '' || answer.toolCalls.length === 0) content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) content.push({ type: 'tool_use', id: call.id, name: call.name, input: call.arguments });
  return {
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStopOf(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.usage?.input ?? 0, output_tokens: answer.usage?.output ?? 0 },
  };
}

/** The canonical answer as a Messages SSE stream. */
export function anthropicSSEOf(answer: CanonicalAnswer, model: string): string {
  const frames: string[] = [];
  frames.push(sseEvent('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_aggregator',
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: answer.usage?.input ?? 0, output_tokens: 0 },
    },
  }));
  let index = 0;
  if (answer.text !== '' || answer.toolCalls.length === 0) {
    frames.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    if (answer.text !== '') frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: answer.text } }));
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of answer.toolCalls) {
    frames.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} } }));
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: JSON.stringify(call.arguments) } }));
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: anthropicStopOf(answer.stop), stop_sequence: null }, usage: { output_tokens: answer.usage?.output ?? 0 } }));
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}

// ---------- Request-side conversions: build the upstream body in the provider's wire ----------

interface NormalizedMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  text: string;
  images: { dataUrl?: string; mediaType?: string; data?: string; url?: string }[];
  toolCalls: { id: string; name: string; arguments: Json }[];
  toolResults: { tool_call_id: string; content: string }[];
}

function normalizeOpenAIMessages(body: Json): { system: string; messages: NormalizedMessage[] } {
  const systemParts: string[] = [];
  const messages: NormalizedMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'system' ? 'system' : raw.role === 'assistant' ? 'assistant' : raw.role === 'tool' ? 'tool' : 'user';
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    const msg: NormalizedMessage = { role, text: '', images: [], toolCalls: [], toolResults: [] };
    if (typeof raw.content === 'string') msg.text = raw.content;
    else if (Array.isArray(raw.content)) {
      msg.text = textOf(raw.content);
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          const url = block.image_url.url;
          const data = splitDataUrl(url);
          if (data !== null) msg.images.push({ dataUrl: url, mediaType: data.media_type, data: data.data });
          else msg.images.push({ url });
        }
      }
    }
    if (role === 'assistant' && Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        msg.toolCalls.push({ id: stringOf(call.id) || `call_${msg.toolCalls.length}`, name: stringOf(call.function.name), arguments: argumentsOf(call.function.arguments) });
      }
    }
    if (role === 'tool') msg.toolResults.push({ tool_call_id: stringOf(raw.tool_call_id), content: typeof raw.content === 'string' ? raw.content : JSON.stringify(raw.content ?? '') });
    messages.push(msg);
  }
  return { system: systemParts.join('\n'), messages };
}

function normalizeAnthropicMessages(body: Json): { system: string; messages: NormalizedMessage[] } {
  const system = textOf(body.system);
  const messages: NormalizedMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'assistant' : 'user';
    const msg: NormalizedMessage = { role, text: '', images: [], toolCalls: [], toolResults: [] };
    if (typeof raw.content === 'string') msg.text = raw.content;
    else if (Array.isArray(raw.content)) {
      msg.text = textOf(raw.content);
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'image' && isRecord(block.source)) {
          const sourceBlock = block.source;
          if (sourceBlock.type === 'base64' && typeof sourceBlock.data === 'string') {
            msg.images.push({ mediaType: stringOf(sourceBlock.media_type) || 'image/png', data: sourceBlock.data });
          } else if (sourceBlock.type === 'url' && typeof sourceBlock.url === 'string') {
            msg.images.push({ url: sourceBlock.url });
          }
        } else if (block.type === 'tool_use') {
          msg.toolCalls.push({ id: stringOf(block.id) || `toolu_${msg.toolCalls.length}`, name: stringOf(block.name), arguments: isRecord(block.input) ? block.input : {} });
        } else if (block.type === 'tool_result') {
          const content = block.content;
          msg.toolResults.push({ tool_call_id: stringOf(block.tool_use_id), content: typeof content === 'string' ? content : textOf(content) !== '' ? textOf(content) : JSON.stringify(content ?? '') });
        }
      }
    }
    messages.push(msg);
  }
  return { system, messages };
}

function normalizeLegacy(body: Json, api: 'openai' | 'anthropic'): { system: string; messages: NormalizedMessage[] } {
  return api === 'openai' ? normalizeOpenAIMessages(body) : normalizeAnthropicMessages(body);
}

export function isOpenAIJsonMode(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

function openAIToolsFrom(tools: unknown): { name: string; description: string; parameters: Json }[] {
  const out: { name: string; description: string; parameters: Json }[] = [];
  if (!Array.isArray(tools)) return out;
  for (const raw of tools) {
    if (!isRecord(raw)) continue;
    if (isRecord(raw.function)) {
      out.push({ name: stringOf(raw.function.name), description: stringOf(raw.function.description), parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } });
    } else if (typeof raw.name === 'string') {
      out.push({ name: raw.name, description: stringOf(raw.description), parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } });
    }
  }
  return out;
}

function normalizedFrom(body: Json, api: 'openai' | 'anthropic'): { system: string; messages: NormalizedMessage[] } {
  return normalizeLegacy(body, api);
}

function stopStringsOf(body: Json, api: 'openai' | 'anthropic'): string[] {
  const raw = api === 'openai' ? body.stop : body.stop_sequences;
  if (typeof raw === 'string') return [raw];
  if (Array.isArray(raw)) return raw.filter((x): x is string => typeof x === 'string');
  return [];
}

function maxTokensOf(body: Json, api: 'openai' | 'anthropic'): number | null {
  const raw = api === 'openai' ? (body.max_tokens ?? body.max_completion_tokens) : body.max_tokens;
  return numberOrNull(raw);
}

function temperatureOf(body: Json): number | null {
  return numberOrNull(body.temperature);
}

function topPOf(body: Json): number | null {
  return numberOrNull(body.top_p);
}

export interface UpstreamRequest {
  url: string;
  headers: Record<string, string>;
  body: string;
}

/** The upstream request for an OpenAI-wire provider. */
export function buildOpenAIUpstream(body: Json, api: 'openai' | 'anthropic', model: string, base: string, key: string, auth: string, stream: boolean): UpstreamRequest {
  let out: Json;
  if (api === 'openai') {
    out = { ...body, model };
  } else {
    out = anthropicToOpenAI(body, model);
  }
  if (stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  } else {
    delete out.stream;
    delete out.stream_options;
  }
  return {
    url: `${base}/v1/chat/completions`,
    headers: { 'content-type': 'application/json', ...authHeadersFor(auth, key), ...(stream ? { accept: 'text/event-stream' } : {}) },
    body: JSON.stringify(out),
  };
}

function authHeadersFor(auth: string, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: 'Bearer ' + key };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** The upstream request for an Anthropic-wire provider. */
export function buildAnthropicUpstream(body: Json, api: 'openai' | 'anthropic', model: string, base: string, key: string, auth: string, stream: boolean): UpstreamRequest {
  let out: Json;
  if (api === 'anthropic') {
    out = { ...body, model };
  } else {
    const { system, messages } = normalizedFrom(body, 'openai');
    const outMessages: Json[] = [];
    for (const message of messages) {
      const content: Json[] = [];
      if (message.text !== '') content.push({ type: 'text', text: message.text });
      for (const image of message.images) {
        if (image.data !== undefined) content.push({ type: 'image', source: { type: 'base64', media_type: image.mediaType ?? 'image/png', data: image.data } });
        else if (image.url !== undefined) content.push({ type: 'image', source: { type: 'url', url: image.url } });
      }
      if (message.role === 'assistant') {
        for (const call of message.toolCalls) content.push({ type: 'tool_use', id: call.id, name: call.name, input: call.arguments });
        outMessages.push({ role: 'assistant', content: content.length > 0 ? content : [{ type: 'text', text: '' }] });
        continue;
      }
      for (const result of message.toolResults) content.push({ type: 'tool_result', tool_use_id: result.tool_call_id, content: result.content });
      outMessages.push({ role: 'user', content: content.length > 0 ? content : [{ type: 'text', text: '' }] });
    }
    out = { model, messages: outMessages };
    if (system !== '') out.system = system;
    const tools = openAIToolsFrom(body.tools).map((tool) => ({ name: tool.name, description: tool.description, input_schema: tool.parameters }));
    if (tools.length > 0) out.tools = tools;
    const choice = body.tool_choice;
    if (choice === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice === 'required') out.tool_choice = { type: 'any' };
    else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function)) out.tool_choice = { type: 'tool', name: stringOf(choice.function.name) };
    const maxTokens = maxTokensOf(body, 'openai');
    if (maxTokens !== null) out.max_tokens = maxTokens;
    const stops = stopStringsOf(body, 'openai');
    if (stops.length > 0) out.stop_sequences = stops;
    const temperature = temperatureOf(body);
    if (temperature !== null) out.temperature = temperature;
    const topP = topPOf(body);
    if (topP !== null) out.top_p = topP;
  }
  if (stream) out.stream = true;
  else delete out.stream;
  return {
    url: `${base}/v1/messages`,
    headers: { 'content-type': 'application/json', 'anthropic-version': '2023-06-01', ...authHeadersFor(auth, key), ...(stream ? { accept: 'text/event-stream' } : {}) },
    body: JSON.stringify(out),
  };
}

/** The upstream request for a Gemini-wire provider. */
export function buildGeminiUpstream(body: Json, api: 'openai' | 'anthropic', model: string, base: string, key: string, auth: string, stream: boolean): UpstreamRequest {
  const { system, messages } = normalizedFrom(body, api);
  const toolNameById = new Map<string, string>();
  for (const message of messages) for (const call of message.toolCalls) toolNameById.set(call.id, call.name);
  const contents: Json[] = [];
  for (const message of messages) {
    const parts: Json[] = [];
    if (message.role === 'tool') {
      for (const result of message.toolResults) parts.push({ functionResponse: { name: toolNameById.get(result.tool_call_id) ?? result.tool_call_id, response: { result: result.content } } });
      contents.push({ role: 'user', parts });
      continue;
    }
    if (message.text !== '') parts.push({ text: message.text });
    for (const image of message.images) {
      if (image.data !== undefined) parts.push({ inlineData: { mimeType: image.mediaType ?? 'image/png', data: image.data } });
      else if (image.url !== undefined) parts.push({ fileData: { fileUri: image.url } });
    }
    for (const call of message.toolCalls) parts.push({ functionCall: { name: call.name, args: call.arguments } });
    for (const result of message.toolResults) parts.push({ functionResponse: { name: toolNameById.get(result.tool_call_id) ?? result.tool_call_id, response: { result: result.content } } });
    if (parts.length === 0) parts.push({ text: '' });
    contents.push({ role: message.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarationSource = api === 'openai' ? openAIToolsFrom(body.tools) : (Array.isArray(body.tools) ? body.tools : []).flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [{ name: raw.name, description: stringOf(raw.description), parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
  });
  if (declarationSource.length > 0) out.tools = [{ functionDeclarations: declarationSource.map((tool) => ({ name: tool.name, description: tool.description, parameters: tool.parameters })) }];
  const generationConfig: Json = {};
  const maxTokens = maxTokensOf(body, api);
  if (maxTokens !== null) generationConfig.maxOutputTokens = maxTokens;
  const stops = stopStringsOf(body, api);
  if (stops.length > 0) generationConfig.stopSequences = stops;
  const temperature = temperatureOf(body);
  if (temperature !== null) generationConfig.temperature = temperature;
  const topP = topPOf(body);
  if (topP !== null) generationConfig.topP = topP;
  if (api === 'openai' && isOpenAIJsonMode(body)) generationConfig.responseMimeType = 'application/json';
  if (Object.keys(generationConfig).length > 0) out.generationConfig = generationConfig;
  const suffix = stream ? 'streamGenerateContent?alt=sse' : 'generateContent';
  return {
    url: `${base}/v1beta/models/${encodeURIComponent(model)}:${suffix}`,
    headers: { 'content-type': 'application/json', ...authHeadersFor(auth, key), ...(stream ? { accept: 'text/event-stream' } : {}) },
    body: JSON.stringify(out),
  };
}

/** An Anthropic Messages request as an OpenAI Chat Completions request that names `model`. */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const { system, messages } = normalizeAnthropicMessages(body);
  const outMessages: Json[] = [];
  if (system !== '') outMessages.push({ role: 'system', content: system });
  for (const message of messages) {
    if (message.role === 'assistant') {
      const toolCalls = message.toolCalls.map((call) => ({ id: call.id, type: 'function', function: { name: call.name, arguments: JSON.stringify(call.arguments) } }));
      outMessages.push({ role: 'assistant', content: message.text !== '' ? message.text : toolCalls.length > 0 ? null : '', ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    for (const result of message.toolResults) outMessages.push({ role: 'tool', tool_call_id: result.tool_call_id, content: result.content });
    const parts: Json[] = [];
    if (message.text !== '') parts.push({ type: 'text', text: message.text });
    for (const image of message.images) {
      if (image.data !== undefined) parts.push({ type: 'image_url', image_url: { url: `data:${image.mediaType ?? 'image/png'};base64,${image.data}` } });
      else if (image.url !== undefined) parts.push({ type: 'image_url', image_url: { url: image.url } });
    }
    if (parts.length > 0) outMessages.push({ role: 'user', content: parts.length === 1 && parts[0].type === 'text' ? message.text : parts });
  }
  const out: Json = { model, messages: outMessages };
  const maxTokens = maxTokensOf(body, 'anthropic');
  if (maxTokens !== null) out.max_tokens = maxTokens;
  const temperature = temperatureOf(body);
  if (temperature !== null) out.temperature = temperature;
  const topP = topPOf(body);
  if (topP !== null) out.top_p = topP;
  const stops = stopStringsOf(body, 'anthropic');
  if (stops.length > 0) out.stop = stops;
  const tools = openAIToolsFrom(body.tools).map((tool) => ({ type: 'function', function: { name: tool.name, description: tool.description, parameters: tool.parameters } }));
  if (tools.length > 0) out.tools = tools;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object (non-stream clients only). */
export function openAIToAnthropic(body: Json, model: string): Json {
  return anthropicJsonOf(canonicalFromOpenAIJson(body), model);
}

/** An Anthropic Messages SSE answer as the canonical answer (kept for compatibility). */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
  }
  buffer += decoder.decode();
  yield anthropicSSEOf(canonicalFromOpenAISSE(buffer), model);
}

/** An OpenAI Chat Completions answer as an Anthropic Messages SSE stream body. */
export function anthropicSSEFromOpenAI(upstreamText: string, model: string): string {
  return anthropicSSEOf(canonicalFromOpenAISSE(upstreamText), model);
}
