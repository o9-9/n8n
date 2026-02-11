import { AssistantHandler } from '../assistant-handler';
import type { AssistantContext, AssistantSdkClient, SdkStreamChunk, StreamWriter } from '../types';
import type { AgentMessageChunk, StreamChunk, ToolProgressChunk } from '../../types/streaming';

/** Same separator used by the backend stream protocol */
const STREAM_SEPARATOR = '⧉⇋⇋➽⌑⧉§§\n';

/**
 * Create a mock Response with a ReadableStream body encoding the given chunks
 * as JSON separated by STREAM_SEPARATOR.
 */
function createMockSdkResponse(chunks: SdkStreamChunk[], ok = true, status = 200): Response {
	const encoder = new TextEncoder();
	const encoded = chunks.map((c) => JSON.stringify(c) + STREAM_SEPARATOR);

	const stream = new ReadableStream<Uint8Array>({
		start(controller) {
			for (const segment of encoded) {
				controller.enqueue(encoder.encode(segment));
			}
			controller.close();
		},
	});

	return {
		ok,
		status,
		body: stream,
	} as unknown as Response;
}

/**
 * Create a mock Response where the body is split into arbitrary byte boundaries
 * to test incomplete JSON buffering.
 */
function createSplitResponse(chunks: SdkStreamChunk[], splitAt: number): Response {
	const encoder = new TextEncoder();
	const fullString = chunks.map((c) => JSON.stringify(c) + STREAM_SEPARATOR).join('');
	const fullBytes = encoder.encode(fullString);

	const stream = new ReadableStream<Uint8Array>({
		start(controller) {
			// Split at the given byte index
			controller.enqueue(fullBytes.slice(0, splitAt));
			controller.enqueue(fullBytes.slice(splitAt));
			controller.close();
		},
	});

	return { ok: true, status: 200, body: stream } as unknown as Response;
}

function createMockClient(response: Response): AssistantSdkClient {
	return {
		chat: jest.fn().mockResolvedValue(response),
	};
}

describe('AssistantHandler', () => {
	let writtenChunks: StreamChunk[];
	let writer: StreamWriter;

	beforeEach(() => {
		writtenChunks = [];
		writer = (chunk) => writtenChunks.push(chunk);
	});

	// -----------------------------------------------------------------------
	// Test 1: Basic text message
	// -----------------------------------------------------------------------
	it('should map SDK text message to AgentMessageChunk', async () => {
		const response = createMockSdkResponse([
			{
				sessionId: 'sess-1',
				messages: [{ role: 'assistant', type: 'message', text: 'Hello, how can I help?' }],
			},
		]);
		const client = createMockClient(response);
		const handler = new AssistantHandler(client);

		const result = await handler.execute({ query: 'Help me' }, 'user-1', writer);

		expect(writtenChunks).toHaveLength(1);
		const chunk = writtenChunks[0] as AgentMessageChunk;
		expect(chunk.role).toBe('assistant');
		expect(chunk.type).toBe('message');
		expect(chunk.text).toBe('Hello, how can I help?');
		expect(result.responseText).toBe('Hello, how can I help?');
	});

	// -----------------------------------------------------------------------
	// Test 2: Session ID tracking
	// -----------------------------------------------------------------------
	it('should track sessionId from first chunk', async () => {
		const response = createMockSdkResponse([
			{
				sessionId: 'sess-abc',
				messages: [{ role: 'assistant', type: 'message', text: 'Hi' }],
			},
			{
				sessionId: 'sess-xyz',
				messages: [{ role: 'assistant', type: 'message', text: 'More' }],
			},
		]);
		const client = createMockClient(response);
		const handler = new AssistantHandler(client);

		const result = await handler.execute({ query: 'test' }, 'user-1', writer);

		expect(result.sdkSessionId).toBe('sess-abc');
	});

	// -----------------------------------------------------------------------
	// Test 3: Multiple stream chunks
	// -----------------------------------------------------------------------
	it('should process multiple stream chunks', async () => {
		const response = createMockSdkResponse([
			{
				sessionId: 'sess-1',
				messages: [{ role: 'assistant', type: 'message', text: 'First' }],
			},
			{
				sessionId: 'sess-1',
				messages: [{ role: 'assistant', type: 'message', text: 'Second' }],
			},
		]);
		const client = createMockClient(response);
		const handler = new AssistantHandler(client);

		await handler.execute({ query: 'test' }, 'user-1', writer);

		expect(writtenChunks).toHaveLength(2);
		expect((writtenChunks[0] as AgentMessageChunk).text).toBe('First');
		expect((writtenChunks[1] as AgentMessageChunk).text).toBe('Second');
	});

	// -----------------------------------------------------------------------
	// Test 4: code-diff degradation
	// -----------------------------------------------------------------------
	it('should degrade code-diff to AgentMessageChunk with fenced code block', async () => {
		const response = createMockSdkResponse([
			{
				sessionId: 'sess-1',
				messages: [
					{
						role: 'assistant',
						type: 'code-diff',
						description: 'Here is the fix',
						codeDiff: '- old\n+ new',
						suggestionId: 'sug-1',
					},
				],
			},
		]);
		const client = createMockClient(response);
		const handler = new AssistantHandler(client);

		const result = await handler.execute({ query: 'fix it' }, 'user-1', writer);

		expect(writtenChunks).toHaveLength(1);
		const chunk = writtenChunks[0] as AgentMessageChunk;
		expect(chunk.type).toBe('message');
		expect(chunk.text).toContain('Here is the fix');
		expect(chunk.text).toContain('```diff');
		expect(chunk.text).toContain('- old\n+ new');
		expect(result.hasCodeDiff).toBe(true);
		expect(result.suggestionIds).toContain('sug-1');
	});

	// -----------------------------------------------------------------------
	// Test 5: summary degradation
	// -----------------------------------------------------------------------
	it('should degrade summary message to markdown text', async () => {
		const response = createMockSdkResponse([
			{
				sessionId: 'sess-1',
				messages: [
					{
						role: 'assistant',
						type: 'summary',
						title: 'Summary Title',
						content: 'Summary content here',
					},
				],
			},
		]);
		const client = createMockClient(response);
		const handler = new AssistantHandler(client);

		await handler.execute({ query: 'test' }, 'user-1', writer);

		expect(writtenChunks).toHaveLength(1);
		const chunk = writtenChunks[0] as AgentMessageChunk;
		expect(chunk.text).toBe('**Summary Title**\n\nSummary content here');
	});

	// -----------------------------------------------------------------------
	// Test 6: agent-suggestion degradation
	// -----------------------------------------------------------------------
	it('should degrade agent-suggestion to markdown text', async () => {
		const response = createMockSdkResponse([
			{
				sessionId: 'sess-1',
				messages: [
					{
						role: 'assistant',
						type: 'agent-suggestion',
						title: 'Try this',
						text: 'Use the HTTP node instead',
						suggestionId: 'sug-2',
					},
				],
			},
		]);
		const client = createMockClient(response);
		const handler = new AssistantHandler(client);

		const result = await handler.execute({ query: 'test' }, 'user-1', writer);

		expect(writtenChunks).toHaveLength(1);
		const chunk = writtenChunks[0] as AgentMessageChunk;
		expect(chunk.text).toBe('**Try this**\n\nUse the HTTP node instead');
		expect(result.suggestionIds).toContain('sug-2');
	});

	// -----------------------------------------------------------------------
	// Test 7: intermediate-step → ToolProgressChunk
	// -----------------------------------------------------------------------
	it('should map intermediate-step to ToolProgressChunk', async () => {
		const response = createMockSdkResponse([
			{
				sessionId: 'sess-1',
				messages: [
					{
						role: 'assistant',
						type: 'intermediate-step',
						text: 'Searching documentation...',
						step: 'n8n_documentation',
					},
				],
			},
		]);
		const client = createMockClient(response);
		const handler = new AssistantHandler(client);

		await handler.execute({ query: 'test' }, 'user-1', writer);

		expect(writtenChunks).toHaveLength(1);
		const chunk = writtenChunks[0] as ToolProgressChunk;
		expect(chunk.type).toBe('tool');
		expect(chunk.toolName).toBe('assistant');
		expect(chunk.status).toBe('Searching documentation...');
	});

	// -----------------------------------------------------------------------
	// Test 8: event silently consumed
	// -----------------------------------------------------------------------
	it('should silently consume event messages without writing', async () => {
		const response = createMockSdkResponse([
			{
				sessionId: 'sess-1',
				messages: [{ type: 'event', eventName: 'end-session' }],
			},
		]);
		const client = createMockClient(response);
		const handler = new AssistantHandler(client);

		await handler.execute({ query: 'test' }, 'user-1', writer);

		expect(writtenChunks).toHaveLength(0);
	});

	// -----------------------------------------------------------------------
	// Test 9: error → AgentMessageChunk
	// -----------------------------------------------------------------------
	it('should map error message to AgentMessageChunk', async () => {
		const response = createMockSdkResponse([
			{
				sessionId: 'sess-1',
				messages: [
					{
						role: 'assistant',
						type: 'error',
						text: 'Something went wrong',
					},
				],
			},
		]);
		const client = createMockClient(response);
		const handler = new AssistantHandler(client);

		await handler.execute({ query: 'test' }, 'user-1', writer);

		expect(writtenChunks).toHaveLength(1);
		const chunk = writtenChunks[0] as AgentMessageChunk;
		expect(chunk.type).toBe('message');
		expect(chunk.text).toBe('Something went wrong');
	});

	// -----------------------------------------------------------------------
	// Test 10: Empty text filtered
	// -----------------------------------------------------------------------
	it('should not write chunks for empty text messages', async () => {
		const response = createMockSdkResponse([
			{
				sessionId: 'sess-1',
				messages: [{ role: 'assistant', type: 'message', text: '' }],
			},
		]);
		const client = createMockClient(response);
		const handler = new AssistantHandler(client);

		await handler.execute({ query: 'test' }, 'user-1', writer);

		expect(writtenChunks).toHaveLength(0);
	});

	// -----------------------------------------------------------------------
	// Test 11: Text collection & summary truncation
	// -----------------------------------------------------------------------
	it('should collect all text and truncate summary to 200 chars', async () => {
		const longText = 'A'.repeat(150);
		const response = createMockSdkResponse([
			{
				sessionId: 'sess-1',
				messages: [
					{ role: 'assistant', type: 'message', text: longText },
					{ role: 'assistant', type: 'message', text: longText },
				],
			},
		]);
		const client = createMockClient(response);
		const handler = new AssistantHandler(client);

		const result = await handler.execute({ query: 'test' }, 'user-1', writer);

		expect(result.responseText).toBe(`${longText}\n${longText}`);
		expect(result.summary.length).toBeLessThanOrEqual(203); // 200 + '...'
		expect(result.summary.endsWith('...')).toBe(true);
	});

	// -----------------------------------------------------------------------
	// Test 12: SDK HTTP error
	// -----------------------------------------------------------------------
	it('should throw on non-ok SDK response', async () => {
		const response = createMockSdkResponse([], false, 500);
		const client = createMockClient(response);
		const handler = new AssistantHandler(client);

		await expect(handler.execute({ query: 'test' }, 'user-1', writer)).rejects.toThrow(
			'Assistant SDK returned HTTP 500',
		);
	});

	// -----------------------------------------------------------------------
	// Test 13: Null response body
	// -----------------------------------------------------------------------
	it('should throw when response body is null', async () => {
		const response = { ok: true, status: 200, body: null } as unknown as Response;
		const client = createMockClient(response);
		const handler = new AssistantHandler(client);

		await expect(handler.execute({ query: 'test' }, 'user-1', writer)).rejects.toThrow(
			'Assistant SDK response has no body',
		);
	});

	// -----------------------------------------------------------------------
	// Test 14: AbortSignal cancellation
	// -----------------------------------------------------------------------
	it('should return immediately without error when signal is pre-aborted', async () => {
		const controller = new AbortController();
		controller.abort(); // Pre-abort before calling execute

		const response = createMockSdkResponse([
			{
				sessionId: 'sess-1',
				messages: [{ role: 'assistant', type: 'message', text: 'Should not process' }],
			},
		]);
		const client = createMockClient(response);
		const handler = new AssistantHandler(client);

		const result = await handler.execute({ query: 'test' }, 'user-1', writer, controller.signal);

		// Pre-aborted signal causes immediate return with empty results — no throw
		expect(result.responseText).toBe('');
		expect(result.hasCodeDiff).toBe(false);
		expect(writtenChunks).toHaveLength(0);
	});

	// -----------------------------------------------------------------------
	// Test 15: Incomplete JSON buffering
	// -----------------------------------------------------------------------
	it('should correctly parse JSON split across two reader.read() calls', async () => {
		const chunks: SdkStreamChunk[] = [
			{
				sessionId: 'sess-1',
				messages: [{ role: 'assistant', type: 'message', text: 'Buffered correctly' }],
			},
		];

		// Split the encoded data at byte 10 to simulate partial reads
		const response = createSplitResponse(chunks, 10);
		const client = createMockClient(response);
		const handler = new AssistantHandler(client);

		const result = await handler.execute({ query: 'test' }, 'user-1', writer);

		expect(writtenChunks).toHaveLength(1);
		expect((writtenChunks[0] as AgentMessageChunk).text).toBe('Buffered correctly');
		expect(result.responseText).toBe('Buffered correctly');
	});

	// -----------------------------------------------------------------------
	// Test 16: Payload — initial message (no sessionId)
	// -----------------------------------------------------------------------
	it('should build init-support-chat payload for initial message', () => {
		const client = createMockClient(createMockSdkResponse([]));
		const handler = new AssistantHandler(client);

		const context: AssistantContext = {
			query: 'How do I use the HTTP node?',
			userName: 'Alice',
		};

		const payload = handler.buildSdkPayload(context);

		expect(payload.sessionId).toBeUndefined();
		expect(payload.payload).toEqual(
			expect.objectContaining({
				role: 'user',
				type: 'init-support-chat',
				user: { firstName: 'Alice' },
				question: 'How do I use the HTTP node?',
			}),
		);
	});

	// -----------------------------------------------------------------------
	// Test 17: Payload — continuation (has sessionId)
	// -----------------------------------------------------------------------
	it('should build UserChatMessage payload for continuation', () => {
		const client = createMockClient(createMockSdkResponse([]));
		const handler = new AssistantHandler(client);

		const context: AssistantContext = {
			query: 'Can you explain more?',
			sdkSessionId: 'sess-existing',
		};

		const payload = handler.buildSdkPayload(context);

		expect(payload.sessionId).toBe('sess-existing');
		expect(payload.payload).toEqual({
			role: 'user',
			type: 'message',
			text: 'Can you explain more?',
		});
	});

	// -----------------------------------------------------------------------
	// Test 18: Payload — with workflow context
	// -----------------------------------------------------------------------
	it('should include workflowContext in init payload when workflowJSON is provided', () => {
		const client = createMockClient(createMockSdkResponse([]));
		const handler = new AssistantHandler(client);

		const context: AssistantContext = {
			query: 'Why is my workflow failing?',
			workflowJSON: {
				name: 'Test Workflow',
				nodes: [],
				connections: {},
			},
		};

		const payload = handler.buildSdkPayload(context);

		expect(payload.payload).toEqual(
			expect.objectContaining({
				type: 'init-support-chat',
				workflowContext: {
					currentWorkflow: {
						name: 'Test Workflow',
						nodes: [],
						connections: {},
					},
				},
			}),
		);
	});

	// -----------------------------------------------------------------------
	// Additional edge case: hasCodeDiff is false when no code-diff messages
	// -----------------------------------------------------------------------
	it('should set hasCodeDiff to false when no code-diff messages present', async () => {
		const response = createMockSdkResponse([
			{
				sessionId: 'sess-1',
				messages: [{ role: 'assistant', type: 'message', text: 'Just text' }],
			},
		]);
		const client = createMockClient(response);
		const handler = new AssistantHandler(client);

		const result = await handler.execute({ query: 'test' }, 'user-1', writer);

		expect(result.hasCodeDiff).toBe(false);
		expect(result.suggestionIds).toEqual([]);
	});

	// -----------------------------------------------------------------------
	// Additional edge case: default userName when not provided
	// -----------------------------------------------------------------------
	it('should default userName to "User" when not provided', () => {
		const client = createMockClient(createMockSdkResponse([]));
		const handler = new AssistantHandler(client);

		const payload = handler.buildSdkPayload({ query: 'Hello' });

		expect(payload.payload).toEqual(
			expect.objectContaining({
				user: { firstName: 'User' },
			}),
		);
	});
});
