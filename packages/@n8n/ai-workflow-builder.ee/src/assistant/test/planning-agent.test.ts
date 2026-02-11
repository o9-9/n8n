import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { AIMessage } from '@langchain/core/messages';

import type { ConversationEntry } from '../../code-builder/utils/code-builder-session';
import type { StreamOutput } from '../../types/streaming';
import type { ChatPayload } from '../../workflow-builder-agent';
import type { AssistantHandler } from '../assistant-handler';
import { PlanningAgent } from '../planning-agent';
import type { PlanningAgentOutcome } from '../planning-agent';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function createMockPayload(message = 'test message'): ChatPayload {
	return {
		id: 'msg-1',
		message,
	};
}

/**
 * Create a mock LLM that returns a sequence of responses (one per invoke call).
 * Supports the agent loop pattern where the LLM may be called multiple times.
 */
function createMockLlm(responses: AIMessage | AIMessage[]): BaseChatModel {
	const responseList = Array.isArray(responses) ? responses : [responses];
	let callIndex = 0;
	const boundModel = {
		invoke: jest.fn().mockImplementation(async () => {
			const response = responseList[callIndex] ?? responseList[responseList.length - 1];
			callIndex++;
			return response;
		}),
	};
	return {
		bindTools: jest.fn().mockReturnValue(boundModel),
	} as unknown as BaseChatModel;
}

function createMockAssistantHandler(
	result = {
		responseText: 'Assistant says hi',
		summary: 'Assistant says hi',
		sdkSessionId: 'sdk-sess-1',
		hasCodeDiff: false,
		suggestionIds: [],
	},
): AssistantHandler {
	return {
		// eslint-disable-next-line @typescript-eslint/no-unsafe-call
		execute: jest
			.fn()
			.mockImplementation(
				async (_ctx: unknown, _userId: unknown, writer: (chunk: unknown) => void) => {
					writer({
						role: 'assistant' as const,
						type: 'message' as const,
						text: result.responseText,
					});
					return result;
				},
			),
	} as unknown as AssistantHandler;
}

/**
 * Create a mock buildWorkflow function that yields a sequence of chunks.
 */
function createMockBuildWorkflow(
	chunks: StreamOutput[] = [],
): (
	payload: ChatPayload,
	userId: string,
	abortSignal?: AbortSignal,
) => AsyncIterable<StreamOutput> {
	return async function* () {
		for (const chunk of chunks) {
			yield chunk;
		}
	};
}

/**
 * Collect all yielded StreamOutput chunks and the final return value from the generator.
 */
async function collectGenerator(
	gen: AsyncGenerator<StreamOutput, PlanningAgentOutcome>,
): Promise<{ chunks: StreamOutput[]; result: PlanningAgentOutcome }> {
	const chunks: StreamOutput[] = [];
	let iterResult = await gen.next();
	while (!iterResult.done) {
		chunks.push(iterResult.value);
		iterResult = await gen.next();
	}
	return { chunks, result: iterResult.value };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('PlanningAgent', () => {
	// -----------------------------------------------------------------------
	// Test 1: LLM returns ask_assistant -> tool executes -> LLM sees result -> outcome from state
	// -----------------------------------------------------------------------
	it('should execute ask_assistant, push ToolMessage, then derive outcome from state', async () => {
		// First call: LLM picks ask_assistant
		const firstResponse = new AIMessage({
			content: '',
			tool_calls: [
				{
					id: 'tc-1',
					name: 'ask_assistant',
					args: { query: 'How do I use the HTTP node?' },
				},
			],
		});
		// Second call: LLM sees the ToolMessage summary and responds with text
		const secondResponse = new AIMessage({
			content: 'Based on the assistant response, here is more info.',
		});

		const llm = createMockLlm([firstResponse, secondResponse]);
		const handler = createMockAssistantHandler();

		const agent = new PlanningAgent({
			llm,
			assistantHandler: handler,
			buildWorkflow: createMockBuildWorkflow(),
		});
		const { chunks, result } = await collectGenerator(
			agent.run({ payload: createMockPayload(), userId: 'user-1' }),
		);

		// Outcome has assistantSummary because ask_assistant was called
		expect(result.assistantSummary).toBe('Assistant says hi');
		expect(result.sdkSessionId).toBe('sdk-sess-1');
		expect(result.buildExecuted).toBeFalsy();
		expect(handler.execute).toHaveBeenCalledTimes(1);

		// Should have tool progress chunks (running, assistant chunk, completed) + final text
		const toolRunning = chunks.find(
			(c) =>
				c.messages[0].type === 'tool' &&
				'status' in c.messages[0] &&
				c.messages[0].status === 'running',
		);
		const toolCompleted = chunks.find(
			(c) =>
				c.messages[0].type === 'tool' &&
				'status' in c.messages[0] &&
				c.messages[0].status === 'completed',
		);
		const textChunk = chunks.find(
			(c) =>
				c.messages[0].type === 'message' &&
				'text' in c.messages[0] &&
				c.messages[0].text === 'Based on the assistant response, here is more info.',
		);

		expect(toolRunning).toBeDefined();
		expect(toolCompleted).toBeDefined();
		expect(textChunk).toBeDefined();

		// Verify LLM was called twice (agent loop)
		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-assignment
		const boundModel = (llm.bindTools as jest.Mock).mock.results[0].value;
		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
		expect(boundModel.invoke).toHaveBeenCalledTimes(2);
	});

	// -----------------------------------------------------------------------
	// Test 2: LLM returns build_workflow tool call (terminal) — now yields builder chunks
	// -----------------------------------------------------------------------
	it('should execute build_workflow, yield builder chunks, and set buildExecuted', async () => {
		const builderChunk: StreamOutput = {
			messages: [{ role: 'assistant', type: 'message', text: 'Built workflow' }],
		};

		const response = new AIMessage({
			content: '',
			tool_calls: [
				{
					id: 'tc-2',
					name: 'build_workflow',
					args: { instructions: 'Create a Slack notification workflow' },
				},
			],
		});
		const llm = createMockLlm(response);
		const handler = createMockAssistantHandler();

		const agent = new PlanningAgent({
			llm,
			assistantHandler: handler,
			buildWorkflow: createMockBuildWorkflow([builderChunk]),
		});
		const { chunks, result } = await collectGenerator(
			agent.run({ payload: createMockPayload(), userId: 'user-1' }),
		);

		expect(result.buildExecuted).toBe(true);
		expect(result.assistantSummary).toBeUndefined();
		// Builder chunks are now yielded
		expect(chunks).toHaveLength(1);
		expect(chunks[0]).toEqual(builderChunk);
		expect(handler.execute).not.toHaveBeenCalled();
	});

	// -----------------------------------------------------------------------
	// Test 3: LLM returns text-only response (no tool calls)
	// -----------------------------------------------------------------------
	it('should yield text as AgentMessageChunk and return outcome with no fields set', async () => {
		const response = new AIMessage({
			content: 'Here is a plan: first we add a trigger, then a Slack node.',
		});
		const llm = createMockLlm(response);
		const handler = createMockAssistantHandler();

		const agent = new PlanningAgent({
			llm,
			assistantHandler: handler,
			buildWorkflow: createMockBuildWorkflow(),
		});
		const { chunks, result } = await collectGenerator(
			agent.run({ payload: createMockPayload(), userId: 'user-1' }),
		);

		expect(result.buildExecuted).toBeFalsy();
		expect(result.assistantSummary).toBeUndefined();
		expect(chunks).toHaveLength(1);
		expect(chunks[0].messages[0]).toEqual(
			expect.objectContaining({
				role: 'assistant',
				type: 'message',
				text: 'Here is a plan: first we add a trigger, then a Slack node.',
			}),
		);
		expect(handler.execute).not.toHaveBeenCalled();
	});

	// -----------------------------------------------------------------------
	// Test 4: LLM returns no content and no tool calls
	// -----------------------------------------------------------------------
	it('should handle empty response gracefully and return empty outcome', async () => {
		const response = new AIMessage({ content: '' });
		const llm = createMockLlm(response);
		const handler = createMockAssistantHandler();

		const agent = new PlanningAgent({
			llm,
			assistantHandler: handler,
			buildWorkflow: createMockBuildWorkflow(),
		});
		const { chunks, result } = await collectGenerator(
			agent.run({ payload: createMockPayload(), userId: 'user-1' }),
		);

		expect(result.buildExecuted).toBeFalsy();
		expect(result.assistantSummary).toBeUndefined();
		expect(chunks).toHaveLength(0); // no text to yield
		expect(handler.execute).not.toHaveBeenCalled();
	});

	// -----------------------------------------------------------------------
	// Test 5: sdkSessionId passed through to assistant handler
	// -----------------------------------------------------------------------
	it('should pass sdkSessionId to assistant handler and return it', async () => {
		const firstResponse = new AIMessage({
			content: '',
			tool_calls: [
				{
					id: 'tc-3',
					name: 'ask_assistant',
					args: { query: 'Follow up question' },
				},
			],
		});
		// LLM sees assistant summary and responds with text
		const secondResponse = new AIMessage({
			content: 'Got it.',
		});

		const llm = createMockLlm([firstResponse, secondResponse]);
		const handler = createMockAssistantHandler({
			responseText: 'Follow up answer',
			summary: 'Follow up answer',
			sdkSessionId: 'sdk-sess-existing',
			hasCodeDiff: false,
			suggestionIds: [],
		});

		const agent = new PlanningAgent({
			llm,
			assistantHandler: handler,
			buildWorkflow: createMockBuildWorkflow(),
		});
		await collectGenerator(
			agent.run({
				payload: createMockPayload('Follow up question'),
				userId: 'user-1',
				sdkSessionId: 'sdk-sess-prev',
			}),
		);

		// Verify the sdkSessionId was passed to the handler
		const executeCall = (handler.execute as jest.Mock).mock.calls[0];
		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
		expect(executeCall[0].sdkSessionId).toBe('sdk-sess-prev');
	});

	// -----------------------------------------------------------------------
	// Test 6: Unknown tool name -> ToolMessage error pushed, LLM retried -> build_workflow
	// -----------------------------------------------------------------------
	it('should push error ToolMessage for unknown tools and let LLM retry', async () => {
		const firstResponse = new AIMessage({
			content: '',
			tool_calls: [
				{
					id: 'tc-4',
					name: 'unknown_tool',
					args: { foo: 'bar' },
				},
			],
		});
		// After seeing the error ToolMessage, LLM falls back to build_workflow
		const secondResponse = new AIMessage({
			content: '',
			tool_calls: [
				{
					id: 'tc-5',
					name: 'build_workflow',
					args: { instructions: 'Build it' },
				},
			],
		});

		const llm = createMockLlm([firstResponse, secondResponse]);
		const handler = createMockAssistantHandler();
		const mockLogger = { warn: jest.fn(), debug: jest.fn() };

		const agent = new PlanningAgent({
			llm,
			assistantHandler: handler,
			buildWorkflow: createMockBuildWorkflow(),
			logger: mockLogger as unknown as PlanningAgent extends { logger?: infer L } ? L : never,
		});
		const { result } = await collectGenerator(
			agent.run({ payload: createMockPayload(), userId: 'user-1' }),
		);

		expect(result.buildExecuted).toBe(true);
		expect(handler.execute).not.toHaveBeenCalled();
		expect(mockLogger.warn).toHaveBeenCalledWith(
			expect.stringContaining('Unknown tool call'),
			expect.objectContaining({ toolName: 'unknown_tool' }),
		);

		// LLM was called twice (retry after unknown tool)
		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-assignment
		const boundModel = (llm.bindTools as jest.Mock).mock.results[0].value;
		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
		expect(boundModel.invoke).toHaveBeenCalledTimes(2);
	});

	// -----------------------------------------------------------------------
	// Test 7: Conversation history included in system message
	// -----------------------------------------------------------------------
	it('should include conversation history in LLM system message when provided', async () => {
		const response = new AIMessage({ content: 'Noted context.' });
		const llm = createMockLlm(response);
		const handler = createMockAssistantHandler();

		const history: ConversationEntry[] = [
			{ type: 'build-request', message: 'Build a Slack workflow' },
			{ type: 'assistant-exchange', userQuery: 'How?', assistantSummary: 'Use Slack node' },
		];

		const agent = new PlanningAgent({
			llm,
			assistantHandler: handler,
			buildWorkflow: createMockBuildWorkflow(),
		});
		await collectGenerator(
			agent.run({
				payload: createMockPayload('Next step?'),
				userId: 'user-1',
				conversationHistory: history,
			}),
		);

		// Get the bound model's invoke call to inspect messages
		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-assignment
		const boundModel = (llm.bindTools as jest.Mock).mock.results[0].value;
		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-assignment
		const invokeArgs = (boundModel.invoke as jest.Mock).mock.calls[0][0];
		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
		const systemMessage = invokeArgs[0];

		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
		expect(systemMessage.content).toContain('Conversation history:');
		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
		expect(systemMessage.content).toContain('Build a Slack workflow');
		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
		expect(systemMessage.content).toContain('[Help] Q: How?');
	});

	// -----------------------------------------------------------------------
	// Test 8: No history section when conversationHistory is empty
	// -----------------------------------------------------------------------
	it('should not include conversation history section when empty', async () => {
		const response = new AIMessage({ content: 'Reply.' });
		const llm = createMockLlm(response);
		const handler = createMockAssistantHandler();

		const agent = new PlanningAgent({
			llm,
			assistantHandler: handler,
			buildWorkflow: createMockBuildWorkflow(),
		});
		await collectGenerator(
			agent.run({
				payload: createMockPayload(),
				userId: 'user-1',
				conversationHistory: [],
			}),
		);

		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-assignment
		const boundModel = (llm.bindTools as jest.Mock).mock.results[0].value;
		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-assignment
		const invokeArgs = (boundModel.invoke as jest.Mock).mock.calls[0][0];
		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
		const systemMessage = invokeArgs[0];

		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
		expect(systemMessage.content).not.toContain('Conversation history:');
	});

	// -----------------------------------------------------------------------
	// Test 9: Multi-step reasoning — ask_assistant then build_workflow
	// -----------------------------------------------------------------------
	it('should support multi-step: ask_assistant -> see result -> build_workflow', async () => {
		const builderChunk: StreamOutput = {
			messages: [{ role: 'assistant', type: 'message', text: 'Built the fix' }],
		};

		// First call: LLM picks ask_assistant
		const firstResponse = new AIMessage({
			content: '',
			tool_calls: [
				{
					id: 'tc-1',
					name: 'ask_assistant',
					args: { query: 'How do I fix the Google Sheets error?' },
				},
			],
		});
		// Second call: LLM sees assistant summary, decides to build
		const secondResponse = new AIMessage({
			content: '',
			tool_calls: [
				{
					id: 'tc-2',
					name: 'build_workflow',
					args: { instructions: 'Fix the Google Sheets node based on the error' },
				},
			],
		});

		const llm = createMockLlm([firstResponse, secondResponse]);
		const handler = createMockAssistantHandler({
			responseText: 'The error is caused by missing credentials',
			summary: 'Missing credentials error',
			sdkSessionId: 'sdk-sess-1',
			hasCodeDiff: false,
			suggestionIds: [],
		});

		const agent = new PlanningAgent({
			llm,
			assistantHandler: handler,
			buildWorkflow: createMockBuildWorkflow([builderChunk]),
		});
		const { chunks, result } = await collectGenerator(
			agent.run({ payload: createMockPayload(), userId: 'user-1' }),
		);

		// Final outcome: build was executed
		expect(result.buildExecuted).toBe(true);
		expect(handler.execute).toHaveBeenCalledTimes(1);

		// Builder chunks should be in the output
		expect(chunks.some((c) => c === builderChunk)).toBe(true);

		// LLM was called twice
		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-assignment
		const boundModel = (llm.bindTools as jest.Mock).mock.results[0].value;
		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
		expect(boundModel.invoke).toHaveBeenCalledTimes(2);

		// Verify ToolMessage was pushed with the summary (inspect the second invoke call's messages)
		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-assignment
		const secondCallMessages = (boundModel.invoke as jest.Mock).mock.calls[1][0];
		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-call
		const toolMessage = secondCallMessages.find(
			// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
			(m: { constructor: { name: string }; content: string }) =>
				m.constructor.name === 'ToolMessage',
		);
		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
		expect(toolMessage?.content).toBe('Missing credentials error');
	});

	// -----------------------------------------------------------------------
	// Test 10: Max iterations exhausted — fallback with assistant summary
	// -----------------------------------------------------------------------
	it('should return outcome with assistantSummary when max iterations reached', async () => {
		// All 10 iterations: LLM keeps calling ask_assistant (unusual but tests the limit)
		const askResponse = new AIMessage({
			content: '',
			tool_calls: [
				{
					id: 'tc-loop',
					name: 'ask_assistant',
					args: { query: 'More help' },
				},
			],
		});

		// createMockLlm repeats the last response when index exceeds array length
		const llm = createMockLlm([askResponse]);
		const handler = createMockAssistantHandler();

		const agent = new PlanningAgent({
			llm,
			assistantHandler: handler,
			buildWorkflow: createMockBuildWorkflow(),
		});
		const { result } = await collectGenerator(
			agent.run({ payload: createMockPayload(), userId: 'user-1' }),
		);

		// Should return outcome with assistantSummary since assistant was called
		expect(result.assistantSummary).toBe('Assistant says hi');
		expect(result.sdkSessionId).toBe('sdk-sess-1');
		expect(result.buildExecuted).toBeFalsy();
		expect(handler.execute).toHaveBeenCalledTimes(10);
	});

	// -----------------------------------------------------------------------
	// Test 11: Max iterations exhausted — fallback with no state accumulated
	// -----------------------------------------------------------------------
	it('should return empty outcome when max iterations reached without state', async () => {
		// All iterations: LLM keeps calling unknown tools
		const unknownResponse = new AIMessage({
			content: '',
			tool_calls: [
				{
					id: 'tc-unknown',
					name: 'unknown_tool',
					args: {},
				},
			],
		});

		// createMockLlm repeats the last response when index exceeds array length
		const llm = createMockLlm([unknownResponse]);
		const handler = createMockAssistantHandler();
		const mockLogger = { warn: jest.fn(), debug: jest.fn() };

		const agent = new PlanningAgent({
			llm,
			assistantHandler: handler,
			buildWorkflow: createMockBuildWorkflow(),
			logger: mockLogger as unknown as PlanningAgent extends { logger?: infer L } ? L : never,
		});
		const { result } = await collectGenerator(
			agent.run({ payload: createMockPayload(), userId: 'user-1' }),
		);

		// All outcome fields falsy
		expect(result.buildExecuted).toBeFalsy();
		expect(result.assistantSummary).toBeUndefined();
		expect(handler.execute).not.toHaveBeenCalled();
		expect(mockLogger.warn).toHaveBeenCalledWith(expect.stringContaining('Max iterations reached'));
	});

	// -----------------------------------------------------------------------
	// Test 12: Tool progress chunks yielded for ask_assistant
	// -----------------------------------------------------------------------
	it('should yield running and completed tool progress chunks for ask_assistant', async () => {
		const firstResponse = new AIMessage({
			content: '',
			tool_calls: [
				{
					id: 'tc-1',
					name: 'ask_assistant',
					args: { query: 'Help me' },
				},
			],
		});
		const secondResponse = new AIMessage({ content: 'Done.' });

		const llm = createMockLlm([firstResponse, secondResponse]);
		const handler = createMockAssistantHandler();

		const agent = new PlanningAgent({
			llm,
			assistantHandler: handler,
			buildWorkflow: createMockBuildWorkflow(),
		});
		const { chunks } = await collectGenerator(
			agent.run({ payload: createMockPayload(), userId: 'user-1' }),
		);

		// Find tool progress chunks
		const toolChunks = chunks.filter((c) => c.messages[0].type === 'tool');
		expect(toolChunks.length).toBeGreaterThanOrEqual(2);

		const statuses = toolChunks.map((c) => {
			const msg = c.messages[0];
			return 'status' in msg ? msg.status : undefined;
		});
		expect(statuses).toContain('running');
		expect(statuses).toContain('completed');
	});

	// -----------------------------------------------------------------------
	// Test 13: Schema-only tools used (no placeholder tool functions)
	// -----------------------------------------------------------------------
	it('should bind schema-only tools to the LLM', async () => {
		const response = new AIMessage({ content: 'Hi' });
		const llm = createMockLlm(response);
		const handler = createMockAssistantHandler();

		const agent = new PlanningAgent({
			llm,
			assistantHandler: handler,
			buildWorkflow: createMockBuildWorkflow(),
		});
		await collectGenerator(agent.run({ payload: createMockPayload(), userId: 'user-1' }));

		// Verify bindTools was called with schema-only tool definitions (not LangChain tool instances)
		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-assignment
		const bindToolsCall = (llm.bindTools as jest.Mock).mock.calls[0][0];
		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
		expect(bindToolsCall[0].name).toBe('ask_assistant');
		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
		expect(bindToolsCall[1].name).toBe('build_workflow');

		// Schema-only tools should have a schema property and a name, but no invoke method
		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
		expect(bindToolsCall[0].schema).toBeDefined();
		// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
		expect(bindToolsCall[0].invoke).toBeUndefined();
	});

	// -----------------------------------------------------------------------
	// Test 14: Priority — buildExecuted beats assistantSummary
	// -----------------------------------------------------------------------
	it('should prioritize build_workflow over ask_assistant in outcome', async () => {
		const builderChunk: StreamOutput = {
			messages: [{ role: 'assistant', type: 'message', text: 'Built it' }],
		};

		// ask_assistant first (sets assistantSummary), then build_workflow (sets buildExecuted)
		const firstResponse = new AIMessage({
			content: '',
			tool_calls: [
				{
					id: 'tc-1',
					name: 'ask_assistant',
					args: { query: 'What is this workflow doing?' },
				},
			],
		});
		const secondResponse = new AIMessage({
			content: '',
			tool_calls: [
				{
					id: 'tc-2',
					name: 'build_workflow',
					args: { instructions: 'Now build it' },
				},
			],
		});

		const llm = createMockLlm([firstResponse, secondResponse]);
		const handler = createMockAssistantHandler();

		const agent = new PlanningAgent({
			llm,
			assistantHandler: handler,
			buildWorkflow: createMockBuildWorkflow([builderChunk]),
		});
		const { result } = await collectGenerator(
			agent.run({ payload: createMockPayload(), userId: 'user-1' }),
		);

		// buildExecuted is set because build_workflow ran
		expect(result.buildExecuted).toBe(true);
		expect(handler.execute).toHaveBeenCalledTimes(1);
	});
});
