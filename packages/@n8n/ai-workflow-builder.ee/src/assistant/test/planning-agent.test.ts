import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { AIMessage } from '@langchain/core/messages';

import type { StreamOutput } from '../../types/streaming';
import type { AssistantHandler } from '../assistant-handler';
import { PlanningAgent } from '../planning-agent';
import type { PlanningAgentResult } from '../planning-agent';
import type { ChatPayload } from '../../workflow-builder-agent';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function createMockPayload(message = 'test message'): ChatPayload {
	return {
		id: 'msg-1',
		message,
	};
}

function createMockLlm(response: AIMessage): BaseChatModel {
	const boundModel = {
		invoke: jest.fn().mockResolvedValue(response),
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
		execute: jest.fn().mockImplementation(async (_ctx, _userId, writer) => {
			writer({
				role: 'assistant' as const,
				type: 'message' as const,
				text: result.responseText,
			});
			return result;
		}),
	} as unknown as AssistantHandler;
}

/**
 * Collect all yielded StreamOutput chunks and the final return value from the generator.
 */
async function collectGenerator(
	gen: AsyncGenerator<StreamOutput, PlanningAgentResult>,
): Promise<{ chunks: StreamOutput[]; result: PlanningAgentResult }> {
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
	// Test 1: LLM returns ask_assistant tool call
	// -----------------------------------------------------------------------
	it('should route to ask_assistant and yield assistant chunks', async () => {
		const response = new AIMessage({
			content: '',
			tool_calls: [
				{
					id: 'tc-1',
					name: 'ask_assistant',
					args: { query: 'How do I use the HTTP node?' },
				},
			],
		});
		const llm = createMockLlm(response);
		const handler = createMockAssistantHandler();

		const agent = new PlanningAgent({ llm, assistantHandler: handler });
		const { chunks, result } = await collectGenerator(
			agent.run({ payload: createMockPayload(), userId: 'user-1' }),
		);

		expect(result.route).toBe('ask_assistant');
		expect(result.sdkSessionId).toBe('sdk-sess-1');
		expect(result.assistantSummary).toBe('Assistant says hi');
		expect(chunks.length).toBeGreaterThan(0);
		expect(chunks[0].messages[0]).toEqual(
			expect.objectContaining({ type: 'message', text: 'Assistant says hi' }),
		);
		expect(handler.execute).toHaveBeenCalledTimes(1);
	});

	// -----------------------------------------------------------------------
	// Test 2: LLM returns build_workflow tool call
	// -----------------------------------------------------------------------
	it('should route to build_workflow without invoking assistant handler', async () => {
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

		const agent = new PlanningAgent({ llm, assistantHandler: handler });
		const { chunks, result } = await collectGenerator(
			agent.run({ payload: createMockPayload(), userId: 'user-1' }),
		);

		expect(result.route).toBe('build_workflow');
		expect(chunks).toHaveLength(0);
		expect(handler.execute).not.toHaveBeenCalled();
	});

	// -----------------------------------------------------------------------
	// Test 3: LLM returns text-only response (no tool calls)
	// -----------------------------------------------------------------------
	it('should yield text as AgentMessageChunk and return text_response route', async () => {
		const response = new AIMessage({
			content: 'Here is a plan: first we add a trigger, then a Slack node.',
		});
		const llm = createMockLlm(response);
		const handler = createMockAssistantHandler();

		const agent = new PlanningAgent({ llm, assistantHandler: handler });
		const { chunks, result } = await collectGenerator(
			agent.run({ payload: createMockPayload(), userId: 'user-1' }),
		);

		expect(result.route).toBe('text_response');
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
	it('should handle empty response gracefully and return text_response', async () => {
		const response = new AIMessage({ content: '' });
		const llm = createMockLlm(response);
		const handler = createMockAssistantHandler();

		const agent = new PlanningAgent({ llm, assistantHandler: handler });
		const { chunks, result } = await collectGenerator(
			agent.run({ payload: createMockPayload(), userId: 'user-1' }),
		);

		expect(result.route).toBe('text_response');
		expect(chunks).toHaveLength(0); // no text to yield
		expect(handler.execute).not.toHaveBeenCalled();
	});

	// -----------------------------------------------------------------------
	// Test 5: sdkSessionId passed through and returned
	// -----------------------------------------------------------------------
	it('should pass sdkSessionId to assistant handler and return it', async () => {
		const response = new AIMessage({
			content: '',
			tool_calls: [
				{
					id: 'tc-3',
					name: 'ask_assistant',
					args: { query: 'Follow up question' },
				},
			],
		});
		const llm = createMockLlm(response);
		const handler = createMockAssistantHandler({
			responseText: 'Follow up answer',
			summary: 'Follow up answer',
			sdkSessionId: 'sdk-sess-existing',
			hasCodeDiff: false,
			suggestionIds: [],
		});

		const agent = new PlanningAgent({ llm, assistantHandler: handler });
		const { result } = await collectGenerator(
			agent.run({
				payload: createMockPayload('Follow up question'),
				userId: 'user-1',
				sdkSessionId: 'sdk-sess-prev',
			}),
		);

		expect(result.route).toBe('ask_assistant');
		expect(result.sdkSessionId).toBe('sdk-sess-existing');

		// Verify the sdkSessionId was passed to the handler
		const executeCall = (handler.execute as jest.Mock).mock.calls[0];
		expect(executeCall[0].sdkSessionId).toBe('sdk-sess-prev');
	});

	// -----------------------------------------------------------------------
	// Test 6: Unknown tool name falls back to build_workflow
	// -----------------------------------------------------------------------
	it('should fall back to build_workflow for unknown tool names', async () => {
		const response = new AIMessage({
			content: '',
			tool_calls: [
				{
					id: 'tc-4',
					name: 'unknown_tool',
					args: { foo: 'bar' },
				},
			],
		});
		const llm = createMockLlm(response);
		const handler = createMockAssistantHandler();
		const mockLogger = { warn: jest.fn(), debug: jest.fn() };

		const agent = new PlanningAgent({
			llm,
			assistantHandler: handler,
			logger: mockLogger as unknown as PlanningAgent extends { logger?: infer L } ? L : never,
		});
		const { result } = await collectGenerator(
			agent.run({ payload: createMockPayload(), userId: 'user-1' }),
		);

		expect(result.route).toBe('build_workflow');
		expect(handler.execute).not.toHaveBeenCalled();
		expect(mockLogger.warn).toHaveBeenCalledWith(
			expect.stringContaining('Unknown tool call'),
			expect.objectContaining({ toolName: 'unknown_tool' }),
		);
	});
});
