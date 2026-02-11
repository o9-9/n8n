import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { AIMessageChunk, BaseMessage } from '@langchain/core/messages';
import { HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages';
import type { Logger } from '@n8n/backend-common';

import { prompt } from '@/prompts/builder';
import type { StreamChunk, StreamOutput } from '@/types/streaming';

import { ASK_ASSISTANT_TOOL } from './ask-assistant.tool';
import type { AssistantHandler } from './assistant-handler';
import { BUILD_WORKFLOW_TOOL } from './build-workflow.tool';
import { type ConversationEntry, entryToString } from '../code-builder/utils/code-builder-session';
import type { ChatPayload } from '../workflow-builder-agent';
import type { StreamWriter } from './types';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PlanningAgentConfig {
	llm: BaseChatModel;
	assistantHandler: AssistantHandler;
	buildWorkflow: (
		payload: ChatPayload,
		userId: string,
		abortSignal?: AbortSignal,
	) => AsyncIterable<StreamOutput>;
	logger?: Logger;
}

export interface PlanningAgentParams {
	payload: ChatPayload;
	userId: string;
	abortSignal?: AbortSignal;
	sdkSessionId?: string;
	conversationHistory?: ConversationEntry[];
}

export interface PlanningAgentOutcome {
	sdkSessionId?: string;
	assistantSummary?: string;
	buildExecuted?: boolean;
}

/** Result of dispatching a single tool call */
interface ToolResult {
	content: string;
	terminal?: boolean;
}

/** Mutable state tracked across agent loop iterations */
interface PlanningAgentState {
	sdkSessionId?: string;
	assistantSummary?: string;
	buildExecuted?: boolean;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Maximum agent loop iterations to prevent runaway loops */
const MAX_ITERATIONS = 10;

// ---------------------------------------------------------------------------
// Prompt
// ---------------------------------------------------------------------------

const PLANNING_PROMPT = prompt()
	.section(
		'role',
		`You are a routing agent for the n8n workflow builder.
You have tools available. Use them when appropriate, or respond directly if neither tool fits.

Rules:
- Make exactly zero or one tool call per turn
- Pass the user's query faithfully to ask_assistant
- Include the full user request as instructions for build_workflow
- If neither tool is appropriate, respond directly with helpful text
- Use conversation history (if provided) to understand context from previous turns`,
	)
	.build();

// ---------------------------------------------------------------------------
// PlanningAgent
// ---------------------------------------------------------------------------

/**
 * Planning agent that classifies user messages and executes tools directly:
 * - `ask_assistant` — help/debug queries via AssistantHandler (no credits)
 * - `build_workflow` — workflow generation via CodeWorkflowBuilder (credits consumed)
 * - direct text reply — plan discussions (no credits)
 *
 * Unlike a router, this agent executes tools in-place, streaming all chunks
 * (assistant + builder) through a single generator. The consumer sees the
 * final outcome (facts, not routing decisions) after the generator completes.
 */
export class PlanningAgent {
	private readonly llm: BaseChatModel;

	private readonly assistantHandler: AssistantHandler;

	private readonly buildWorkflow: (
		payload: ChatPayload,
		userId: string,
		abortSignal?: AbortSignal,
	) => AsyncIterable<StreamOutput>;

	private readonly logger?: Logger;

	constructor(config: PlanningAgentConfig) {
		this.llm = config.llm;
		this.assistantHandler = config.assistantHandler;
		this.buildWorkflow = config.buildWorkflow;
		this.logger = config.logger;
	}

	/**
	 * Run the planning agent loop: LLM -> tool call -> execute -> ToolMessage -> loop.
	 * The loop is tool-agnostic — all tool knowledge lives in executeTool().
	 * Yields `StreamOutput` chunks and returns the outcome.
	 */
	async *run(params: PlanningAgentParams): AsyncGenerator<StreamOutput, PlanningAgentOutcome> {
		const { payload, userId, abortSignal, sdkSessionId, conversationHistory } = params;

		if (!this.llm.bindTools) {
			throw new Error('LLM does not support bindTools');
		}
		const llmWithTools = this.llm.bindTools([ASK_ASSISTANT_TOOL, BUILD_WORKFLOW_TOOL]);

		let systemContent = PLANNING_PROMPT;
		if (conversationHistory && conversationHistory.length > 0) {
			const lines = conversationHistory.map((e, i) => `${i + 1}. ${entryToString(e)}`);
			systemContent += `\n\nConversation history:\n${lines.join('\n')}`;
		}

		const messages: BaseMessage[] = [
			new SystemMessage(systemContent),
			new HumanMessage(payload.message),
		];

		const ctx = { userId, payload, abortSignal };
		const state: PlanningAgentState = { sdkSessionId };

		let reachedMaxIterations = true;

		for (let iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
			const response: AIMessageChunk = await llmWithTools.invoke(messages, {
				signal: abortSignal,
			});
			messages.push(response);

			const toolCalls = response.tool_calls ?? [];

			// No tool call -> natural termination (text response)
			if (toolCalls.length === 0) {
				const text = typeof response.content === 'string' ? response.content : '';
				if (text) {
					yield this.wrapChunk({
						role: 'assistant',
						type: 'message',
						text,
					});
				}
				reachedMaxIterations = false;
				break;
			}

			// Process tool calls — loop is tool-agnostic
			for (const toolCall of toolCalls) {
				const toolCallId = toolCall.id ?? `tc-${iteration}`;
				const result = yield* this.executeToolWithStreaming(toolCall, ctx, state);

				messages.push(
					new ToolMessage({
						tool_call_id: toolCallId,
						content: result.content,
					}),
				);

				if (result.terminal) {
					return this.getOutcome(state);
				}
			}

			// Loop continues — LLM sees tool results on next iteration
		}

		if (reachedMaxIterations) {
			this.logger?.warn('[PlanningAgent] Max iterations reached');
		}

		return this.getOutcome(state);
	}

	// -----------------------------------------------------------------------
	// Tool execution bridge
	// -----------------------------------------------------------------------

	/**
	 * Generic bridge: starts a tool, drains its streaming queue concurrently,
	 * and yields chunks as they arrive. Tool-agnostic — all tool knowledge
	 * lives in executeTool().
	 */
	private async *executeToolWithStreaming(
		toolCall: { name: string; args: Record<string, unknown> },
		ctx: { userId: string; payload: ChatPayload; abortSignal?: AbortSignal },
		state: PlanningAgentState,
	): AsyncGenerator<StreamOutput, ToolResult> {
		const queue: StreamOutput[] = [];
		let resolveNext: (() => void) | undefined;
		let done = false;

		const enqueue = (output: StreamOutput) => {
			queue.push(output);
			resolveNext?.();
		};

		const toolPromise = this.executeTool(toolCall, ctx, state, enqueue).finally(() => {
			done = true;
			resolveNext?.();
		});

		// Drain the queue while the tool runs
		while (!done || queue.length > 0) {
			if (queue.length > 0) {
				yield queue.shift()!;
			} else if (!done) {
				await new Promise<void>((resolve) => {
					resolveNext = resolve;
					// Check again in case chunk arrived between the check and await
					if (queue.length > 0 || done) {
						resolve();
					}
				});
			}
		}

		return await toolPromise;
	}

	// -----------------------------------------------------------------------
	// Tool executors
	// -----------------------------------------------------------------------

	/**
	 * Plain async function containing all tool knowledge. Streams via
	 * the enqueue() side-effect callback.
	 */
	private async executeTool(
		toolCall: { name: string; args: Record<string, unknown> },
		ctx: { userId: string; payload: ChatPayload; abortSignal?: AbortSignal },
		state: PlanningAgentState,
		enqueue: (output: StreamOutput) => void,
	): Promise<ToolResult> {
		switch (toolCall.name) {
			case 'ask_assistant': {
				const writer: StreamWriter = (chunk: StreamChunk) => {
					enqueue({ messages: [chunk] });
				};

				// Yield tool progress: running
				enqueue(
					this.wrapChunk({
						type: 'tool',
						toolName: 'assistant',
						status: 'running',
					}),
				);

				const currentWorkflow = ctx.payload.workflowContext?.currentWorkflow;
				const workflowJSON = currentWorkflow
					? {
							name: currentWorkflow.name ?? '',
							nodes: currentWorkflow.nodes ?? [],
							connections: currentWorkflow.connections ?? {},
						}
					: undefined;

				const result = await this.assistantHandler.execute(
					{
						query: (toolCall.args as { query: string }).query,
						sdkSessionId: state.sdkSessionId,
						workflowJSON,
					},
					ctx.userId,
					writer,
					ctx.abortSignal,
				);

				// Yield tool progress: completed
				enqueue(
					this.wrapChunk({
						type: 'tool',
						toolName: 'assistant',
						status: 'completed',
					}),
				);

				state.sdkSessionId = result.sdkSessionId;
				state.assistantSummary = result.summary;
				return { content: result.summary };
			}

			case 'build_workflow': {
				for await (const chunk of this.buildWorkflow(ctx.payload, ctx.userId, ctx.abortSignal)) {
					enqueue(chunk);
				}
				state.buildExecuted = true;
				return { content: 'Workflow built.', terminal: true };
			}

			default:
				this.logger?.warn('[PlanningAgent] Unknown tool call', {
					toolName: toolCall.name,
				});
				return { content: `Unknown tool: ${toolCall.name}` };
		}
	}

	// -----------------------------------------------------------------------
	// Outcome
	// -----------------------------------------------------------------------

	/**
	 * Trivial state copy — returns facts about what happened, not routing decisions.
	 */
	private getOutcome(state: PlanningAgentState): PlanningAgentOutcome {
		return {
			sdkSessionId: state.sdkSessionId,
			assistantSummary: state.assistantSummary,
			buildExecuted: state.buildExecuted,
		};
	}

	// -----------------------------------------------------------------------
	// Helpers
	// -----------------------------------------------------------------------

	private wrapChunk(chunk: StreamChunk): StreamOutput {
		return { messages: [chunk] };
	}
}
