import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { AIMessageChunk } from '@langchain/core/messages';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { tool } from '@langchain/core/tools';
import type { Logger } from '@n8n/backend-common';
import { z } from 'zod';

import { prompt } from '@/prompts/builder';
import type { StreamChunk, StreamOutput } from '@/types/streaming';

import type { AssistantHandler } from './assistant-handler';
import { BUILD_WORKFLOW_TOOL } from './build-workflow.tool';
import type { ChatPayload } from '../workflow-builder-agent';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PlanningAgentConfig {
	llm: BaseChatModel;
	assistantHandler: AssistantHandler;
	logger?: Logger;
}

export interface PlanningAgentParams {
	payload: ChatPayload;
	userId: string;
	abortSignal?: AbortSignal;
	sdkSessionId?: string;
}

export interface PlanningAgentResult {
	route: 'ask_assistant' | 'build_workflow' | 'text_response';
	sdkSessionId?: string;
	assistantSummary?: string;
}

// ---------------------------------------------------------------------------
// Prompt
// ---------------------------------------------------------------------------

const PLANNING_PROMPT = prompt()
	.section(
		'role',
		`You are a planning agent for the n8n workflow builder.

1. CLASSIFY the user's message:
   - "question" → help, debugging, errors, credentials, "how does X work?"
   - "build" → create or modify a workflow, add/connect nodes
   - "plan" → discuss an approach before building

2. Route:
   - QUESTION → call ask_assistant with the user's query
   - BUILD → call build_workflow with instructions
   - PLAN → respond with a brief text plan. No tool call.

Rules:
- Make exactly zero or one tool call
- Pass the user's query faithfully to ask_assistant
- Include the full user request as instructions for build_workflow`,
	)
	.build();

// ---------------------------------------------------------------------------
// ask_assistant tool (placeholder — execution handled by handleAskAssistant)
// ---------------------------------------------------------------------------

const ASK_ASSISTANT_TOOL = tool(async () => '', {
	name: 'ask_assistant',
	description:
		'Route a help, debugging, or credential question to the n8n assistant. Use this for questions about how n8n works, troubleshooting errors, or credential setup.',
	schema: z.object({
		query: z.string().describe('The user question to send to the assistant'),
	}),
});

// ---------------------------------------------------------------------------
// build_workflow tool (placeholder — caller handles execution)
// ---------------------------------------------------------------------------

const BUILD_WORKFLOW_PLACEHOLDER = tool(async () => '', {
	name: BUILD_WORKFLOW_TOOL.name,
	description: BUILD_WORKFLOW_TOOL.description,
	schema: BUILD_WORKFLOW_TOOL.schema,
});

// ---------------------------------------------------------------------------
// PlanningAgent
// ---------------------------------------------------------------------------

/**
 * Single-step planning agent that classifies user messages and routes to either:
 * - `ask_assistant` — help/debug queries via AssistantHandler (no credits)
 * - `build_workflow` — workflow generation via CodeWorkflowBuilder (credits consumed)
 * - `text_response` — direct text reply for plan discussions (no credits)
 */
export class PlanningAgent {
	private readonly llm: BaseChatModel;

	private readonly assistantHandler: AssistantHandler;

	private readonly logger?: Logger;

	constructor(config: PlanningAgentConfig) {
		this.llm = config.llm;
		this.assistantHandler = config.assistantHandler;
		this.logger = config.logger;
	}

	/**
	 * Run the planning agent: single LLM call → classify → route.
	 * Yields `StreamOutput` chunks and returns the routing result.
	 */
	async *run(params: PlanningAgentParams): AsyncGenerator<StreamOutput, PlanningAgentResult> {
		const { payload, userId, abortSignal, sdkSessionId } = params;

		if (!this.llm.bindTools) {
			throw new Error('LLM does not support bindTools');
		}
		const llmWithTools = this.llm.bindTools([ASK_ASSISTANT_TOOL, BUILD_WORKFLOW_PLACEHOLDER]);

		const messages = [new SystemMessage(PLANNING_PROMPT), new HumanMessage(payload.message)];

		const response: AIMessageChunk = await llmWithTools.invoke(messages, {
			signal: abortSignal,
		});

		const toolCalls = response.tool_calls ?? [];

		// No tool call → text response
		if (toolCalls.length === 0) {
			const text = typeof response.content === 'string' ? response.content : '';
			if (text) {
				yield this.wrapChunk({
					role: 'assistant',
					type: 'message',
					text,
				});
			}
			return { route: 'text_response' };
		}

		const toolCall = toolCalls[0];

		if (toolCall.name === 'ask_assistant') {
			return yield* this.handleAskAssistant(toolCall.args as { query: string }, {
				userId,
				payload,
				abortSignal,
				sdkSessionId,
			});
		}

		if (toolCall.name === 'build_workflow') {
			return { route: 'build_workflow' };
		}

		// Unknown tool name — defensive fallback to build_workflow
		this.logger?.warn('[PlanningAgent] Unknown tool call, falling back to build_workflow', {
			toolName: toolCall.name,
		});
		return { route: 'build_workflow' };
	}

	// -----------------------------------------------------------------------
	// Private helpers
	// -----------------------------------------------------------------------

	/**
	 * Bridge AssistantHandler's callback-based streaming to an async generator.
	 * Starts handler.execute() in background, yields chunks as they arrive.
	 */
	private async *handleAskAssistant(
		args: { query: string },
		ctx: {
			userId: string;
			payload: ChatPayload;
			abortSignal?: AbortSignal;
			sdkSessionId?: string;
		},
	): AsyncGenerator<StreamOutput, PlanningAgentResult> {
		const chunks: StreamChunk[] = [];
		let resolveNext: (() => void) | undefined;
		let done = false;

		const writer = (chunk: StreamChunk) => {
			chunks.push(chunk);
			resolveNext?.();
		};

		const currentWorkflow = ctx.payload.workflowContext?.currentWorkflow;
		const workflowJSON = currentWorkflow
			? {
					name: currentWorkflow.name ?? '',
					nodes: currentWorkflow.nodes ?? [],
					connections: currentWorkflow.connections ?? {},
				}
			: undefined;

		// Start handler in background
		const executePromise = this.assistantHandler
			.execute(
				{
					query: args.query,
					sdkSessionId: ctx.sdkSessionId,
					workflowJSON,
				},
				ctx.userId,
				writer,
				ctx.abortSignal,
			)
			.then((result) => {
				done = true;
				resolveNext?.();
				return result;
			});

		// Yield chunks as they arrive
		let cursor = 0;
		while (!done) {
			if (cursor < chunks.length) {
				yield this.wrapChunk(chunks[cursor]);
				cursor++;
			} else {
				await new Promise<void>((resolve) => {
					resolveNext = resolve;
					// Check again in case chunk arrived between the check and await
					if (cursor < chunks.length || done) {
						resolve();
					}
				});
			}
		}

		// Yield any remaining chunks
		while (cursor < chunks.length) {
			yield this.wrapChunk(chunks[cursor]);
			cursor++;
		}

		const result = await executePromise;

		return {
			route: 'ask_assistant',
			sdkSessionId: result.sdkSessionId,
			assistantSummary: result.summary,
		};
	}

	private wrapChunk(chunk: StreamChunk): StreamOutput {
		return { messages: [chunk] };
	}
}
