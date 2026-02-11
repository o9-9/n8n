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
import { type ConversationEntry, entryToString } from '../code-builder/utils/code-builder-session';
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
	conversationHistory?: ConversationEntry[];
}

export interface PlanningAgentResult {
	route: 'ask_assistant' | 'build_workflow' | 'direct_reply';
	sdkSessionId?: string;
	assistantSummary?: string;
}

// ---------------------------------------------------------------------------
// Prompt
// ---------------------------------------------------------------------------

const PLANNING_PROMPT = prompt()
	.section(
		'role',
		`You are a routing agent for the n8n workflow builder.
You have tools available. Use them when appropriate, or respond directly if neither tool fits.

Rules:
- Make exactly zero or one tool call
- Pass the user's query faithfully to ask_assistant
- Include the full user request as instructions for build_workflow
- If neither tool is appropriate, respond directly with helpful text
- Use conversation history (if provided) to understand context from previous turns`,
	)
	.build();

// ---------------------------------------------------------------------------
// ask_assistant tool (placeholder — execution handled by handleAskAssistant)
// ---------------------------------------------------------------------------

const ASK_ASSISTANT_TOOL = tool(async () => '', {
	name: 'ask_assistant',
	description:
		'Ask the n8n assistant a question. Use this when the user needs help understanding n8n concepts, troubleshooting node errors, debugging workflow executions, setting up credentials, or asking "how does X work?" questions. Do NOT use this for requests to create, modify, or build workflows.',
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
 * - `direct_reply` — direct text reply for plan discussions (no credits)
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
		const { payload, userId, abortSignal, sdkSessionId, conversationHistory } = params;

		if (!this.llm.bindTools) {
			throw new Error('LLM does not support bindTools');
		}
		const llmWithTools = this.llm.bindTools([ASK_ASSISTANT_TOOL, BUILD_WORKFLOW_PLACEHOLDER]);

		let systemContent = PLANNING_PROMPT;
		if (conversationHistory && conversationHistory.length > 0) {
			const lines = conversationHistory.map((e, i) => `${i + 1}. ${entryToString(e)}`);
			systemContent += `\n\nConversation history:\n${lines.join('\n')}`;
		}

		const messages = [new SystemMessage(systemContent), new HumanMessage(payload.message)];

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
			return { route: 'direct_reply' };
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
