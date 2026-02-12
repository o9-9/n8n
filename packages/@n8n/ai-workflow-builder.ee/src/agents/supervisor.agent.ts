import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { BaseMessage } from '@langchain/core/messages';
import { HumanMessage } from '@langchain/core/messages';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import type { RunnableConfig } from '@langchain/core/runnables';
import { z } from 'zod';

import { buildSupervisorPrompt } from '@/prompts';

import type { CoordinationLogEntry } from '../types/coordination';
import type { SimpleWorkflow } from '../types/workflow';
import {
	buildSelectedNodesSummary,
	buildSimplifiedExecutionContext,
	buildWorkflowSummary,
} from '../utils/context-builders';
import { summarizeCoordinationLog } from '../utils/coordination-log';
import type { ChatPayload } from '../workflow-builder-agent';

function createSupervisorSystemPrompt(enableAssistant: boolean) {
	return ChatPromptTemplate.fromMessages([
		[
			'system',
			[
				{
					type: 'text',
					text: buildSupervisorPrompt({ enableAssistant }),
					cache_control: { type: 'ephemeral' },
				},
			],
		],
		['placeholder', '{messages}'],
	]);
}

/**
 * Schema for supervisor routing decision.
 * When `enableAssistant` is true, includes the `assistant` routing option.
 */
export function createSupervisorRoutingSchema(enableAssistant: boolean) {
	const targets = enableAssistant
		? (['responder', 'discovery', 'builder', 'assistant'] as const)
		: (['responder', 'discovery', 'builder'] as const);

	return z.object({
		reasoning: z.string().describe('One sentence explaining why this agent should act next'),
		next: z.enum(targets).describe('The next agent to call'),
	});
}

export const supervisorRoutingSchema = createSupervisorRoutingSchema(true);

export type SupervisorRouting = z.infer<typeof supervisorRoutingSchema>;

export interface SupervisorAgentConfig {
	llm: BaseChatModel;
	/** Whether the assistant subgraph is available for routing. */
	enableAssistant?: boolean;
}

/**
 * Context required for the supervisor to make routing decisions
 */
export interface SupervisorContext {
	/** Conversation messages */
	messages: BaseMessage[];
	/** Current workflow state */
	workflowJSON: SimpleWorkflow;
	/** Coordination log tracking subgraph completion */
	coordinationLog: CoordinationLogEntry[];
	/** Summary of previous conversation (from compaction) */
	previousSummary?: string;
	/** Workflow context with execution data */
	workflowContext?: ChatPayload['workflowContext'];
}

/**
 * Supervisor Agent
 *
 * Coordinates the multi-agent workflow building process.
 * Routes to Discovery or Builder agents based on current state.
 */
export class SupervisorAgent {
	private llm: BaseChatModel;

	private enableAssistant: boolean;

	constructor(config: SupervisorAgentConfig) {
		this.llm = config.llm;
		this.enableAssistant = config.enableAssistant ?? false;
	}

	/**
	 * Build context message with workflow summary and completed phases
	 */
	private buildContextMessage(context: SupervisorContext): HumanMessage | null {
		const contextParts: string[] = [];

		// 1. Previous conversation summary (from compaction)
		if (context.previousSummary) {
			contextParts.push('<previous_conversation_summary>');
			contextParts.push(context.previousSummary);
			contextParts.push('</previous_conversation_summary>');
		}

		const selectedNodesSummary = buildSelectedNodesSummary(context.workflowContext);
		if (selectedNodesSummary) {
			contextParts.push('<selected_nodes>');
			contextParts.push(selectedNodesSummary);
			contextParts.push('</selected_nodes>');
		}

		if (context.workflowJSON.nodes.length > 0) {
			contextParts.push('<workflow_summary>');
			contextParts.push(buildWorkflowSummary(context.workflowJSON));
			contextParts.push('</workflow_summary>');
		}

		if (context.coordinationLog.length > 0) {
			contextParts.push('<completed_phases>');
			contextParts.push(summarizeCoordinationLog(context.coordinationLog));
			contextParts.push('</completed_phases>');
		}

		if (context.workflowContext) {
			contextParts.push(
				buildSimplifiedExecutionContext(context.workflowContext, context.workflowJSON.nodes),
			);
		}

		if (contextParts.length === 0) {
			return null;
		}

		return new HumanMessage({ content: contextParts.join('\n\n') });
	}

	/**
	 * Invoke the supervisor to get routing decision
	 * @param context - Supervisor context with messages and workflow state
	 * @param config - Optional RunnableConfig for tracing callbacks
	 */
	async invoke(context: SupervisorContext, config?: RunnableConfig): Promise<SupervisorRouting> {
		const routingSchema = createSupervisorRoutingSchema(this.enableAssistant);
		const agent = createSupervisorSystemPrompt(this.enableAssistant).pipe<SupervisorRouting>(
			this.llm.withStructuredOutput(routingSchema, {
				name: 'routing_decision',
			}),
		);

		const contextMessage = this.buildContextMessage(context);
		const messagesToSend = contextMessage
			? [...context.messages, contextMessage]
			: context.messages;

		return await agent.invoke({ messages: messagesToSend }, config);
	}
}
