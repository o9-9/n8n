import { z } from 'zod';

/**
 * Schema-only tool definition for the `build_workflow` tool.
 * Used by `PlanningAgent` with `llm.bindTools()` — not a full LangChain tool.
 * The planning agent returns a routing decision; the caller handles execution.
 */
export const BUILD_WORKFLOW_TOOL = {
	name: 'build_workflow',
	description: 'Build or modify an n8n workflow based on the user request.',
	schema: z.object({
		instructions: z.string().describe('Clear instructions for what to build or modify'),
	}),
};
