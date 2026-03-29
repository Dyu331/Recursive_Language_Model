// Types matching the RLM log format

export interface RLMChatCompletion {
  root_model?: string;
  prompt: string | Record<string, unknown> | Array<Record<string, unknown>>;
  response: string;
  prompt_tokens?: number;
  completion_tokens?: number;
  execution_time: number;
  usage_summary?: UsageSummary;
  metadata?: RLMTrajectoryMetadata | null;
}

export interface REPLResult {
  stdout: string;
  stderr: string;
  locals: Record<string, unknown>;
  execution_time: number;
  rlm_calls: RLMChatCompletion[];
}

export interface CodeBlock {
  code: string;
  result: REPLResult;
}

export interface RLMIteration {
  type?: string;
  iteration: number;
  timestamp: string;
  prompt: Array<{ role: string; content: string }>;
  response: string;
  code_blocks: CodeBlock[];
  final_answer: string | [string, string] | null;
  iteration_time: number | null;
}

// Metadata saved at the start of a log file about RLM configuration
export interface RLMConfigMetadata {
  root_model: string | null;
  max_depth: number | null;
  max_iterations: number | null;
  backend: string | null;
  backend_kwargs: Record<string, unknown> | null;
  environment_type: string | null;
  environment_kwargs: Record<string, unknown> | null;
  other_backends: string[] | null;
}

export interface RLMLogFile {
  fileName: string;
  filePath: string;
  iterations: RLMIteration[];
  metadata: LogMetadata;
  config: RLMConfigMetadata;
}

export interface LogMetadata {
  totalIterations: number;
  totalCodeBlocks: number;
  totalSubLMCalls: number;
  totalSubLLMCalls: number;
  totalSubRLMCalls: number;
  contextQuestion: string;
  finalAnswer: string | null;
  totalExecutionTime: number;
  hasErrors: boolean;
}

export interface ModelUsageSummary {
  total_calls: number;
  total_input_tokens: number;
  total_output_tokens: number;
  total_cost?: number;
}

export interface UsageSummary {
  model_usage_summaries: Record<string, ModelUsageSummary>;
  total_cost?: number;
}

export interface RLMTrajectoryMetadata {
  run_metadata: Record<string, unknown>;
  iterations: RLMIteration[];
}

export function extractFinalAnswer(answer: string | [string, string] | null): string | null {
  if (!answer) return null;
  if (Array.isArray(answer)) {
    return answer[1];
  }
  return answer;
}

export function extractUsageTotals(usageSummary?: UsageSummary | null) {
  let inputTokens = 0;
  let outputTokens = 0;
  let totalCalls = 0;

  if (!usageSummary) {
    return { inputTokens, outputTokens, totalCalls };
  }

  for (const usage of Object.values(usageSummary.model_usage_summaries ?? {})) {
    inputTokens += usage.total_input_tokens ?? 0;
    outputTokens += usage.total_output_tokens ?? 0;
    totalCalls += usage.total_calls ?? 0;
  }

  return { inputTokens, outputTokens, totalCalls };
}

export function isRecursiveSubcall(call: RLMChatCompletion): boolean {
  return call.metadata != null;
}

export function countSubLLMCalls(calls: RLMChatCompletion[]): number {
  return calls.filter((call) => !isRecursiveSubcall(call)).length;
}

export function countSubRLMCalls(calls: RLMChatCompletion[]): number {
  return calls.filter((call) => isRecursiveSubcall(call)).length;
}

export function countIterationSubcalls(iteration: RLMIteration): number {
  return iteration.code_blocks.reduce(
    (total, block) => total + (block.result?.rlm_calls?.length || 0),
    0,
  );
}

export function countIterationSubLLMCalls(iteration: RLMIteration): number {
  return iteration.code_blocks.reduce(
    (total, block) => total + countSubLLMCalls(block.result?.rlm_calls ?? []),
    0,
  );
}

export function countIterationSubRLMCalls(iteration: RLMIteration): number {
  return iteration.code_blocks.reduce(
    (total, block) => total + countSubRLMCalls(block.result?.rlm_calls ?? []),
    0,
  );
}

export function countTrajectorySubcalls(iterations: RLMIteration[]): number {
  return iterations.reduce((total, iteration) => total + countIterationSubcalls(iteration), 0);
}

export function countTrajectorySubLLMCalls(iterations: RLMIteration[]): number {
  return iterations.reduce((total, iteration) => total + countIterationSubLLMCalls(iteration), 0);
}

export function countTrajectorySubRLMCalls(iterations: RLMIteration[]): number {
  return iterations.reduce((total, iteration) => total + countIterationSubRLMCalls(iteration), 0);
}

export function countTrajectoryCodeBlocks(iterations: RLMIteration[]): number {
  return iterations.reduce((total, iteration) => total + iteration.code_blocks.length, 0);
}

export function extractTrajectoryFinalAnswer(metadata?: RLMTrajectoryMetadata | null): string | null {
  if (!metadata) return null;

  for (let i = metadata.iterations.length - 1; i >= 0; i -= 1) {
    const finalAnswer = extractFinalAnswer(metadata.iterations[i].final_answer);
    if (finalAnswer) {
      return finalAnswer;
    }
  }

  return null;
}
