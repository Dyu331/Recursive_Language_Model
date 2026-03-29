'use client';

import { useState } from 'react';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  RLMChatCompletion,
  RLMIteration,
  countIterationSubcalls,
  countTrajectoryCodeBlocks,
  countTrajectorySubcalls,
  extractFinalAnswer,
  extractTrajectoryFinalAnswer,
  extractUsageTotals,
} from '@/lib/types';
import { cn } from '@/lib/utils';

interface SubLMCallCardProps {
  call: RLMChatCompletion;
  index: number;
  originLabel?: string;
  depth?: number;
}

function formatValue(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }

  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function formatPrompt(prompt: RLMChatCompletion['prompt']): string {
  return formatValue(prompt);
}

function summarizePromptMessages(prompt: Array<{ role: string; content: string }>): string {
  const roleSummary = prompt.map((message) => message.role).join(' → ');
  const totalChars = prompt.reduce((sum, message) => sum + message.content.length, 0);
  return `${prompt.length} message${prompt.length === 1 ? '' : 's'} (${roleSummary}) • ${totalChars.toLocaleString()} chars`;
}

function summarizePromptValue(prompt: RLMChatCompletion['prompt']): string {
  if (typeof prompt === 'string') {
    return `Prompt hidden • ${prompt.length.toLocaleString()} chars`;
  }

  if (Array.isArray(prompt)) {
    const totalChars = prompt.reduce((sum, item) => sum + formatValue(item).length, 0);
    return `Prompt hidden • ${prompt.length} item${prompt.length === 1 ? '' : 's'} • ${totalChars.toLocaleString()} chars`;
  }

  const serialized = formatValue(prompt);
  return `Prompt hidden • object payload • ${serialized.length.toLocaleString()} chars`;
}

function PromptDisclosure({
  summary,
  content,
  defaultOpen = false,
}: {
  summary: string;
  content: string;
  defaultOpen?: boolean;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="rounded-md border border-border/60 bg-muted/20">
      <button
        type="button"
        onClick={() => setIsOpen((value) => !value)}
        className="flex w-full items-center justify-between gap-3 px-3 py-2 text-left text-xs text-muted-foreground transition-colors hover:bg-muted/30"
      >
        <span>{summary}</span>
        <span className="font-mono text-[10px]">{isOpen ? 'Hide' : 'Show'}</span>
      </button>
      {isOpen && (
        <div className="border-t border-border/60 px-3 py-3">
          <pre className="max-h-64 overflow-y-auto whitespace-pre-wrap text-xs font-mono leading-relaxed text-foreground/90">
            {content}
          </pre>
        </div>
      )}
    </div>
  );
}

function SubagentIterationView({ iteration, depth }: { iteration: RLMIteration; depth: number }) {
  const nestedSubcalls = countIterationSubcalls(iteration);
  const promptContent = iteration.prompt
    .map((message) => `[${message.role}]\n${message.content}`)
    .join('\n\n');

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        <Badge variant="outline" className="text-[10px] font-mono">
          {iteration.code_blocks.length} code
        </Badge>
        <Badge variant="outline" className="text-[10px] font-mono">
          {nestedSubcalls} nested sub-LM
        </Badge>
        {iteration.iteration_time != null && (
          <Badge variant="outline" className="text-[10px] font-mono">
            {iteration.iteration_time.toFixed(2)}s
          </Badge>
        )}
        {iteration.final_answer && (
          <Badge className="bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30 text-[10px]">
            Final Answer
          </Badge>
        )}
      </div>

      <PromptDisclosure
        summary={`Prompt hidden for subagent iteration • ${summarizePromptMessages(iteration.prompt)}`}
        content={promptContent}
      />

      <div className="rounded-md border border-sky-500/20 bg-sky-500/5 p-3">
        <div className="mb-1 text-[10px] uppercase tracking-wider text-sky-600 dark:text-sky-400">
          Response
        </div>
        <pre className="whitespace-pre-wrap text-xs font-mono leading-relaxed text-foreground/90">
          {iteration.response}
        </pre>
      </div>

      {iteration.final_answer && (
        <div className="rounded-md border border-emerald-500/20 bg-emerald-500/5 p-3 text-xs text-foreground/90">
          {extractFinalAnswer(iteration.final_answer)}
        </div>
      )}

      <div className="space-y-3">
        {iteration.code_blocks.map((block, blockIndex) => {
          const blockSubcalls = block.result?.rlm_calls?.length || 0;
          return (
            <div key={blockIndex} className="rounded-lg border border-border bg-background/80">
              <div className="flex flex-wrap items-center gap-2 border-b border-border/60 px-3 py-2">
                <span className="text-xs font-medium">Code Block #{blockIndex + 1}</span>
                <Badge variant="outline" className="text-[10px] font-mono">
                  {block.result.execution_time?.toFixed(2) ?? '0.00'}s
                </Badge>
                <Badge variant="outline" className="text-[10px] font-mono">
                  nested sub-LM: {blockSubcalls > 0 ? 'yes' : 'no'}
                </Badge>
                {blockSubcalls > 0 && (
                  <Badge className="bg-fuchsia-500/15 text-fuchsia-600 dark:text-fuchsia-400 border-fuchsia-500/30 text-[10px]">
                    {blockSubcalls} spawned
                  </Badge>
                )}
                {block.result.stderr && (
                  <Badge variant="destructive" className="text-[10px]">
                    Error
                  </Badge>
                )}
              </div>
              <div className="space-y-3 p-3">
                <div>
                  <div className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
                    Code
                  </div>
                  <pre className="overflow-x-auto rounded-md border border-border/60 bg-muted/40 p-3 text-xs font-mono leading-relaxed">
                    {block.code}
                  </pre>
                </div>
                {block.result.stdout && (
                  <div>
                    <div className="mb-1 text-[10px] uppercase tracking-wider text-emerald-600 dark:text-emerald-400">
                      stdout
                    </div>
                    <pre className="overflow-x-auto rounded-md border border-emerald-500/20 bg-emerald-500/5 p-3 text-xs font-mono leading-relaxed text-emerald-700 dark:text-emerald-300">
                      {block.result.stdout}
                    </pre>
                  </div>
                )}
                {block.result.stderr && (
                  <div>
                    <div className="mb-1 text-[10px] uppercase tracking-wider text-red-600 dark:text-red-400">
                      stderr
                    </div>
                    <pre className="overflow-x-auto rounded-md border border-red-500/20 bg-red-500/5 p-3 text-xs font-mono leading-relaxed text-red-700 dark:text-red-300">
                      {block.result.stderr}
                    </pre>
                  </div>
                )}
                {blockSubcalls > 0 && (
                  <div className="space-y-3 rounded-md border border-fuchsia-500/20 bg-fuchsia-500/5 p-3">
                    {block.result.rlm_calls.map((nestedCall, nestedIndex) => (
                      <SubLMCallCard
                        key={`${depth}-${blockIndex}-${nestedIndex}`}
                        call={nestedCall}
                        index={nestedIndex}
                        originLabel={`Subagent iteration ${iteration.iteration}, block ${blockIndex + 1}`}
                        depth={depth + 1}
                      />
                    ))}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function SubLMCallCard({ call, index, originLabel, depth = 0 }: SubLMCallCardProps) {
  const isRecursiveSubagent = Boolean(call.metadata);
  const nestedIterations = call.metadata?.iterations ?? [];
  const nestedSubcalls = countTrajectorySubcalls(nestedIterations);
  const nestedCodeBlocks = countTrajectoryCodeBlocks(nestedIterations);
  const nestedFinalAnswer = extractTrajectoryFinalAnswer(call.metadata);
  const usageTotals = extractUsageTotals(call.usage_summary);
  const prompt = formatPrompt(call.prompt);
  const borderClass = isRecursiveSubagent
    ? 'border-fuchsia-500/30 bg-fuchsia-500/5 dark:border-fuchsia-400/30 dark:bg-fuchsia-400/5'
    : 'border-border bg-background';

  return (
    <Card className={cn('overflow-hidden', borderClass, depth > 0 && 'shadow-none')}>
      <CardHeader className="py-3 px-4">
        <div className="flex items-start justify-between gap-3 flex-wrap">
          <div className="space-y-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <span className={cn('w-2 h-2 rounded-full', isRecursiveSubagent ? 'bg-fuchsia-500 dark:bg-fuchsia-400' : 'bg-sky-500 dark:bg-sky-400')} />
              {isRecursiveSubagent ? `RLM Subagent #${index + 1}` : `LLM Call #${index + 1}`}
            </CardTitle>
            <div className="flex flex-wrap gap-2">
              {originLabel && (
                <Badge variant="outline" className="text-[10px] font-mono">
                  {originLabel}
                </Badge>
              )}
              {call.root_model && (
                <Badge variant="outline" className="text-[10px] font-mono">
                  {call.root_model}
                </Badge>
              )}
              <Badge variant="outline" className="text-[10px] font-mono">
                {call.execution_time.toFixed(2)}s
              </Badge>
              {(usageTotals.inputTokens > 0 || usageTotals.outputTokens > 0) && (
                <>
                  <Badge variant="outline" className="text-[10px] font-mono">
                    {usageTotals.inputTokens} in
                  </Badge>
                  <Badge variant="outline" className="text-[10px] font-mono">
                    {usageTotals.outputTokens} out
                  </Badge>
                </>
              )}
              {usageTotals.totalCalls > 0 && (
                <Badge variant="outline" className="text-[10px] font-mono">
                  {usageTotals.totalCalls} LM calls
                </Badge>
              )}
            </div>
          </div>
          <div className="flex flex-wrap gap-2 justify-end">
            <Badge className={cn(
              'text-[10px]',
              isRecursiveSubagent
                ? 'bg-fuchsia-500/15 text-fuchsia-600 dark:text-fuchsia-400 border-fuchsia-500/30'
                : 'bg-sky-500/15 text-sky-600 dark:text-sky-400 border-sky-500/30'
            )}>
              {isRecursiveSubagent ? 'Recursive Subagent' : 'Direct LM Query'}
            </Badge>
            {isRecursiveSubagent && (
              <Badge variant="outline" className="text-[10px] font-mono">
                nested sub-LM: {nestedSubcalls > 0 ? `yes (${nestedSubcalls})` : 'no'}
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="px-4 pb-4 space-y-3">
        <div>
          <p className="text-xs text-muted-foreground mb-1.5 font-medium uppercase tracking-wider">
            Prompt
          </p>
          <div className="bg-muted/50 rounded-lg p-3 border border-border">
            {isRecursiveSubagent ? (
              <PromptDisclosure
                summary={summarizePromptValue(call.prompt)}
                content={prompt}
              />
            ) : (
              <pre className="max-h-40 overflow-y-auto text-xs whitespace-pre-wrap font-mono">
                {prompt}
              </pre>
            )}
          </div>
        </div>
        <div>
          <p className="text-xs text-muted-foreground mb-1.5 font-medium uppercase tracking-wider">
            Response
          </p>
          <div className="bg-background rounded-lg p-3 max-h-56 overflow-y-auto border border-border">
            <pre className="text-xs whitespace-pre-wrap font-mono text-foreground/90">
              {call.response}
            </pre>
          </div>
        </div>
        {isRecursiveSubagent && (
          <div className="rounded-lg border border-fuchsia-500/20 bg-fuchsia-500/5 p-3 space-y-3">
            <div className="flex flex-wrap gap-2">
              <Badge variant="outline" className="text-[10px] font-mono">
                {nestedIterations.length} iteration{nestedIterations.length === 1 ? '' : 's'}
              </Badge>
              <Badge variant="outline" className="text-[10px] font-mono">
                {nestedCodeBlocks} code block{nestedCodeBlocks === 1 ? '' : 's'}
              </Badge>
              <Badge variant="outline" className="text-[10px] font-mono">
                spawned nested sub-LM queries: {nestedSubcalls > 0 ? 'yes' : 'no'}
              </Badge>
              {nestedFinalAnswer && (
                <Badge className="bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30 text-[10px]">
                  Final answer available
                </Badge>
              )}
            </div>
            {nestedFinalAnswer && (
              <div className="rounded-md border border-emerald-500/20 bg-emerald-500/5 p-3 text-xs text-foreground/90">
                {nestedFinalAnswer}
              </div>
            )}
            <div className="space-y-3">
              {nestedIterations.map((iteration) => {
                const iterationSubcalls = countIterationSubcalls(iteration);
                return (
                  <details
                    key={iteration.iteration}
                    open={depth === 0 && iteration.iteration === 1}
                    className="rounded-lg border border-border/70 bg-background"
                  >
                    <summary className="cursor-pointer list-none px-3 py-2 text-sm font-medium">
                      <div className="flex flex-wrap items-center gap-2">
                        <span>Iteration {iteration.iteration}</span>
                        <Badge variant="outline" className="text-[10px] font-mono">
                          {iteration.code_blocks.length} code
                        </Badge>
                        <Badge variant="outline" className="text-[10px] font-mono">
                          {iterationSubcalls} nested sub-LM
                        </Badge>
                        {iteration.final_answer && (
                          <Badge className="bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30 text-[10px]">
                            Final
                          </Badge>
                        )}
                      </div>
                    </summary>
                    <div className="border-t border-border/60 px-3 py-3">
                      <SubagentIterationView iteration={iteration} depth={depth + 1} />
                    </div>
                  </details>
                );
              })}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
