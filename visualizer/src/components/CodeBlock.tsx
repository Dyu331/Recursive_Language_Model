'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { cn } from '@/lib/utils';
import { CodeBlock as CodeBlockType } from '@/lib/types';
import { CodeWithLineNumbers } from './CodeWithLineNumbers';
import { SubLMCallCard } from './SubLMCallCard';

interface CodeBlockProps {
  block: CodeBlockType;
  index: number;
}

function formatVariableValue(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }

  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function getVariablePreview(value: string): string {
  const firstLine = value.split('\n')[0] ?? '';
  const trimmed = firstLine.trim();
  if (trimmed.length <= 90) {
    return trimmed || '(empty)';
  }
  return `${trimmed.slice(0, 90)}…`;
}

function VariableCard({ name, value }: { name: string; value: unknown }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const formattedValue = formatVariableValue(value);
  const preview = getVariablePreview(formattedValue);

  return (
    <button
      type="button"
      onClick={() => setIsExpanded((current) => !current)}
      className="min-w-0 overflow-hidden rounded border border-border bg-background px-2 py-1.5 text-left font-mono text-xs transition-colors hover:bg-muted/30"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <span className="break-all text-sky-600 dark:text-sky-400">{name}</span>
          <span className="text-muted-foreground mx-1">=</span>
          <span className="break-words text-amber-600 dark:text-amber-400 [overflow-wrap:anywhere]">
            {isExpanded ? formattedValue : preview}
          </span>
        </div>
        <span className="shrink-0 font-mono text-[10px] text-muted-foreground">
          {isExpanded ? 'Hide' : 'Show'}
        </span>
      </div>
    </button>
  );
}

export function CodeBlock({ block, index }: CodeBlockProps) {
  const [isOpen, setIsOpen] = useState(true);
  const hasError = block.result?.stderr && block.result.stderr.length > 0;
  const hasOutput = block.result?.stdout && block.result.stdout.length > 0;
  const executionTime = block.result?.execution_time 
    ? block.result.execution_time.toFixed(2) 
    : null;

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <Card className={cn(
        'border overflow-hidden transition-all',
        hasError 
          ? 'border-red-500/40 bg-red-500/5 dark:border-red-400/40 dark:bg-red-400/5' 
          : 'border-emerald-500/30 bg-emerald-500/5 dark:border-emerald-400/30 dark:bg-emerald-400/5'
      )}>
        <CollapsibleTrigger asChild>
          <CardHeader className="py-2 px-4 cursor-pointer hover:bg-muted/30 transition-colors">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-emerald-600 dark:text-emerald-400 font-mono text-sm">
                  {'>'}_
                </span>
                <CardTitle className="text-sm font-medium">
                  Code Block #{index + 1}
                </CardTitle>
              </div>
              <div className="flex items-center gap-2">
                {executionTime && (
                  <Badge variant="outline" className="font-mono text-xs">
                    {executionTime}s
                  </Badge>
                )}
                {hasError && (
                  <Badge variant="destructive" className="text-xs">
                    Error
                  </Badge>
                )}
                {hasOutput && !hasError && (
                  <Badge className="bg-emerald-500 text-white dark:bg-emerald-400 dark:text-emerald-950 text-xs">
                    Output
                  </Badge>
                )}
                <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                  <span className="text-xs">{isOpen ? '▼' : '▶'}</span>
                </Button>
              </div>
            </div>
          </CardHeader>
        </CollapsibleTrigger>
        
        <CollapsibleContent>
          <CardContent className="p-0">
            {/* Code */}
            <div className="bg-muted border-t border-border">
              <div className="px-3 py-1.5 border-b border-border/50 flex items-center gap-2">
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">
                  Python
                </span>
              </div>
              <div className="code-block min-w-0 p-4 overflow-hidden">
                <CodeWithLineNumbers code={block.code} language="python" />
              </div>
            </div>

            {/* Output */}
            {hasOutput && (
              <div className="border-t border-border bg-emerald-500/5 dark:bg-emerald-400/5">
                <div className="px-3 py-1.5 border-b border-border/50 flex items-center gap-2">
                  <span className="text-[10px] uppercase tracking-wider text-emerald-600 dark:text-emerald-400 font-medium">
                    stdout
                  </span>
                </div>
                <pre className="code-block min-w-0 whitespace-pre-wrap break-words p-4 [overflow-wrap:anywhere]">
                  <code className="whitespace-pre-wrap break-words text-emerald-700 dark:text-emerald-300 [overflow-wrap:anywhere]">
                    {block.result.stdout}
                  </code>
                </pre>
              </div>
            )}

            {/* Errors */}
            {hasError && (
              <div className="border-t border-border bg-red-500/5 dark:bg-red-400/5">
                <div className="px-3 py-1.5 border-b border-border/50 flex items-center gap-2">
                  <span className="text-[10px] uppercase tracking-wider text-red-600 dark:text-red-400 font-medium">
                    stderr
                  </span>
                </div>
                <pre className="code-block min-w-0 whitespace-pre-wrap break-words p-4 [overflow-wrap:anywhere]">
                  <code className="whitespace-pre-wrap break-words text-red-700 dark:text-red-300 [overflow-wrap:anywhere]">
                    {block.result.stderr}
                  </code>
                </pre>
              </div>
            )}

            {/* Locals */}
            {block.result?.locals && Object.keys(block.result.locals).length > 0 && (
              <div className="border-t border-border bg-muted/50">
                <div className="px-3 py-1.5 border-b border-border/50 flex items-center gap-2">
                  <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">
                    Variables
                  </span>
                </div>
                <div className="grid grid-cols-1 gap-2 p-4 md:grid-cols-2 xl:grid-cols-3">
                  {Object.entries(block.result.locals).map(([key, value]) => (
                    <VariableCard
                      key={key} 
                      name={key}
                      value={value}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Sub-LM Calls */}
            {block.result?.rlm_calls && block.result.rlm_calls.length > 0 && (
              <div className="border-t border-border bg-fuchsia-500/5 dark:bg-fuchsia-400/5">
                <div className="px-3 py-1.5 border-b border-border/50 flex items-center gap-2">
                  <span className="text-[10px] uppercase tracking-wider text-fuchsia-600 dark:text-fuchsia-400 font-medium">
                    Sub-LM Calls ({block.result.rlm_calls.length})
                  </span>
                </div>
                <div className="p-4 space-y-3">
                  {block.result.rlm_calls.map((call, i) => (
                    <SubLMCallCard
                      key={i}
                      call={call}
                      index={i}
                      originLabel={`Code Block #${index + 1}`}
                    />
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}
