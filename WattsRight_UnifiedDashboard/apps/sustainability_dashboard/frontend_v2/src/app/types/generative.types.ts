// ─── Generative Model Evaluation Types ───
// Structures for WikiText-style language modeling perplexity evaluation.
// Mock data mirrors what the real backend will eventually produce.

export interface RunMetrics {
  threshold: number;
  sparsity: number;
  perplexity: number;
  perplexityStd?: number;
  perplexityDeltaPct: number;
  crossEntropy: number;
  flops: number;
  flopsReductionPct: number;
  latencyMs: number;
  tokensPerSec: number;
  energyKwhPer1kCalls: number;
  co2KgPer1kCalls: number;
  memoryMb: number;
}

export interface LossDistribution {
  bins: number[];
  baseCounts: number[];
  prunedCounts: number[];
  percentiles: {
    base: { p50: number; p90: number; p99: number };
    pruned: { p50: number; p90: number; p99: number };
  };
}

export interface DeltaDistribution {
  bins: string[];
  percentages: number[];
}

export interface LengthBucket {
  range: string;
  avgLength: number;
  basePerplexity: number;
  prunedPerplexity: number;
}

export interface ExampleSequence {
  text: string;
  baseTokenLogprobs: number[];
  prunedTokenLogprobs: number[];
  deltaLoss: number;
}

export interface UsageBase {
  energyKwhPer1kCalls: number;
  co2KgPer1kCalls: number;
  latencyMs: number;
}

export interface ParetoPoint {
  threshold: number;
  perplexity: number;
}

export interface SurrogateInfo {
  kneeThreshold: number;
  kneePerplexity: number;
  kneeFlopsReduction: number;
  paretoIndices: number[];
  paretoFront?: ParetoPoint[];
  anchorsUsed: number[];
  isPreset: boolean;
}

export interface GenerativeDashboardData {
  runs: RunMetrics[];
  lossDistribution: LossDistribution;
  deltaDistribution: DeltaDistribution;
  lengthBuckets: LengthBucket[];
  examples: ExampleSequence[];
  usageBase: UsageBase;
  surrogateInfo?: SurrogateInfo;
}
