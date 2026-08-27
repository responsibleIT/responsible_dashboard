import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, of, delay, tap } from 'rxjs';
import {
  GenerativeDashboardData,
  RunMetrics,
  LossDistribution,
  DeltaDistribution,
  LengthBucket,
  ExampleSequence,
  UsageBase,
  SurrogateInfo,
} from '@app/types/generative.types';

// ─── Static Mock Data ────────────────────────────────────────────────
// Values mimic a real WikiText-2 perplexity evaluation with unstructured
// pruning applied at increasing thresholds.  Trends:
//   higher pruning → lower FLOPs / energy / memory
//   higher pruning → higher perplexity (gradual degradation)

const MOCK_RUNS: RunMetrics[] = [
  {
    threshold: 0,
    sparsity: 0,
    perplexity: 18.2,
    perplexityStd: 0.3,
    perplexityDeltaPct: 0,
    crossEntropy: 2.9,
    flops: 1.0e12,
    flopsReductionPct: 0,
    latencyMs: 120,
    tokensPerSec: 45,
    energyKwhPer1kCalls: 0.42,
    co2KgPer1kCalls: 0.18,
    memoryMb: 5200,
  },
  {
    threshold: 1,
    sparsity: 0.12,
    perplexity: 18.4,
    perplexityStd: 0.4,
    perplexityDeltaPct: 1.1,
    crossEntropy: 2.91,
    flops: 8.8e11,
    flopsReductionPct: 12,
    latencyMs: 110,
    tokensPerSec: 50,
    energyKwhPer1kCalls: 0.38,
    co2KgPer1kCalls: 0.16,
    memoryMb: 4800,
  },
  {
    threshold: 2,
    sparsity: 0.25,
    perplexity: 18.9,
    perplexityStd: 0.5,
    perplexityDeltaPct: 3.8,
    crossEntropy: 2.94,
    flops: 7.5e11,
    flopsReductionPct: 25,
    latencyMs: 95,
    tokensPerSec: 60,
    energyKwhPer1kCalls: 0.31,
    co2KgPer1kCalls: 0.13,
    memoryMb: 4100,
  },
  {
    threshold: 3,
    sparsity: 0.38,
    perplexity: 19.5,
    perplexityStd: 0.7,
    perplexityDeltaPct: 7.1,
    crossEntropy: 2.97,
    flops: 6.2e11,
    flopsReductionPct: 38,
    latencyMs: 82,
    tokensPerSec: 72,
    energyKwhPer1kCalls: 0.26,
    co2KgPer1kCalls: 0.11,
    memoryMb: 3500,
  },
  {
    threshold: 4,
    sparsity: 0.5,
    perplexity: 20.1,
    perplexityStd: 0.9,
    perplexityDeltaPct: 10.4,
    crossEntropy: 3.0,
    flops: 5.0e11,
    flopsReductionPct: 50,
    latencyMs: 70,
    tokensPerSec: 82,
    energyKwhPer1kCalls: 0.22,
    co2KgPer1kCalls: 0.09,
    memoryMb: 3000,
  },
  {
    threshold: 5,
    sparsity: 0.6,
    perplexity: 21.3,
    perplexityStd: 1.2,
    perplexityDeltaPct: 17.0,
    crossEntropy: 3.06,
    flops: 4.0e11,
    flopsReductionPct: 60,
    latencyMs: 60,
    tokensPerSec: 94,
    energyKwhPer1kCalls: 0.18,
    co2KgPer1kCalls: 0.07,
    memoryMb: 2500,
  },
  {
    threshold: 6,
    sparsity: 0.7,
    perplexity: 23.4,
    perplexityStd: 1.8,
    perplexityDeltaPct: 28.6,
    crossEntropy: 3.15,
    flops: 3.0e11,
    flopsReductionPct: 70,
    latencyMs: 52,
    tokensPerSec: 108,
    energyKwhPer1kCalls: 0.14,
    co2KgPer1kCalls: 0.06,
    memoryMb: 2000,
  },
  {
    threshold: 7,
    sparsity: 0.78,
    perplexity: 27.1,
    perplexityStd: 2.5,
    perplexityDeltaPct: 48.9,
    crossEntropy: 3.3,
    flops: 2.2e11,
    flopsReductionPct: 78,
    latencyMs: 45,
    tokensPerSec: 122,
    energyKwhPer1kCalls: 0.11,
    co2KgPer1kCalls: 0.05,
    memoryMb: 1600,
  },
  {
    threshold: 8,
    sparsity: 0.85,
    perplexity: 33.8,
    perplexityStd: 3.6,
    perplexityDeltaPct: 85.7,
    crossEntropy: 3.52,
    flops: 1.5e11,
    flopsReductionPct: 85,
    latencyMs: 40,
    tokensPerSec: 135,
    energyKwhPer1kCalls: 0.08,
    co2KgPer1kCalls: 0.03,
    memoryMb: 1200,
  },
  {
    threshold: 9,
    sparsity: 0.92,
    perplexity: 48.5,
    perplexityStd: 5.2,
    perplexityDeltaPct: 166.5,
    crossEntropy: 3.88,
    flops: 8.0e10,
    flopsReductionPct: 92,
    latencyMs: 35,
    tokensPerSec: 148,
    energyKwhPer1kCalls: 0.05,
    co2KgPer1kCalls: 0.02,
    memoryMb: 900,
  },
];

const MOCK_LOSS_DISTRIBUTION: LossDistribution = {
  bins: [0, 1, 2, 3, 4, 5, 6],
  baseCounts: [120, 340, 500, 300, 120, 40],
  prunedCounts: [80, 300, 520, 360, 180, 90],
  percentiles: {
    base: { p50: 2.8, p90: 4.5, p99: 6.2 },
    pruned: { p50: 3.0, p90: 5.2, p99: 7.8 },
  },
};

const MOCK_DELTA_DISTRIBUTION: DeltaDistribution = {
  bins: ['<-10%', '-10% to 0%', '0 to +10%', '+10% to +25%', '>+25%'],
  percentages: [3, 22, 45, 20, 10],
};

const MOCK_LENGTH_BUCKETS: LengthBucket[] = [
  { range: 'short (1-64 tokens)',   avgLength: 30,  basePerplexity: 17.5, prunedPerplexity: 18.0 },
  { range: 'medium (65-192 tokens)', avgLength: 120, basePerplexity: 18.3, prunedPerplexity: 20.0 },
  { range: 'long (193-384 tokens)',  avgLength: 300, basePerplexity: 19.5, prunedPerplexity: 23.5 },
  { range: 'very long (385+ tokens)', avgLength: 500, basePerplexity: 21.0, prunedPerplexity: 28.2 },
];

const MOCK_EXAMPLES: ExampleSequence[] = [
  {
    text: 'The novel was published in 1939 by Harper & Brothers.',
    baseTokenLogprobs:   [-0.20, -0.30, -0.10, -0.50, -0.20, -0.40, -0.15, -0.25, -0.35],
    prunedTokenLogprobs: [-0.30, -0.60, -0.20, -1.20, -0.40, -0.80, -0.35, -0.55, -0.70],
    deltaLoss: 1.8,
  },
  {
    text: 'He later became one of the most influential figures in physics.',
    baseTokenLogprobs:   [-0.10, -0.20, -0.30, -0.20, -0.15, -0.25, -0.10, -0.35, -0.40, -0.20],
    prunedTokenLogprobs: [-0.50, -0.80, -0.60, -0.90, -0.45, -0.70, -0.55, -1.10, -0.95, -0.60],
    deltaLoss: 2.5,
  },
  {
    text: 'The committee voted unanimously to approve the new regulations.',
    baseTokenLogprobs:   [-0.25, -0.15, -0.40, -0.20, -0.30, -0.10, -0.35, -0.20],
    prunedTokenLogprobs: [-0.40, -0.35, -0.70, -0.50, -0.65, -0.30, -0.80, -0.45],
    deltaLoss: 1.4,
  },
  {
    text: 'In 2014 , the population of the city was estimated at 1 @.@ 2 million .',
    baseTokenLogprobs:   [-0.30, -0.50, -0.20, -0.60, -0.45, -0.35, -0.80, -0.55, -0.25, -0.40, -0.70],
    prunedTokenLogprobs: [-0.60, -1.20, -0.45, -1.50, -0.90, -0.75, -1.60, -1.10, -0.55, -0.85, -1.40],
    deltaLoss: 3.2,
  },
  {
    text: 'The species is found in tropical forests across Southeast Asia.',
    baseTokenLogprobs:   [-0.15, -0.25, -0.20, -0.35, -0.10, -0.30, -0.45, -0.20],
    prunedTokenLogprobs: [-0.30, -0.55, -0.40, -0.80, -0.25, -0.65, -0.90, -0.50],
    deltaLoss: 1.6,
  },
];

const MOCK_USAGE_BASE: UsageBase = {
  energyKwhPer1kCalls: 0.22,
  co2KgPer1kCalls: 0.09,
  latencyMs: 70,
};

const MOCK_SURROGATE_INFO: SurrogateInfo = {
  kneeThreshold: 4,
  kneePerplexity: 20.1,
  kneeFlopsReduction: 50,
  paretoIndices: [0, 1, 2, 3, 4, 5],
  paretoFront: [
    { threshold: 0, perplexity: 18.2 },
    { threshold: 1, perplexity: 18.4 },
    { threshold: 2, perplexity: 18.9 },
    { threshold: 3, perplexity: 19.5 },
    { threshold: 4, perplexity: 20.1 },
    { threshold: 5, perplexity: 21.3 },
  ],
  anchorsUsed: [0, 2, 4, 6, 8],
  isPreset: true,
};

const MOCK_TEXT_EXAMPLES = [
  {
    bucket: 'Short',
    numTokens: 42,
    deltaPct: 3.2,
    text: 'The novel was published in 1939 by Harper & Brothers and became an immediate bestseller.',
    originalPerplexity: 15.8,
    prunedPerplexity: 16.3,
    positions: [
      {
        context: '...published in',
        actualToken: '1939',
        original: {
          actualRank: 1,
          actualProb: 0.3142,
          topTokens: [
            { token: '1939', prob: 0.3142, isActual: true },
            { token: '1940', prob: 0.1821, isActual: false },
            { token: 'the', prob: 0.0952, isActual: false },
            { token: '1938', prob: 0.0741, isActual: false },
            { token: '1941', prob: 0.0523, isActual: false },
          ],
        },
        pruned: {
          actualRank: 2,
          actualProb: 0.1987,
          topTokens: [
            { token: '1940', prob: 0.2215, isActual: false },
            { token: '1939', prob: 0.1987, isActual: true },
            { token: 'the', prob: 0.1102, isActual: false },
            { token: '1941', prob: 0.0834, isActual: false },
            { token: '1938', prob: 0.0621, isActual: false },
          ],
        },
      },
      {
        context: '...by Harper',
        actualToken: '&',
        original: {
          actualRank: 1,
          actualProb: 0.7823,
          topTokens: [
            { token: '&', prob: 0.7823, isActual: true },
            { token: 'and', prob: 0.1245, isActual: false },
            { token: ',', prob: 0.0312, isActual: false },
          ],
        },
        pruned: {
          actualRank: 1,
          actualProb: 0.6541,
          topTokens: [
            { token: '&', prob: 0.6541, isActual: true },
            { token: 'and', prob: 0.2103, isActual: false },
            { token: ',', prob: 0.0452, isActual: false },
          ],
        },
      },
    ],
  },
  {
    bucket: 'Medium',
    numTokens: 128,
    deltaPct: 8.7,
    text: 'He later became one of the most influential figures in modern physics, contributing to quantum mechanics and general relativity.',
    originalPerplexity: 18.3,
    prunedPerplexity: 19.9,
    positions: [
      {
        context: '...most influential',
        actualToken: 'figures',
        original: {
          actualRank: 1,
          actualProb: 0.4521,
          topTokens: [
            { token: 'figures', prob: 0.4521, isActual: true },
            { token: 'people', prob: 0.2134, isActual: false },
            { token: 'scientists', prob: 0.1012, isActual: false },
          ],
        },
        pruned: {
          actualRank: 1,
          actualProb: 0.3102,
          topTokens: [
            { token: 'figures', prob: 0.3102, isActual: true },
            { token: 'people', prob: 0.2543, isActual: false },
            { token: 'scientists', prob: 0.1321, isActual: false },
          ],
        },
      },
      {
        context: '...contributing to',
        actualToken: 'quantum',
        original: {
          actualRank: 1,
          actualProb: 0.3845,
          topTokens: [
            { token: 'quantum', prob: 0.3845, isActual: true },
            { token: 'the', prob: 0.1923, isActual: false },
            { token: 'modern', prob: 0.0812, isActual: false },
          ],
        },
        pruned: {
          actualRank: 3,
          actualProb: 0.0934,
          topTokens: [
            { token: 'the', prob: 0.2541, isActual: false },
            { token: 'modern', prob: 0.1234, isActual: false },
            { token: 'quantum', prob: 0.0934, isActual: true },
            { token: 'general', prob: 0.0821, isActual: false },
          ],
        },
      },
    ],
  },
  {
    bucket: 'Long',
    numTokens: 256,
    deltaPct: 22.4,
    text: 'The committee voted unanimously to approve the new regulations governing environmental standards for industrial emissions across the European Union member states.',
    originalPerplexity: 19.5,
    prunedPerplexity: 23.9,
    positions: [
      {
        context: '...voted unanimously',
        actualToken: 'to',
        original: {
          actualRank: 1,
          actualProb: 0.8912,
          topTokens: [
            { token: 'to', prob: 0.8912, isActual: true },
            { token: 'in', prob: 0.0412, isActual: false },
            { token: 'on', prob: 0.0234, isActual: false },
          ],
        },
        pruned: {
          actualRank: 1,
          actualProb: 0.7234,
          topTokens: [
            { token: 'to', prob: 0.7234, isActual: true },
            { token: 'in', prob: 0.0823, isActual: false },
            { token: 'on', prob: 0.0512, isActual: false },
          ],
        },
      },
      {
        context: '...standards for',
        actualToken: 'industrial',
        original: {
          actualRank: 2,
          actualProb: 0.1823,
          topTokens: [
            { token: 'the', prob: 0.2341, isActual: false },
            { token: 'industrial', prob: 0.1823, isActual: true },
            { token: 'manufacturing', prob: 0.0912, isActual: false },
          ],
        },
        pruned: {
          actualRank: 5,
          actualProb: 0.0512,
          topTokens: [
            { token: 'the', prob: 0.2812, isActual: false },
            { token: 'all', prob: 0.1234, isActual: false },
            { token: 'manufacturing', prob: 0.0823, isActual: false },
            { token: 'new', prob: 0.0612, isActual: false },
            { token: 'industrial', prob: 0.0512, isActual: true },
          ],
        },
      },
    ],
  },
];

const MOCK_GENERATION_EXAMPLE = {
  prompt: 'The future of renewable energy depends on',
  originalCompletion: 'the development of more efficient storage technologies. Battery innovations, particularly in solid-state and lithium-sulfur designs, could enable grid-scale energy storage that makes solar and wind power reliable around the clock.',
  prunedCompletion: 'the development of better storage solutions. New battery technologies and improved grid management systems could make solar and wind energy more reliable for everyday use across many regions.',
};

// ─── Service ─────────────────────────────────────────────────────────

@Injectable({ providedIn: 'root' })
export class GenerativeMockDataService {
  private loading$ = new BehaviorSubject<boolean>(false);
  private data$ = new BehaviorSubject<GenerativeDashboardData | null>(null);

  get isLoading$(): Observable<boolean> {
    return this.loading$.asObservable();
  }

  get dashboardData$(): Observable<GenerativeDashboardData | null> {
    return this.data$.asObservable();
  }

  /**
   * Simulate an async evaluation call.
   * Returns the full dashboard payload after a realistic delay.
   */
  fetchEvaluationData(): Observable<GenerativeDashboardData> {
    this.loading$.next(true);

    const payload: GenerativeDashboardData = {
      runs: MOCK_RUNS,
      lossDistribution: MOCK_LOSS_DISTRIBUTION,
      deltaDistribution: MOCK_DELTA_DISTRIBUTION,
      lengthBuckets: MOCK_LENGTH_BUCKETS,
      examples: MOCK_EXAMPLES,
      usageBase: MOCK_USAGE_BASE,
      surrogateInfo: MOCK_SURROGATE_INFO,
    };

    // Simulate 400–700 ms network / compute delay
    const delayMs = 400 + Math.floor(Math.random() * 300);

    return of(payload).pipe(
      delay(delayMs),
      tap((data) => {
        this.data$.next(data);
        this.loading$.next(false);
      }),
    );
  }

  /**
   * Return mock text examples for the analysis tab.
   */
  getTextExamples(): any[] {
    return MOCK_TEXT_EXAMPLES;
  }

  /**
   * Return a mock generation example.
   */
  getGenerationExample(): any {
    return MOCK_GENERATION_EXAMPLE;
  }

  /**
   * Return runs filtered to a specific threshold (or all if null).
   */
  getRunsByThreshold(threshold: number | null): RunMetrics[] {
    const all = this.data$.getValue()?.runs ?? [];
    if (threshold === null) return all;
    return all.filter((r) => r.threshold === threshold);
  }

  /**
   * Compute scaled usage projection.
   * @param callsPerDay user-controlled daily call count
   * @param days projection window
   */
  projectUsage(callsPerDay: number, days: number = 30): {
    totalEnergyKwh: number;
    totalCo2Kg: number;
    avgLatencyMs: number;
  } {
    const base = this.data$.getValue()?.usageBase ?? MOCK_USAGE_BASE;
    const totalCalls = callsPerDay * days;
    const factor = totalCalls / 1000;
    return {
      totalEnergyKwh: +(base.energyKwhPer1kCalls * factor).toFixed(3),
      totalCo2Kg: +(base.co2KgPer1kCalls * factor).toFixed(4),
      avgLatencyMs: base.latencyMs,
    };
  }
}
