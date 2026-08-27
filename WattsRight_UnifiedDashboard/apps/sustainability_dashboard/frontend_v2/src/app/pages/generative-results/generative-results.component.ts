import { Component, OnInit } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { GenerativeMockDataService } from '@app/services/generative-mock-data.service';
import { GenerativeDataService } from '@app/services/generative-data.service';
import { GenerativeDashboardData, RunMetrics, SurrogateInfo } from '@app/types/generative.types';
import { BenchmarkMetric } from '@app/types/benchmark.types';
import { BenchmarkMetricCardComponent } from '@app/pages/benchmark-results/components/benchmark-details/benchmark-metric-cards/benchmark-metric-card/benchmark-metric-card.component';
import { ChartComponent } from '@app/pages/pruning-adjustments/components/pruning-results/pruning-details/pruning-charts/components/chart/chart.component';
import { ButtonDirective } from '@app/domains/ui/directives/button/button.directive';
import { HttpClient } from '@angular/common/http';
import { environment } from '@env/environment';

@Component({
  selector: 'app-generative-results',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    BenchmarkMetricCardComponent,
    ChartComponent,
  ],
  templateUrl: './generative-results.component.html',
  styleUrls: ['./generative-results.component.scss'],
})
export class GenerativeResultsComponent implements OnInit {
  loading = true;
  data: GenerativeDashboardData | null = null;

  // Active tab/section for layout scalability
  activeSection: 'summary' | 'analysis' | 'generation' | 'transparency' = 'summary';

  // Collapsible state for token examples
  expandedExamples: boolean[] = [];

  // Custom prompt generation
  customPrompt = '';
  customGenerationHistory: { prompt: string; originalCompletion: string; prunedCompletion: string }[] = [];

  // Export state
  exportMessage: string | null = null;

  // KPI cards (base vs pruned at selected threshold)
  perplexityCard: BenchmarkMetric | null = null;
  energyCard: BenchmarkMetric | null = null;
  latencyCard: BenchmarkMetric | null = null;
  flopsCard: BenchmarkMetric | null = null;
  co2Card: BenchmarkMetric | null = null;

  // Chart data (Record<threshold, value>)
  perplexityChart: Record<number, number> = {};
  energyChart: Record<number, number> = {};
  latencyChart: Record<number, number> = {};
  flopsChart: Record<number, number> = {};
  tokensPerSecChart: Record<number, number> = {};
  co2Chart: Record<number, number> = {};

  // Uncertainty band for perplexity chart
  perplexityUpper: Record<number, number> = {};
  perplexityLower: Record<number, number> = {};

  // Length bucket table
  lengthBuckets: GenerativeDashboardData['lengthBuckets'] = [];

  // Loss distribution
  lossDistribution: GenerativeDashboardData['lossDistribution'] | null = null;

  // Delta distribution
  deltaDistribution: GenerativeDashboardData['deltaDistribution'] | null = null;

  // Examples
  examples: GenerativeDashboardData['examples'] = [];

  // Surrogate info (knee point, Pareto front)
  surrogateInfo: SurrogateInfo | null = null;

  // Predicted vs actual comparison (from benchmark)
  benchmarkData: any = null;
  predictedPerplexity: number | null = null;
  actualPerplexity: number | null = null;
  predictedFlopsReduction: number | null = null;
  actualFlopsReduction: number | null = null;

  // Text generation examples (original vs pruned)
  textExamples: any[] = [];

  // Generation example (simple prompt completion)
  generationExample: { prompt: string; originalCompletion: string; prunedCompletion: string } | null = null;

  // Currently selected threshold for KPI comparison
  selectedThreshold = 4;

  // Context passed from pruning-adjustments
  gpuLabel = '-';
  locationLabel = '-';
  thresholdFromSettings: number | null = null;

  private uploadId: string | null = null;
  private readonly apiBase = `${environment.api.schema}://${environment.api.hostname}`;

  constructor(
    private readonly mockDataService: GenerativeMockDataService,
    private readonly generativeService: GenerativeDataService,
    private readonly route: ActivatedRoute,
    private readonly router: Router,
    private readonly http: HttpClient,
  ) {}

  ngOnInit(): void {
    this.uploadId = this.route.snapshot.queryParamMap.get('upload_id');
    const thresholdParam = this.route.snapshot.queryParamMap.get('threshold');

    if (thresholdParam !== null) {
      const parsed = parseFloat(thresholdParam);
      if (!isNaN(parsed)) {
        this.thresholdFromSettings = parsed;
        // Map the pruning threshold to the closest run index
        this.selectedThreshold = this.findClosestRunIndex(parsed);
      }
    }

    if (this.uploadId) {
      // Try real backend first, fall back to mock
      this.generativeService.fetchData(this.uploadId).subscribe((data) => {
        if (data) {
          this.hydrate(data);
        } else {
          this.loadMockData();
        }
      });

      // Also fetch actual benchmark data for predicted vs actual comparison
      this.http.get<any>(`${this.apiBase}/benchmark/${this.uploadId}`).subscribe({
        next: (bench) => {
          this.benchmarkData = bench;
          this.actualPerplexity = bench?.overall?.perplexity?.pruned ?? bench?.metricCards?.performance?.pruned ?? null;
          this.actualFlopsReduction = bench?.metricCards?.compute ? (
            bench.metricCards.compute.original > 0
              ? (1 - bench.metricCards.compute.pruned / bench.metricCards.compute.original) * 100
              : null
          ) : null;

          // Populate text examples from benchmark data
          if (Array.isArray(bench?.textExamples)) {
            this.textExamples = bench.textExamples;
            this.expandedExamples = new Array(this.textExamples.length).fill(false);
            if (this.expandedExamples.length > 0) {
              this.expandedExamples[this.expandedExamples.length - 1] = true;
            }
          }

          // Populate generation example
          if (bench?.generationExample) {
            this.generationExample = bench.generationExample;
          }

          // Find predicted values at the benchmark threshold
          if (this.data && this.thresholdFromSettings !== null) {
            const idx = this.findClosestRunIndex(this.thresholdFromSettings);
            const run = this.data.runs[idx];
            if (run) {
              this.predictedPerplexity = run.perplexity;
              this.predictedFlopsReduction = run.flopsReductionPct;
            }
          }
        },
        error: () => { /* benchmark not yet run that's fine */ },
      });
    } else {
      this.loadMockData();
    }
  }

  private findClosestRunIndex(threshold: number): number {
    // Map a threshold value (0.0–10.0) to the closest run array index
    // Runs are at 0.0, 0.1, 0.2, ..., 10.0 (101 entries)
    if (!this.data) {
      return Math.min(100, Math.max(0, Math.round(threshold * 10)));
    }
    let bestIdx = 0;
    let bestDist = Infinity;
    for (let i = 0; i < this.data.runs.length; i++) {
      const dist = Math.abs(this.data.runs[i].threshold - threshold);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  private loadMockData(): void {
    this.mockDataService.fetchEvaluationData().subscribe((data) => {
      this.hydrate(data);

      // Populate generation-tab and analysis-tab data from mock service
      this.textExamples = this.mockDataService.getTextExamples();
      this.generationExample = this.mockDataService.getGenerationExample();
      this.expandedExamples = new Array(this.textExamples.length).fill(false);
      if (this.expandedExamples.length > 0) {
        this.expandedExamples[this.expandedExamples.length - 1] = true;
      }

      // Simulate benchmark comparison data for predicted-vs-actual sidebar
      const run = data.runs[this.selectedThreshold];
      if (run) {
        this.predictedPerplexity = run.perplexity;
        this.predictedFlopsReduction = run.flopsReductionPct;
        this.actualPerplexity = run.perplexity * (1 + (Math.random() * 0.06 - 0.03));
        this.actualFlopsReduction = run.flopsReductionPct * (1 + (Math.random() * 0.04 - 0.02));
        this.benchmarkData = { mock: true };
      }
    });
  }

  private hydrate(data: GenerativeDashboardData): void {
    this.data = data;

    // Build chart series from runs
    for (const run of data.runs) {
      this.perplexityChart[run.threshold] = run.perplexity;
      this.energyChart[run.threshold] = run.energyKwhPer1kCalls;
      this.latencyChart[run.threshold] = run.latencyMs;
      this.flopsChart[run.threshold] = run.flops / 1e9; // display as GFLOPs
      this.tokensPerSecChart[run.threshold] = run.tokensPerSec;
      this.co2Chart[run.threshold] = run.co2KgPer1kCalls * 1000; // display as gCO2

      // Uncertainty band for perplexity
      const std = run.perplexityStd ?? 0;
      this.perplexityUpper[run.threshold] = run.perplexity + std;
      this.perplexityLower[run.threshold] = Math.max(0, run.perplexity - std);
    }

    // Set sub-data
    this.lengthBuckets = data.lengthBuckets;
    this.lossDistribution = data.lossDistribution;
    this.deltaDistribution = data.deltaDistribution;
    this.examples = data.examples;
    this.surrogateInfo = data.surrogateInfo ?? null;

    // If surrogate recommends a knee, use it as default selected threshold
    if (this.surrogateInfo && this.thresholdFromSettings === null) {
      this.selectedThreshold = this.findClosestRunIndex(this.surrogateInfo.kneeThreshold);
    }

    // Build KPI cards at default threshold
    this.updateCards();

    // Set predicted values for comparison (if benchmark already loaded)
    const run = data.runs[this.selectedThreshold];
    if (run) {
      this.predictedPerplexity = run.perplexity;
      this.predictedFlopsReduction = run.flopsReductionPct;
    }

    // Initialize collapsible state for text examples (last open by default)
    this.expandedExamples = new Array(this.textExamples.length).fill(false);
    if (this.expandedExamples.length > 0) {
      this.expandedExamples[this.expandedExamples.length - 1] = true;
    }

    this.loading = false;
  }

  onThresholdChange(event: Event): void {
    const value = +(event.target as HTMLInputElement).value;
    this.selectedThreshold = value;
    this.updateCards();
  }

  private updateCards(): void {
    if (!this.data) return;
    const base = this.data.runs[0];
    const selected = this.data.runs[this.selectedThreshold] ?? this.data.runs[this.data.runs.length - 1];
    if (!base || !selected) return;

    this.perplexityCard = {
      title: 'Perplexity',
      unit: 'PPL',
      original: base.perplexity,
      pruned: selected.perplexity,
    };
    this.energyCard = {
      title: 'Energy (per 1k calls)',
      unit: 'kWh',
      original: base.energyKwhPer1kCalls,
      pruned: selected.energyKwhPer1kCalls,
    };
    this.latencyCard = {
      title: 'Latency',
      unit: 'ms',
      original: base.latencyMs,
      pruned: selected.latencyMs,
    };
    this.flopsCard = {
      title: 'FLOPs',
      unit: 'GFLOPs',
      original: base.flops / 1e9,
      pruned: selected.flops / 1e9,
    };
    this.co2Card = {
      title: 'CO₂ (per 1k calls)',
      unit: 'gCO₂',
      original: base.co2KgPer1kCalls * 1000,
      pruned: selected.co2KgPer1kCalls * 1000,
    };
  }

  toggleExample(index: number): void {
    this.expandedExamples[index] = !this.expandedExamples[index];
  }

  exportModel(): void {
    if (!this.uploadId) {
      this.exportMessage = 'No upload ID available for export.';
      return;
    }
    this.exportMessage = 'Preparing download…';
    window.open(`${this.apiBase}/api/export/${this.uploadId}`, '_blank');
    this.exportMessage = 'Download started. Check your downloads folder.';
  }

  // Generation loading state
  generatingCustom = false;

  generateCustom(): void {
    if (!this.customPrompt?.trim()) return;

    // If no upload_id (mock flow), generate a placeholder response
    if (!this.uploadId) {
      this.customGenerationHistory.unshift({
        prompt: this.customPrompt.trim(),
        originalCompletion: 'Custom generation is only available when running a real benchmark. In the test flow, use the benchmark generation example below to see side-by-side comparison.',
        prunedCompletion: 'Custom generation is only available when running a real benchmark. In the test flow, use the benchmark generation example below to see side-by-side comparison.',
      });
      this.customPrompt = '';
      return;
    }

    this.generatingCustom = true;
    const promptText = this.customPrompt.trim();
    this.customPrompt = '';

    this.http.post<any>(`${this.apiBase}/api/generate`, {
      upload_id: this.uploadId,
      prompt: promptText,
    }).subscribe({
      next: (result) => {
        this.generatingCustom = false;
        this.customGenerationHistory.unshift({
          prompt: promptText,
          originalCompletion: result?.original ?? 'Generation not available.',
          prunedCompletion: result?.pruned ?? 'Generation not available.',
        });
      },
      error: (err) => {
        this.generatingCustom = false;
        const msg = err?.error?.error || 'Generation failed. Ensure a benchmark has been run for this model.';
        this.customGenerationHistory.unshift({
          prompt: promptText,
          originalCompletion: msg,
          prunedCompletion: msg,
        });
      },
    });
  }

  goBack(): void {
    this.router.navigateByUrl('/pruning-adjustments');
  }
}
