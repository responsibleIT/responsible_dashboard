import { Component, OnInit } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { UploadService } from '@app/services/upload.service';
import { BenchmarkService } from '@app/services/benchmark.service';
import { CommonModule } from '@angular/common';
import { WebsocketService } from '@app/services/websocket.service';
import {BenchmarkDetailsComponent} from '@app/pages/benchmark-results/components/benchmark-details/benchmark-details.component';
import {BenchmarkData, BenchmarkMetricCardList, ClassPerformance} from '@app/types/pruning.types';

type Pair = { orig: number | null; pruned: number | null };

@Component({
  selector: 'app-benchmark-results',
  standalone: true,
  imports: [CommonModule, BenchmarkDetailsComponent], // reuse existing detail components
  templateUrl: './benchmark-results.component.html',
  styleUrls: ['./benchmark-results.component.scss']
})
export class BenchmarkResultsComponent implements OnInit {
  cards: { power: Pair; performance: Pair; emissions: Pair; compute: Pair } = {
    power: { orig: null, pruned: null },
    performance: { orig: null, pruned: null },
    emissions: { orig: null, pruned: null },
    compute: { orig: null, pruned: null }
  };
  metricCards: BenchmarkMetricCardList | null = null;

  modelName = '—';
  gpuLabel  = '—';
  locationLabel = '—';
  thresholdPct: number | null = null;
  sizeReductionPct: number | null = null; // (1 - pruned/original) * 100 when data available
  originalParameters: number | null = null;
  prunedParameters: number | null = null;

  classes: ClassPerformance[] = [];

  constructor(
    private readonly uploads: UploadService,
    private readonly benchmark: BenchmarkService,
    private readonly router: Router,
    private readonly route: ActivatedRoute,
    private readonly ws: WebsocketService,
  ) {}

  ngOnInit() {
  this.route.data.subscribe((data) => {
    const r = data['benchmark'] as BenchmarkData | null;
    console.log('[BenchmarkResults] resolver delivered data', r);
    if (r) {
      this.hydrate(r);
    } else {
      console.error('[results] no benchmark data resolved');
      // Optionally: navigate back to pruning-adjustments with the same upload_id
      const upload_id = this.route.snapshot.queryParamMap.get('upload_id');
      this.router.navigate(['/pruning-adjustments'], { queryParams: { upload_id } });
    }
  });
}

  private hydrate(res: any): void {
  this.modelName = res?.model ?? res?.modelName ?? '—';
  this.gpuLabel  = res?.gpu ?? res?.gpuLabel ?? '—';
  this.locationLabel = res?.location ?? res?.locationLabel ?? '—';
  this.thresholdPct = typeof res?.threshold === 'number' ? Number(res.threshold) : (typeof res?.pruningThreshold === 'number' ? Number(res.pruningThreshold) : null);

  // parameter counts if present
  this.originalParameters = this.n(res?.originalParameters) || this.n(res?.original_params) || null;
  this.prunedParameters   = this.n(res?.prunedParameters)   || this.n(res?.pruned_params)   || null;

    const mc = res?.metricCards ?? {};
    const pick = (k: string): Pair => ({ orig: this.n(mc?.[k]?.original), pruned: this.n(mc?.[k]?.pruned) });
    this.cards.power       = pick('power');
    this.cards.performance = pick('performance');
    this.cards.emissions   = pick('emissions');
    this.cards.compute     = pick('compute');
    if (mc?.power) {
      this.metricCards = {
        power: { title: 'Power (per 1000 calls)', unit: 'kWh', original: mc.power.original, pruned: mc.power.pruned, change: (mc.power.pruned - mc.power.original) / (mc.power.original || 1) * 100 },
        performance: { title: 'Accuracy', unit: '%', original: mc.performance.original, pruned: mc.performance.pruned, change: (mc.performance.pruned - mc.performance.original)/(mc.performance.original||1)*100 },
        emissions: { title: 'Carbon (per 1000 calls)', unit: 'gCO₂', original: mc.emissions.original, pruned: mc.emissions.pruned, change: (mc.emissions.pruned - mc.emissions.original)/(mc.emissions.original||1)*100 },
        compute: { title: 'Computing Power', unit: 'TFLOPS', original: mc.compute.original, pruned: mc.compute.pruned, change: (mc.compute.pruned - mc.compute.original)/(mc.compute.original||1)*100 }
      };
    }

    // compute size reduction percentage (based on compute or power if compute absent)
    // size reduction: prefer explicit params if available, else fall back to compute/power heuristic
    if (this.originalParameters && this.prunedParameters && this.originalParameters > 0) {
      this.sizeReductionPct = (1 - (this.prunedParameters / this.originalParameters)) * 100;
    } else {
      const baseOrig = this.cards.compute.orig ?? this.cards.power.orig;
      const basePruned = this.cards.compute.pruned ?? this.cards.power.pruned;
      if (typeof baseOrig === 'number' && baseOrig > 0 && typeof basePruned === 'number') {
        this.sizeReductionPct = (1 - (basePruned / baseOrig)) * 100;
      } else {
        this.sizeReductionPct = null;
      }
    }

    // adapt to existing BenchmarkClassesComponent expects ClassPerformance[]: { className, performance }
    if (res?.perClass && typeof res.perClass === 'object') {
      const norm = (v: number | null): number => {
        if (v === null) return 0;
        return v > 1 ? v / 100 : v; // backend might return percentages (e.g., 74.6) we normalize to 0-1
      };
      this.classes = Object.entries(res.perClass).map(([className, perf]: [string, any]) => ({
        className: className,
        performance: {
          accuracy: { original: norm(this.n(perf?.accuracy?.original)), pruned: norm(this.n(perf?.accuracy?.pruned)) },
          precision: { original: norm(this.n(perf?.precision?.original)), pruned: norm(this.n(perf?.precision?.pruned)) },
          recall: { original: norm(this.n(perf?.recall?.original)), pruned: norm(this.n(perf?.recall?.pruned)) },
          f1Score: { original: norm(this.n(perf?.f1Score?.original ?? perf?.f1?.original)), pruned: norm(this.n(perf?.f1Score?.pruned ?? perf?.f1?.pruned)) }
        }
      }));
    }
  }

  // helpers
  n(v: unknown): number | null {
    return (typeof v === 'number' && isFinite(v)) ? v : null;
  }
  hasPair(p: Pair): boolean {
    return typeof p.orig === 'number' || typeof p.pruned === 'number';
  }
  asSci(v: number | null, unit = ''): string {
    if (v === null) return '0.00e+0';
    const exp = v === 0 ? 0 : Math.floor(Math.log10(Math.abs(v)));
    const mant = v / Math.pow(10, exp);
    return `${mant.toFixed(2)}e${exp >= 0 ? '+' : ''}${exp}${unit ? ' ' + unit : ''}`;
  }
  asPct(v: number | null): string {
    if (v === null) return '0.0 %';
    return `${v.toFixed(1)} %`;
  }
  deltaPct(orig: number | null, pruned: number | null): string {
    if (orig === null || pruned === null) return '→ 0.0%';
    const d = ((pruned - orig) / Math.max(Math.abs(orig), 1e-12)) * 100;
    const arrow = d >= 0 ? '↑' : '↓';
    return `${arrow} ${Math.abs(d).toFixed(1)}%`;
  }
  metricDelta(orig: number | null, pruned: number | null): string {
    if (orig === null || pruned === null) return '0.00%';
    const diff = pruned - orig;
    const sign = diff > 0 ? '+' : '';
    return `${sign}${diff.toFixed(2)}%`;
  }

  goBack(): void {
    this.router.navigateByUrl('/pruning-adjustments');
  }

  exportModel() {
    // Placeholder: implement actual export call when backend endpoint available
    console.log('[BenchmarkResults] Export model clicked');
  }
}