import { Component, OnInit } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { UploadService } from '@app/services/upload.service';
import { BenchmarkService } from '@app/services/benchmark.service';
import { CommonModule } from '@angular/common';
import { WebsocketService } from '@app/services/websocket.service';

type Pair = { orig: number | null; pruned: number | null };

@Component({
  selector: 'app-benchmark-results',
  standalone: true,
  imports: [CommonModule], // CommonModule covers NgIf, NgFor, number pipe
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

  modelName = '—';
  gpuLabel  = '—';
  locationLabel = '—';
  thresholdPct: number | null = null;
  sizeReductionPct: number | null = null; // (1 - pruned/original) * 100 when data available
  originalParameters: number | null = null;
  prunedParameters: number | null = null;

  classes: Array<{
    name: string;
    overallAcc: number;
    deltaAcc: number;
    f1: Pair; precision: Pair; recall: Pair; accuracy: Pair;
    expanded: boolean;
  }> = [];

  constructor(
    private readonly uploads: UploadService,
    private readonly benchmark: BenchmarkService,
    private readonly router: Router,
    private readonly route: ActivatedRoute,
    private readonly ws: WebsocketService,
  ) {}

  ngOnInit(): void {
    const qid  = this.route.snapshot.queryParamMap.get('upload_id');
    // 👇 ensure this matches your service naming
    const svcId = (this.uploads as any).upload_id?.value ?? (this.uploads as any).upload_id?.value ?? null;
    const wsId  = this.ws.getUploadId() ?? null;

    const upload_id = qid ?? svcId ?? wsId;

    console.log('[BenchmarkResults] resolved upload_id sources', { qid, svcId, wsId, chosen: upload_id });

    if (!upload_id) {
      console.error('[results] missing upload_id');
      this.router.navigateByUrl('/');
      return;
    }

    this.benchmark.fetchData(upload_id).subscribe({
      next: (res: any) => this.hydrate(res),
      error: (err) => {
        console.error('[results] fetch error', err);
        this.router.navigateByUrl('/');
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

    const cl = res?.classMetrics ?? {};
    this.classes = Array.isArray(cl?.items) ? cl.items.map((it: any, idx: number) => ({
      name: String(it?.name ?? 'Class'),
      overallAcc: this.n(it?.accuracy) ?? 0,
      deltaAcc:   this.n(it?.deltaAccuracy) ?? 0,
      f1:         { orig: this.n(it?.f1?.original),        pruned: this.n(it?.f1?.pruned) },
      precision:  { orig: this.n(it?.precision?.original), pruned: this.n(it?.precision?.pruned) },
      recall:     { orig: this.n(it?.recall?.original),     pruned: this.n(it?.recall?.pruned) },
      accuracy:   { orig: this.n(it?.accuracyOriginal),     pruned: this.n(it?.accuracyPruned) },
      expanded: idx < 2 // open first two by default
    })) : [];
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

  goBack(): void {
    this.router.navigateByUrl('/pruning-adjustments');
  }

  toggleClass(c: any) {
    c.expanded = !c.expanded;
  }

  exportModel() {
    // Placeholder: implement actual export call when backend endpoint available
    console.log('[BenchmarkResults] Export model clicked');
  }
}