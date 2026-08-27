import { Component, ElementRef, OnDestroy, OnInit, ViewChild } from '@angular/core';
import { ButtonDirective } from '@app/domains/ui/directives/button/button.directive';
import { Subscription, interval } from 'rxjs';
import { Router } from '@angular/router';
import { NgFor, NgIf, NgClass } from '@angular/common';
import { WebsocketService } from '@app/services/websocket.service';
import { SettingsService } from '@app/services/settings.service';
import { UploadService } from '@app/services/upload.service';
import { BenchmarkService } from '@app/services/benchmark.service';

interface LoadingStage {
  label: string;
  description: string;
  status: 'pending' | 'active' | 'completed';
}

@Component({
  selector: 'app-benchmark-loader',
  standalone: true,
  imports: [ButtonDirective, NgFor, NgIf, NgClass],
  templateUrl: './benchmark-loader.component.html',
  styleUrls: ['./benchmark-loader.component.scss']
})
export class BenchmarkLoaderComponent implements OnInit, OnDestroy {
  @ViewChild('progressCircle', { static: true }) progressCircle!: ElementRef;

  public message: string = 'Preparing benchmark…';
  public currentHint = '';
  public isGenerative = false;
  private subscription: Subscription = new Subscription();
  private hintIndex = 0;

  public stages: LoadingStage[] = [];

  private readonly generativeStages: LoadingStage[] = [
    { label: 'Applying pruning at selected threshold', description: 'Removing low-magnitude weights at your chosen threshold', status: 'pending' },
    { label: 'Benchmarking selected threshold', description: 'Evaluating pruned model with power measurement', status: 'pending' },
    { label: 'Measuring energy usage', description: 'Sampling GPU power draw and computing kWh', status: 'pending' },
    { label: 'Generating comparisons', description: 'Creating text generation examples and token analysis', status: 'pending' },
    { label: 'Finalizing results', description: 'Computing CO₂ emissions and preparing visualization', status: 'pending' },
  ];

  private readonly classificationStages: LoadingStage[] = [
    { label: 'Pruning model', description: 'Applying threshold', status: 'pending' },
    { label: 'Running benchmark', description: 'Evaluating performance', status: 'pending' },
    { label: 'Computing metrics', description: 'Finalizing results', status: 'pending' },
  ];

  private readonly educationalHints: string[] = [
    'Real benchmarking validates the surrogate predictions with actual model inference.',
    'Power measurement uses GPU sensors to calculate actual energy consumption per inference call.',
    'CO₂ emissions are estimated based on the energy grid carbon intensity at your selected location.',
    'Comparing predicted vs actual results helps calibrate trust in the surrogate model.',
    'Token-level analysis reveals how pruning affects individual prediction confidence.',
    'Pruning affects models differently: Phi-2 degrades 1,200× while OPT-350M only 3.3× at the same threshold.',
    'A surrogate model predicts perplexity using just 2 anchor evaluations, 98% fewer than full grid search.',
    'Energy consumption during inference stays mostly hardware-constant regardless of pruning due to GPU parallelism.',
    'GPT-2 (124M params) becomes unusable after just 20% parameter removal; larger models tolerate more pruning.',
    'WikiText-2 evaluates 290K+ tokens per model to ensure statistical reliability at each pruning threshold.',
    'Bayesian smart sampling selects 31 representative thresholds vs 101 exhaustive evaluations 69% fewer runs.',
    'OPT-350M is 8× more pruning-resistant than GPT-2 due to its weight distribution, despite identical thresholds.',
    'Pruning 38% of GPT-2\'s weights at threshold 2% caused a 50× perplexity jump the tradeoff can be sharp.',
    'A Gaussian Process surrogate outperforms naive log-linear interpolation by leveraging cross-model patterns.',
  ];

  constructor(
    private readonly router: Router,
    private readonly websocketService: WebsocketService,
    private readonly uploadService: UploadService,
    private readonly settingsService: SettingsService,
    private readonly benchmarkService: BenchmarkService
  ) {}

  ngOnInit() {
    const fromQuery = this.router.parseUrl(this.router.url).queryParams['upload_id'];
    const uploadId = fromQuery
      || this.uploadService?.uploadIdValue
      || null;

    if (!uploadId) {
      console.error('[benchmark-loader] Missing upload_id, going back to start');
      this.router.navigate(['/']);
      return;
    }

    this.isGenerative = this.uploadService.modelTypeValue === 'generative';
    this.stages = this.isGenerative
      ? this.generativeStages.map(s => ({ ...s }))
      : this.classificationStages.map(s => ({ ...s }));

    if (this.stages.length) {
      this.stages[0].status = 'active';
    }

    // Rotate educational hints
    if (this.isGenerative) {
      this.currentHint = this.educationalHints[0];
      this.subscription.add(
        interval(6000).subscribe(() => {
          this.hintIndex = (this.hintIndex + 1) % this.educationalHints.length;
          this.currentHint = this.educationalHints[this.hintIndex];
        })
      );
    }

    this.benchmarkService.setUploadId(uploadId);
    this.websocketService.connect(uploadId);

    const threshold = this.settingsService.Threshold;
    const gpu = this.settingsService.Gpu;
    const location = this.settingsService.Location;

    this.subscription.add(
      this.websocketService.getConnectionStatus().subscribe(status => {
        if (status === 'connected') {
          setTimeout(() => {
            this.websocketService.sendMessage({
              event: 'benchmark_real',
              data: { upload_id: uploadId, threshold, gpu, location, model_type: this.uploadService.modelTypeValue }
            });
          }, 200);
        }
      })
    );

    const isGenerative = this.uploadService.modelTypeValue === 'generative';

    this.subscription.add(
      this.websocketService.getMessages().subscribe(msg => {
        if (msg.message) {
          this.message = msg.message;
          this.updateStageFromMessage(msg.message);
        }
        if (msg?.type === 'benchmark-complete') {
          this.message = msg.message || 'Benchmark complete.';
          const target = isGenerative ? '/generative-results' : '/benchmark-results';
          // Complete stages one by one with a 1 second delay each
          this.completeStagesSequentially().then(() => {
            setTimeout(() => {
              this.router.navigate([target], {
                replaceUrl: true,
                queryParams: { upload_id: uploadId, threshold }
              });
            }, 1000);
          });
        }
      })
    );
  }

  private currentStageIndex = 0;

  private updateStageFromMessage(message: string): void {
    const lower = message.toLowerCase();
    let stage = this.currentStageIndex;

    // Match from highest stage first to prevent regression
    if (lower.includes('final') || lower.includes('saving') || lower.includes('co2') || lower.includes('benchmark complete')) {
      stage = 4;
    } else if (lower.includes('generating') || lower.includes('token-level') || lower.includes('completion example') || lower.includes('comparison')) {
      stage = 3;
    } else if (lower.includes('energy') || lower.includes('power') || lower.includes('sampling gpu')) {
      stage = 2;
    } else if (lower.includes('evaluating original') || lower.includes('evaluating pruned') || lower.includes('original eval') || lower.includes('benchmark')) {
      stage = 1;
    } else if (lower.includes('pruning model') || lower.includes('pruning at') || lower.includes('applying')) {
      stage = 0;
    }

    // Only advance forward, never go backward
    if (stage > this.currentStageIndex) {
      this.currentStageIndex = stage;
      this.activateStage(stage);
    } else if (this.currentStageIndex === 0 && stage === 0) {
      this.activateStage(0);
    }
  }

  private activateStage(index: number): void {
    if (index >= this.stages.length) return;

    for (let i = 0; i < this.stages.length; i++) {
      if (i < index) this.stages[i].status = 'completed';
      else if (i === index) this.stages[i].status = 'active';
      else this.stages[i].status = 'pending';
    }
  }

  get completedStages(): number {
    return this.stages.filter(s => s.status === 'completed').length;
  }

  get progressPercent(): number {
    if (!this.stages.length) return 0;
    const completed = this.completedStages;
    const activeBonus = this.stages.some(s => s.status === 'active') ? 0.5 : 0;
    return Math.min(100, ((completed + activeBonus) / this.stages.length) * 100);
  }

  private completeStagesSequentially(): Promise<void> {
    return new Promise<void>((resolve) => {
      let idx = this.currentStageIndex;
      const completeNext = () => {
        if (idx < this.stages.length) {
          this.stages[idx].status = 'completed';
          idx++;
          if (idx < this.stages.length) {
            this.stages[idx].status = 'active';
          }
          setTimeout(completeNext, 1000);
        } else {
          resolve();
        }
      };
      completeNext();
    });
  }

  onCancel(): void {
    this.router.navigate(['/']);
  }

  ngOnDestroy() {
    this.subscription.unsubscribe();
  }
}
