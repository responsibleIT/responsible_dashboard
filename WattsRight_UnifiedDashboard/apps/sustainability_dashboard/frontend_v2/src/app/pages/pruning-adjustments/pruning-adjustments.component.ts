import {ChangeDetectorRef, Component, HostListener, OnDestroy, OnInit} from '@angular/core';
import {
  PruneSettingsFormGroup,
  PruningClassPerformance,
  PruningMetricCardList,
  PruningTab
} from '@app/types/pruning.types';
import {FormBuilder, Validators} from '@angular/forms';
import {firstValueFrom, map, Subscription} from 'rxjs';
import {SettingsService} from '@app/services/settings.service';
import {PruningDataService} from '@app/services/pruning-data.service';
import {UploadService} from '@app/services/upload.service';
import {GenerativeDataService} from '@app/services/generative-data.service';
import {SurrogateInfo} from '@app/types/generative.types';
import {Router} from '@angular/router';
import {
  PruningResultsComponent
} from '@app/pages/pruning-adjustments/components/pruning-results/pruning-results.component';
import {
  PruningMenuLeftComponent
} from '@app/pages/pruning-adjustments/components/pruning-menu-left/pruning-menu-left.component';

@Component({
  selector: 'app-pruning-adjustments',
  standalone: true,
  imports: [
    PruningResultsComponent,
    PruningMenuLeftComponent,
  ],
  templateUrl: './pruning-adjustments.component.html',
  styleUrls: ['./pruning-adjustments.component.scss']
})
export class PruningAdjustmentsComponent implements OnInit, OnDestroy {

  public isMobileMenuOpen = false;
  public isGenerative = false;
  public surrogateInfo: SurrogateInfo | null = null;

  // Perplexity chart overlays (generative only)
  public perplexityUpper: Record<number, number> = {};
  public perplexityLower: Record<number, number> = {};
  public kneeThreshold: number | null = null;
  public gpus: { value: string, label: string }[] = [];
  public locations: { value: string, label: string }[] = [];
  public metrics: { value: string, label: string }[] = [];
  public activeTab: PruningTab = 'Charts';
  public metricCards: PruningMetricCardList = {
    'power': {
      title: 'Power (per 1000 calls)',
      unit: 'kWh',
      values: []
    },
    'performance': {
      title: 'Predicted accuracy',
      unit: '%',
      values: []
    },
    'emissions': {
      title: 'Carbon footprint (per 1000 calls)',
      unit: 'gCO2',
      values: []
    },
    'compute': {
      title: 'Computing Power',
      unit: 'TFLOPS',
      values: []
    }
  };

  public classPerformance: PruningClassPerformance[] = [
    {
      className: 'Positive',
      unit: '%',
      original: 0.70,
      pruned: {
        0: 0.70,
        0.1: 0.71,
      }
    },
    {
      className: 'Neutral',
      unit: '%',
      original: 0.72,
      pruned: {
        0: 0.72,
        0.1: 0.71
      }
    },
    {
      className: 'Negative',
      unit: '%',
      original: 0.71,
      pruned: {
        0: 0.71,
        0.1: 0.67,
      }
    }
  ]

  public settingsFormGroup: PruneSettingsFormGroup = this.formBuilder.group({
    gpu: this.formBuilder.control<string | null>(null, [Validators.required]),
    location: this.formBuilder.control<string | null>(null, [Validators.required]),
    metric: this.formBuilder.control<string | null>(null, [Validators.required]),
    threshold: this.formBuilder.control<number>(0, [Validators.required]),
  });

  private subscriptions: Subscription = new Subscription();

  constructor(
    private readonly formBuilder: FormBuilder,
    private readonly settingsService: SettingsService,
    private readonly pruningDataService: PruningDataService,
    private readonly uploadService: UploadService,
    private readonly generativeDataService: GenerativeDataService,
    private readonly router: Router,
    private readonly cdr: ChangeDetectorRef
  ) {
  }

  ngOnInit() {
    if (!this.uploadService.upload_id) {
      this.router.navigate(['/']);
    }

    this.isGenerative = this.uploadService.modelTypeValue === 'generative';
    console.log('[pruning-adjustments] modelType =', this.uploadService.modelTypeValue, '| isGenerative =', this.isGenerative);

    if (this.isGenerative) {
      this.metricCards.performance = {
        title: 'Predicted perplexity',
        unit: 'PPL',
        values: []
      };

      // Fetch surrogate info for knee recommendation
      const uid = this.uploadService.upload_id.value;
      if (uid) {
        this.generativeDataService.fetchData(uid).subscribe((data) => {
          if (data?.surrogateInfo) {
            this.surrogateInfo = data.surrogateInfo;

            // Knee threshold for vertical marker on chart
            this.kneeThreshold = data.surrogateInfo.kneeThreshold;

            // Build uncertainty band from runs
            for (const run of data.runs) {
              const std = run.perplexityStd ?? 0;
              this.perplexityUpper[run.threshold] = run.perplexity + std;
              this.perplexityLower[run.threshold] = Math.max(0, run.perplexity - std);
            }

            this.cdr.detectChanges();
          }
        });
      }
    }

    this.subscriptions.add(this.settingsFormGroup.controls.threshold.valueChanges.subscribe(threshold => {
      if (threshold === null) {
        return;
      }

      this.settingsService.Threshold = threshold;
    }))

    this.subscriptions.add(this.settingsFormGroup.controls.gpu.valueChanges.subscribe(gpu => {
      if (gpu === null) {
        return;
      }

      this.settingsService.Gpu = gpu;
      this.cdr.detectChanges();
      this.loadPruningData();
    }))

    this.subscriptions.add(this.settingsFormGroup.controls.location.valueChanges.subscribe(location => {
      if (location === null) {
        return;
      }

      this.settingsService.Location = location;
      this.cdr.detectChanges();
      this.loadPruningData();
    }))

    this.subscriptions.add(this.settingsFormGroup.controls.metric.valueChanges.subscribe(metric => {
      if (metric === null) {
        return;
      }

      this.cdr.detectChanges();
      this.loadPruningData();
    }))

    this.loadSettings()
  }

  private loadSettings(): void {
    firstValueFrom(this.pruningDataService.fetchSettings()).then((settings) => {
      this.gpus = settings.gpus.map((gpu) => ({
        value: gpu,
        label: gpu
      }));
      this.locations = settings.locations.map((location) => ({
        value: location,
        label: location
      }));
      this.metrics = settings.metrics.map((metric) => ({
        value: metric,
        label: metric
      }));

      this.settingsFormGroup.controls.gpu.setValue(this.gpus[0].value);
      this.settingsFormGroup.controls.location.setValue(this.locations[0].value);
      this.settingsFormGroup.controls.metric.setValue(this.metrics[0].value);
    }).then(() => {
      this.loadPruningData()
    })
  }

  private loadPruningData(): void {
    if (!this.uploadService.upload_id.value) return;
    const gpu = this.settingsFormGroup.controls.gpu.value!;
    const location = this.settingsFormGroup.controls.location.value!;
    const metric = this.settingsFormGroup.controls.metric.value!;
    if (!gpu || !location || !metric) return;

    firstValueFrom(
      this.pruningDataService.fetchData(
        this.uploadService.upload_id.value!, gpu, location, metric
      ).pipe(
        map((data) => {
          if (!data) return { performance:{}, power:{}, emissions:{}, tflops:{} };
          const toStrKeys = (obj: Record<string, number>) => {
            const out: Record<string, number> = {};
            Object.entries(obj || {}).forEach(([k, v]) => out[String(k)] = Number(v ?? 0));
            return out;
          };
          const result: any = {
            performance: toStrKeys(data.performance),
            power: toStrKeys(data.power),
            emissions: toStrKeys(data.emissions),
            tflops: toStrKeys(data.tflops),
          };
          if ((data as any).perplexity) {
            result.perplexity = toStrKeys((data as any).perplexity);
          }
          return result;
        }),
      )
    ).then((data) => {
      console.log('[chart-data]', data); // <- quick sanity log

      // For generative models, use perplexity instead of accuracy for the performance card
      if (this.isGenerative && data.perplexity && Object.keys(data.perplexity).length > 0) {
        this.metricCards.performance.values = data.perplexity;
      } else {
        this.metricCards.performance.values = data.performance;
      }
      this.metricCards.power.values = data.power;
      this.metricCards.emissions.values = data.emissions;
      this.metricCards.compute.values = data.tflops;

      this.pruningDataService.Data = data;

      // ensure the view updates
      this.cdr.detectChanges();

      this.settingsFormGroup.controls.threshold.setValue(0);
      this.settingsService.Threshold = 0;
    });
  }

  toggleMobileMenu() {
    this.isMobileMenuOpen = !this.isMobileMenuOpen;

    if (window.innerWidth <= 768) {
      if (this.isMobileMenuOpen) {
        document.body.style.overflow = 'hidden';
      } else {
        document.body.style.overflow = '';
      }
    }
  }

  @HostListener('window:resize', ['$event'])
  onResize(event: any) {
    if (event.target.innerWidth > 768 && this.isMobileMenuOpen) {
      this.isMobileMenuOpen = false;
      document.body.style.overflow = ''; // Reset body scroll
    }
  }

  @HostListener('document:keydown.escape', ['$event'])
  onEscapeKey(event: KeyboardEvent) {
    if (this.isMobileMenuOpen) {
      this.toggleMobileMenu();
    }
  }

  onTabChange(newTab: PruningTab): void {
    this.activeTab = newTab;
  }

  ngOnDestroy() {
    this.subscriptions.unsubscribe();
  }

}
