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
import {Router} from '@angular/router';
import {
  PruningResultsComponent
} from '@app/pages/pruning-adjustments/components/pruning-results/pruning-results.component';
import {
  PruningMenuLeftComponent
} from '@app/pages/pruning-adjustments/components/pruning-menu-left/pruning-menu-left.component';

@Component({
  selector: 'app-pruning-adjustments',
  imports: [
    PruningResultsComponent,
    PruningMenuLeftComponent,
  ],
  templateUrl: './pruning-adjustments.component.html',
  styleUrl: './pruning-adjustments.component.scss'
})
export class PruningAdjustmentsComponent implements OnInit, OnDestroy {

  public isMobileMenuOpen = false;
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
    private readonly router: Router,
    private readonly cdr: ChangeDetectorRef
  ) {
  }

  ngOnInit() {
    if (!this.uploadService.uploadId) {
      this.router.navigate(['/']);
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
    if (!this.uploadService.uploadId.value || !this.settingsFormGroup.controls.gpu.value || !this.settingsFormGroup.controls.location.value || !this.settingsFormGroup.controls.metric.value) {
      return;
    }

    firstValueFrom(
      this.pruningDataService.fetchData(
        this.uploadService.uploadId.value!,
        this.settingsFormGroup.controls.gpu.value!,
        this.settingsFormGroup.controls.location.value!,
        this.settingsFormGroup.controls.metric.value!
      ).pipe(
        map((data) => {
          const transformKeys = (obj: Record<string, number>) => {
            const newObj: Record<string, number> = {};
            Object.entries(obj).forEach(([key, value]) => {
              const numKey = parseFloat(key);
              const newKey = numKey % 1 === 0 ? numKey.toString() : key;
              newObj[newKey] = value;
            });
            return newObj;
          };

          return {
            performance: transformKeys(data.performance),
            power: transformKeys(data.power),
            emissions: transformKeys(data.emissions),
            tflops: transformKeys(data.tflops)
          };
        }),
      )
    ).then((data) => {
      this.metricCards.performance.values = data.performance;
      this.metricCards.power.values = data.power;
      this.metricCards.emissions.values = data.emissions;
      this.metricCards.compute.values = data.tflops;

      this.pruningDataService.Data = data;

      this.settingsFormGroup.controls.threshold.setValue(0);
      this.settingsService.Threshold = 0;
    })
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
