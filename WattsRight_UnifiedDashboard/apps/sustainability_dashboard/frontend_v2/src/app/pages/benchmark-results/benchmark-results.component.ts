import {ChangeDetectorRef, Component, HostListener, OnInit} from '@angular/core';
import {
  BenchmarkMenuLeftComponent
} from '@app/pages/benchmark-results/components/benchmark-menu-left/benchmark-menu-left.component';
import {
  BenchmarkDetailsComponent
} from '@app/pages/benchmark-results/components/benchmark-details/benchmark-details.component';
import {BenchmarkService} from '@app/services/benchmark.service';
import {Router} from '@angular/router';
import {BehaviorSubject, firstValueFrom} from 'rxjs';
import {BenchmarkData, BenchmarkMetricCardList, ClassPerformance} from '@app/types/pruning.types';
import {UploadService} from '@app/services/upload.service';
import {AsyncPipe} from '@angular/common';

@Component({
  selector: 'app-benchmark-results',
  imports: [
    BenchmarkMenuLeftComponent,
    BenchmarkDetailsComponent,
    AsyncPipe,
  ],
  templateUrl: './benchmark-results.component.html',
  styleUrl: './benchmark-results.component.scss'
})
export class BenchmarkResultsComponent implements OnInit {

  private uploadId: string | null;

  public isMobileMenuOpen = false;
  public benchmarkData: BenchmarkData | undefined;
  public metricCards$ = new BehaviorSubject<BenchmarkMetricCardList | null>(null);
  public classPerformances$ = new BehaviorSubject<ClassPerformance[]>([]);

  constructor(
    private cdr: ChangeDetectorRef,
    private readonly router: Router,
    private readonly benchmarkService: BenchmarkService,
    private readonly uploadService: UploadService,
  ) {
  }

  ngOnInit() {
    this.uploadId = this.uploadService.uploadIdValue;

    if (!this.uploadId) {
      this.router.navigate(['/']);
    }

    firstValueFrom(this.benchmarkService.fetchData(this.uploadId!)).then(data => {
      this.benchmarkData = data

      const newMetricCards: BenchmarkMetricCardList = {
        'power': {
          title: 'Power (per 1000 calls)',
          unit: 'kWh',
          original: data.metricCards.power.original,
          pruned: data.metricCards.power.pruned,
          change: (data.metricCards.power.pruned - data.metricCards.power.original) / data.metricCards.power.original * 100
        },
        'performance': {
          title: 'Accuracy',
          unit: '%',
          original: data.metricCards.performance.original,
          pruned: data.metricCards.performance.pruned,
          change: data.metricCards.performance.pruned - data.metricCards.performance.original / data.metricCards.performance.original * 100
        },
        'emissions': {
          title: 'Carbon (per 1000 calls)',
          unit: 'gCO2',
          original: data.metricCards.emissions.original,
          pruned: data.metricCards.emissions.pruned,
          change: (data.metricCards.emissions.pruned - data.metricCards.emissions.original) / data.metricCards.emissions.original * 100
        },
        'compute': {
          title: 'Computing Power',
          unit: 'TFLOPS',
          original: data.metricCards.compute.original,
          pruned: data.metricCards.compute.pruned,
          change: (data.metricCards.compute.pruned - data.metricCards.compute.original) / data.metricCards.compute.original * 100
        }
      }

      this.metricCards$.next(newMetricCards);

      let classPerformances: ClassPerformance[] = [];
      if (data.perClass) {
        classPerformances = Object.entries(data.perClass).map(([className, performance]) => ({
          className,
          performance: performance
        }));
      }
      this.classPerformances$.next(classPerformances);
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

}
