import {Component, Input} from '@angular/core';
import {
  BenchmarkMetricCardsComponent
} from '@app/pages/benchmark-results/components/benchmark-details/benchmark-metric-cards/benchmark-metric-cards.component';
import {BenchmarkMetricCardList, ClassPerformance} from '@app/types/pruning.types';
import {NgIf} from '@angular/common';
import {
  BenchmarkClassesComponent
} from '@app/pages/benchmark-results/components/benchmark-details/benchmark-classes/benchmark-classes.component';

@Component({
  selector: 'app-benchmark-details',
  imports: [
    BenchmarkMetricCardsComponent,
    NgIf,
    BenchmarkClassesComponent
  ],
  templateUrl: './benchmark-details.component.html',
  styleUrl: './benchmark-details.component.scss'
})
export class BenchmarkDetailsComponent {

  @Input() metricCards: BenchmarkMetricCardList | null;
  @Input() classes: ClassPerformance[] = [];

}
