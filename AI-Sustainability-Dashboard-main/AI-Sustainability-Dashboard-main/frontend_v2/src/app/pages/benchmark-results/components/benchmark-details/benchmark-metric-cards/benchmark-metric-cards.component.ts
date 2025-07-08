import {Component, Input} from '@angular/core';
import {
    MetricCardComponent
} from "@app/pages/pruning-adjustments/components/pruning-results/pruning-metric-cards/metric-card/metric-card.component";
import {
  BenchmarkMetricCardComponent
} from '@app/pages/benchmark-results/components/benchmark-details/benchmark-metric-cards/benchmark-metric-card/benchmark-metric-card.component';
import {BenchmarkMetricCardList} from '@app/types/pruning.types';
import {AsyncPipe} from '@angular/common';

@Component({
  selector: 'app-benchmark-metric-cards',
  imports: [
    MetricCardComponent,
    BenchmarkMetricCardComponent,
    AsyncPipe
  ],
  templateUrl: './benchmark-metric-cards.component.html',
  styleUrl: './benchmark-metric-cards.component.scss'
})
export class BenchmarkMetricCardsComponent {

  @Input() metrics: BenchmarkMetricCardList | undefined;

}
