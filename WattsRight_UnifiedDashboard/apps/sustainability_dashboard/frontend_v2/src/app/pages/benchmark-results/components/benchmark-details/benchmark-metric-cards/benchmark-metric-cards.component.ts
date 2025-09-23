import { Component, Input } from '@angular/core';
import { NgIf } from '@angular/common';
import { BenchmarkMetric } from '@app/types/benchmark.types';
import { BenchmarkMetricCardComponent } from './benchmark-metric-card/benchmark-metric-card.component';

@Component({
  selector: 'app-benchmark-metric-cards',
  standalone: true,
  imports: [NgIf, BenchmarkMetricCardComponent],
  templateUrl: './benchmark-metric-cards.component.html',
  styleUrls: ['./benchmark-metric-cards.component.scss']
})
export class BenchmarkMetricCardsComponent {
  @Input() metrics:
    | { power?: BenchmarkMetric; performance?: BenchmarkMetric; emissions?: BenchmarkMetric; compute?: BenchmarkMetric }
    | null = null;
}
