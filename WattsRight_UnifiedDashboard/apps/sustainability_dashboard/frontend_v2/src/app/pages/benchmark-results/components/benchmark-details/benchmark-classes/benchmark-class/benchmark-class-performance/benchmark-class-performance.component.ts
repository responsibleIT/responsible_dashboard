import {Component, Input} from '@angular/core';
import {MetricData} from '@app/types/pruning.types';
import {TitleCasePipe} from '@angular/common';
import {
  BenchmarkMetricBarComponent
} from '@app/pages/benchmark-results/components/benchmark-details/benchmark-classes/benchmark-class/benchmark-class-performance/benchmark-metric-bar/benchmark-metric-bar.component';

@Component({
  selector: 'app-benchmark-class-performance',
  imports: [
    TitleCasePipe,
    BenchmarkMetricBarComponent
  ],
  templateUrl: './benchmark-class-performance.component.html',
  styleUrl: './benchmark-class-performance.component.scss'
})
export class BenchmarkClassPerformanceComponent {

  @Input() metricName: string;
  @Input() classPerformance: MetricData;
  @Input() maxValue: number;

  public get pctChange(): number {
    return (this.classPerformance.pruned - this.classPerformance.original) / this.classPerformance.original * 100;
  }

}
