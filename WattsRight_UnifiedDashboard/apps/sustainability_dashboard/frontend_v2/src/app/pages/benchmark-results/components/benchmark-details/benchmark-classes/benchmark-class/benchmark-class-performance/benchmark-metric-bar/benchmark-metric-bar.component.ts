import {Component, Input} from '@angular/core';
import {DecimalPipe, NgIf, TitleCasePipe} from '@angular/common';

@Component({
  selector: 'app-benchmark-metric-bar',
  imports: [
    TitleCasePipe,
    DecimalPipe,
    NgIf
  ],
  templateUrl: './benchmark-metric-bar.component.html',
  styleUrl: './benchmark-metric-bar.component.scss'
})
export class BenchmarkMetricBarComponent {

  @Input() label: string;
  @Input() value: number;
  @Input() maxValue: number;
  @Input() pctChange: number;

  public get positiveChange() {
    return this.pctChange > 0;
  }

  public get negativeChange() {
    return this.pctChange < 0;
  }
}
