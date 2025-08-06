import {Component, Input} from '@angular/core';
import {METRIC_BAR_COLOR_SCHEME} from '@app/constants/metric.constants';
import {DecimalPipe, NgClass, NgIf, TitleCasePipe} from '@angular/common';

@Component({
  selector: 'app-metric-bar',
  imports: [
    TitleCasePipe,
    NgClass,
    DecimalPipe,
    NgIf
  ],
  templateUrl: './metric-bar.component.html',
  styleUrl: './metric-bar.component.scss'
})
export class MetricBarComponent {

  @Input() type: 'original' | 'pruned'
  @Input() value: number;
  @Input() unit: string | null = null;
  @Input() pctChange: number | null = null;

  public get color(): string {
    return METRIC_BAR_COLOR_SCHEME[this.type]
  }
}
