import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';
import { UpperCasePipe, DecimalPipe, NgIf } from '@angular/common';
import { BenchmarkMetric } from '@app/types/benchmark.types';

type ChangeType = 'positive' | 'negative' | 'neutral';

@Component({
  selector: 'app-benchmark-metric-card',
  standalone: true,
  imports: [UpperCasePipe, DecimalPipe, NgIf],
  templateUrl: './benchmark-metric-card.component.html',
  styleUrls: ['./benchmark-metric-card.component.scss']
})
export class BenchmarkMetricCardComponent implements OnChanges {
  @Input() metric: BenchmarkMetric | null = null;
  @Input() scientificNotation = false;
  @Input() type: 'power' | 'performance' | 'emissions' | 'compute' = 'performance';

  originalValue: string | number = 0;
  prunedValue: string | number = 0;

  percentageChange = 0;
  changeType: ChangeType = 'neutral';
  changeIcon = '→';

  private colorScheme = {
    power:        { header: '#6B21A8', value: '#663EB6', background: '#EFE2FE' },
    performance:  { header: '#166534', value: '#05DBAC', background: '#CBFDEC' },
    emissions:    { header: '#9A3412', value: '#EE8438', background: '#FFE6BD' },
    compute:      { header: '#991B1B', value: '#FE17B0', background: '#FEDBEE' }
  };

  get color() { return this.colorScheme[this.type]; }

  ngOnChanges(_: SimpleChanges): void {
    if (!this.metric) {
      this.originalValue = 0;
      this.prunedValue = 0;
      this.percentageChange = 0;
      this.changeType = 'neutral';
      this.changeIcon = '→';
      return;
    }

    const orig = this.metric.original ?? 0;
    const prun = this.metric.pruned ?? 0;

    this.originalValue = this.scientificNotation ? orig.toExponential(2) : orig;
    this.prunedValue   = this.scientificNotation ? prun.toExponential(2) : prun;

    if (orig > 0) {
      this.percentageChange = ((prun - orig) / orig) * 100;
      if (Math.abs(this.percentageChange) < 0.01) {
        this.changeType = 'neutral'; this.changeIcon = '→';
      } else if (this.percentageChange > 0) {
        this.changeType = 'positive'; this.changeIcon = '↑';
      } else {
        this.changeType = 'negative'; this.changeIcon = '↓';
      }
    }
  }

  getPercentageChangeColor(): string {
    const isGood = this.type === 'performance';
    if (this.changeType === 'positive') return isGood ? '#16A34A' : '#DC2626';
    if (this.changeType === 'negative') return isGood ? '#DC2626' : '#16A34A';
    return '#6B7280';
  }
}
