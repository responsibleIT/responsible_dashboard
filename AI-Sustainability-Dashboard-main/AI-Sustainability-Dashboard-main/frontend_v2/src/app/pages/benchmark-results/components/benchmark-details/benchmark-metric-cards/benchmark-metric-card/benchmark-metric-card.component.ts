import {Component, Input, OnChanges, SimpleChanges} from '@angular/core';
import {BenchmarkMetricCard} from '@app/types/pruning.types';
import {MetricColor, MetricColorScheme} from '@app/types/metric.types';
import {SettingsService} from '@app/services/settings.service';
import {DecimalPipe, JsonPipe, NgIf, UpperCasePipe} from '@angular/common';

@Component({
  selector: 'app-benchmark-metric-card',
  imports: [
    UpperCasePipe,
    DecimalPipe,
    NgIf,
    JsonPipe
  ],
  providers: [DecimalPipe],
  templateUrl: './benchmark-metric-card.component.html',
  styleUrl: './benchmark-metric-card.component.scss'
})
export class BenchmarkMetricCardComponent implements OnChanges {

  @Input() type: 'power' | 'performance' | 'emissions' | 'compute';
  @Input() metric: BenchmarkMetricCard | undefined;
  @Input() decimalFormat: string = '1.0-2';
  @Input() scientificNotation = false;

  public percentageChange: number | undefined;
  public changeType: 'positive' | 'negative' | 'neutral' | undefined;
  public changeIcon: string | undefined;
  public originalValue: string | undefined;
  public prunedValue: string | undefined;

  ngOnChanges(changes: SimpleChanges) {
    if (changes.metric) {

      const metric = changes.metric.currentValue as BenchmarkMetricCard;
      const originalValue = metric.original ?? 0;
      const prunedValue = metric.pruned ?? 0;
      this.percentageChange = originalValue === 0 ? 0 : ((prunedValue - originalValue) / originalValue) * 100;
      this.changeType = Math.abs(this.percentageChange) < 0.01 ? 'neutral' : (this.percentageChange > 0 ? 'positive' : 'negative');
      this.changeIcon = this.changeType === 'positive' ? '↑' : (this.changeType === 'negative' ? '↓' : '→');

      if (this.scientificNotation) {
        this.originalValue = metric.original.toExponential(2);
        this.prunedValue = metric.pruned.toExponential(2);
        return
      }

      this.originalValue = this.decimalPipe.transform(metric.original, this.decimalFormat) || '0';
      this.prunedValue = this.decimalPipe.transform(metric.pruned, this.decimalFormat) || '0';
    }
  }

  private colorScheme: MetricColorScheme = {
    power: {
      header: '#6B21A8',
      value: '#663EB6',
      background: '#EFE2FE'
    },
    performance: {
      header: '#166534',
      value: '#05DBAC',
      background: '#CBFDEC'
    },
    emissions: {
      header: '#9A3412',
      value: '#EE8438',
      background: '#FFE6BD'
    },
    compute: {
      header: '#991B1B',
      value: '#FE17B0',
      background: '#FEDBEE'
    }
  };

  constructor(
    private readonly settingsService: SettingsService,
    private decimalPipe: DecimalPipe
  ) {
  }

  get color(): MetricColor {
    return this.colorScheme[this.type];
  }

  // Helper method to get percentage change color
  getPercentageChangeColor(changeType: 'positive' | 'negative' | 'neutral' | null): string {
    // For some metrics, positive change might be bad (e.g., power, emissions)
    // For others, positive change might be good (e.g., performance)
    const isGoodMetric = this.type === 'performance'; // performance is good when it increases

    switch (changeType) {
      case 'positive':
        return isGoodMetric ? '#16A34A' : '#DC2626'; // Green for good, red for bad
      case 'negative':
        return isGoodMetric ? '#DC2626' : '#16A34A'; // Red for good metrics going down, green for bad metrics going down
      default:
        return '#6B7280'; // Gray for neutral
    }
  }

}
