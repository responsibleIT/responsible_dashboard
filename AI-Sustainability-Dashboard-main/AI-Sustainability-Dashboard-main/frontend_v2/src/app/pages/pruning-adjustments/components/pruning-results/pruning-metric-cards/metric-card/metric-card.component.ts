import {Component, Input} from '@angular/core';
import {PruningMetricCard} from '@app/types/pruning.types';
import {MetricColor, MetricColorScheme} from '@app/types/metric.types';
import {SettingsService} from '@app/services/settings.service';
import {map, Observable, tap} from 'rxjs';
import {AsyncPipe, DecimalPipe, NgIf, UpperCasePipe} from '@angular/common';

@Component({
  selector: 'app-metric-card',
  imports: [
    AsyncPipe,
    DecimalPipe,
    NgIf,
    UpperCasePipe
  ],
  providers: [DecimalPipe],
  templateUrl: './metric-card.component.html',
  styleUrl: './metric-card.component.scss'
})
export class MetricCardComponent {

  @Input() type: 'power' | 'performance' | 'emissions' | 'compute';
  @Input() metric: PruningMetricCard;
  @Input() decimalFormat: string = '1.0-2';
  @Input() scientificNotation = false;

  public value$: Observable<number> = this.settingsService.threshold.asObservable().pipe(
    map((threshold) => this.metric.values[threshold] ?? 0),
  );

  // Calculate percentage change using threshold 0 as the original value
  public percentageChange$: Observable<number> = this.value$.pipe(
    map(currentValue => {
      const originalValue = this.metric.values[0] ?? 0;
      if (originalValue === 0) return 0;
      return ((currentValue - originalValue) / originalValue) * 100;
    })
  );

  // Determine if the change is positive, negative, or neutral
  public changeType$: Observable<'positive' | 'negative' | 'neutral'> = this.percentageChange$.pipe(
    map(change => {
      if (Math.abs(change) < 0.01) return 'neutral'; // Consider < 0.01% as neutral
      return change > 0 ? 'positive' : 'negative';
    })
  );

  // Get the appropriate arrow icon
  public changeIcon$: Observable<string> = this.changeType$.pipe(
    map(type => {
      switch (type) {
        case 'positive': return '↑';
        case 'negative': return '↓';
        default: return '→';
      }
    })
  );

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

  get formattedValue$(): Observable<string> {
    return this.value$.pipe(
      map(value => {
        if (this.scientificNotation) {
          return value.toExponential(2);
        } else {
          return this.decimalPipe.transform(value, this.decimalFormat) || '0';
        }
      })
    );
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
