import {Component, Input} from '@angular/core';
import {ClassPerformance} from '@app/types/pruning.types';
import {DecimalPipe, NgClass, NgForOf, NgStyle, TitleCasePipe} from '@angular/common';
import {provideAngularSvgIcon, SvgIconComponent} from 'angular-svg-icon';
import {
  BenchmarkClassPerformanceComponent
} from '@app/pages/benchmark-results/components/benchmark-details/benchmark-classes/benchmark-class/benchmark-class-performance/benchmark-class-performance.component';

@Component({
  selector: 'app-benchmark-class',
  imports: [
    TitleCasePipe,
    SvgIconComponent,
    NgClass,
    NgStyle,
    BenchmarkClassPerformanceComponent,
    DecimalPipe
  ],
  providers: [
    DecimalPipe,
  ],
  templateUrl: './benchmark-class.component.html',
  styleUrl: './benchmark-class.component.scss'
})
export class BenchmarkClassComponent {

  @Input() benchmarkClass: ClassPerformance;

  public collapsed: boolean = true;

  constructor(
    private readonly decimalPipe: DecimalPipe
  ) {
  }

  public toggleCollapsed() {
    this.collapsed = !this.collapsed;
  }

  public get changeColor(): string {
    const originalAccuracy = this.benchmarkClass.performance.accuracy.original;
    const newAccuracy = this.benchmarkClass.performance.accuracy.pruned;

    if (newAccuracy > originalAccuracy) {
      return '#16A34A';
    } else if (newAccuracy < originalAccuracy) {
      return '#DC2626';
    } else {
      return 'var(--color-grayish)';
    }
  }

  public get formattedAccuracyChange(): string {
    const originalAccuracy = this.benchmarkClass.performance.accuracy.original;
    const newAccuracy = this.benchmarkClass.performance.accuracy.pruned;

    const change = ((newAccuracy - originalAccuracy) / originalAccuracy) * 100;

    if (isNaN(change)) {
      return '0%';
    }

    let format = this.decimalPipe.transform(change, '1.0-2') + '%';
    if (change > 0) {
      format = '+' + format;
    }

    return format;
  }

}
