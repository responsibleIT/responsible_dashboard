import {Component, Input} from '@angular/core';
import {
  ChartComponent
} from '@app/pages/pruning-adjustments/components/pruning-results/pruning-details/pruning-charts/components/chart/chart.component';
import {PruningDataService} from '@app/services/pruning-data.service';
import {map} from 'rxjs';
import {AsyncPipe, NgIf, UpperCasePipe} from '@angular/common';

@Component({
  selector: 'app-pruning-charts',
  imports: [
    ChartComponent,
    AsyncPipe,
    NgIf,
    UpperCasePipe
  ],
  templateUrl: './pruning-charts.component.html',
  styleUrls: ['./pruning-charts.component.scss']
})
export class PruningChartsComponent {

  @Input() isGenerative = false;
  @Input() perplexityUpper: Record<number, number> = {};
  @Input() perplexityLower: Record<number, number> = {};
  @Input() kneeThreshold: number | null = null;

  public powerChartData = this.pruningDataService.data$.pipe(
    map(data => data ? data.power : null)
  )

  public emissionsChartData = this.pruningDataService.data$.pipe(
    map(data => data ? data.emissions : null)
  )

  public performanceChartData = this.pruningDataService.data$.pipe(
    map(data => data ? data.performance : null)
  )

  public perplexityChartData = this.pruningDataService.data$.pipe(
    map(data => data?.perplexity ? data.perplexity : null)
  )

  public computeChartData = this.pruningDataService.data$.pipe(
    map(data => data ? data.tflops : null)
  )

  constructor(
    private readonly pruningDataService: PruningDataService,
  ) {
  }

}
