import {Component} from '@angular/core';
import {
  ChartComponent
} from '@app/pages/pruning-adjustments/components/pruning-results/pruning-details/pruning-charts/components/chart/chart.component';
import {PruningDataService} from '@app/services/pruning-data.service';
import {map} from 'rxjs';
import {AsyncPipe, UpperCasePipe} from '@angular/common';

@Component({
  selector: 'app-pruning-charts',
  imports: [
    ChartComponent,
    AsyncPipe,
    UpperCasePipe
  ],
  templateUrl: './pruning-charts.component.html',
  styleUrl: './pruning-charts.component.scss'
})
export class PruningChartsComponent {

  public powerChartData = this.pruningDataService.data$.pipe(
    map(data => data ? data.power : null)
  )

  public emissionsChartData = this.pruningDataService.data$.pipe(
    map(data => data ? data.emissions : null)
  )

  public performanceChartData = this.pruningDataService.data$.pipe(
    map(data => data ? data.performance : null)
  )

  public computeChartData = this.pruningDataService.data$.pipe(
    map(data => data ? data.tflops : null)
  )

  constructor(
    private readonly pruningDataService: PruningDataService,
  ) {
  }

}
