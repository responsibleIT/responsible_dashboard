import {Component, EventEmitter, Input, OnInit, Output} from '@angular/core';
import {
  PruningMetricCardsComponent
} from '@app/pages/pruning-adjustments/components/pruning-results/pruning-metric-cards/pruning-metric-cards.component';
import {
  PruningChartsComponent
} from '@app/pages/pruning-adjustments/components/pruning-results/pruning-details/pruning-charts/pruning-charts.component';
import {PruningClassPerformance, PruningMetricCardList, PruningTab} from '@app/types/pruning.types';

@Component({
  selector: 'app-pruning-results',
  standalone: true,
  imports: [
    PruningMetricCardsComponent,
    PruningChartsComponent,
  ],
  templateUrl: './pruning-results.component.html',
  styleUrl: './pruning-results.component.scss'
})
export class PruningResultsComponent implements OnInit {

  @Input() activeTab: PruningTab;
  @Input() metricCards: PruningMetricCardList;
  @Input() classPerformance: PruningClassPerformance[];

  ngOnInit() {
    console.log('[results] mounted'); // probe
  }

  // @Output() tabChange = new EventEmitter<PruningTab>();

  // public onTabChange(tab: PruningTab): void {
  //   this.tabChange.emit(tab);
  // }

}
