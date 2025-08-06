import {Component, Input} from '@angular/core';
import {
  MetricCardComponent
} from '@app/pages/pruning-adjustments/components/pruning-results/pruning-metric-cards/metric-card/metric-card.component';
import {PruningMetricCardList} from '@app/types/pruning.types';

@Component({
  selector: 'app-pruning-metric-cards',
  imports: [
    MetricCardComponent,
  ],
  templateUrl: './pruning-metric-cards.component.html',
  styleUrl: './pruning-metric-cards.component.scss'
})
export class PruningMetricCardsComponent {

  @Input() metrics: PruningMetricCardList

}
