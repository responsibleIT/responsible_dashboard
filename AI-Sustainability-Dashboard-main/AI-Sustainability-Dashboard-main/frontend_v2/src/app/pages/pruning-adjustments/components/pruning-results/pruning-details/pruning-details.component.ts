import {Component, EventEmitter, Input, Output} from '@angular/core';
import {NgForOf} from '@angular/common';
import {PruningTab} from '@app/types/pruning.types';
import {PRUNING_TABS} from '@app/constants/pruning.constants';

@Component({
  selector: 'app-pruning-details',
  imports: [
    NgForOf
  ],
  templateUrl: './pruning-details.component.html',
  styleUrl: './pruning-details.component.scss'
})
export class PruningDetailsComponent {

  @Input() activeTab: PruningTab = 'Charts';
  @Output() activeTabChange = new EventEmitter<PruningTab>();

  tabs = PRUNING_TABS

  setActiveTab(tab: PruningTab): void {
    this.activeTab = tab;
    this.activeTabChange.emit(tab);
  }

  isActive(tab: string): boolean {
    return this.activeTab === tab;
  }

}
