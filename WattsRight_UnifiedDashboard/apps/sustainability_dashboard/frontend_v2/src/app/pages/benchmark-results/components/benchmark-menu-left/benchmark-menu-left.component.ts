import {Component, EventEmitter, Input, Output} from '@angular/core';
import {ButtonDirective} from "@app/domains/ui/directives/button/button.directive";
import {DecimalPipe, TitleCasePipe, UpperCasePipe} from "@angular/common";
import {BenchmarkData} from '@app/types/pruning.types';

@Component({
  selector: 'app-benchmark-menu-left',
  imports: [
    ButtonDirective,
    UpperCasePipe,
    DecimalPipe,
    TitleCasePipe
  ],
  templateUrl: './benchmark-menu-left.component.html',
  styleUrls: ['./benchmark-menu-left.component.scss']
})
export class BenchmarkMenuLeftComponent {

  @Input() data!: BenchmarkData | undefined;
  @Output() export = new EventEmitter<void>();
  @Output() goBack = new EventEmitter<void>();

  public get changeColor(): string {
    const original = this.data?.originalParameters;
    const pruned = this.data?.prunedParameters;

    if (!original || !pruned) {
      return 'var(--color-grayish)';
    }

    if (pruned > original) {
      return '#DC2626';
    } else if (pruned < original) {
      return '#16A34A';
    } else {
      return 'var(--color-grayish)';
    }
  }
}
