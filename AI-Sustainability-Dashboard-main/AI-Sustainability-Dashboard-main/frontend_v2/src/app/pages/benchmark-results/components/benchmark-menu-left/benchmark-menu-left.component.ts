import {Component, Input} from '@angular/core';
import {ButtonDirective} from "@app/domains/ui/directives/button/button.directive";
import {FormInputSliderComponent} from "@app/domains/ui/components/forms/form-input-slider/form-input-slider.component";
import {
    PruningSettingsComponent
} from "@app/pages/pruning-adjustments/components/pruning-menu-left/pruning-settings/pruning-settings.component";
import {DecimalPipe, TitleCasePipe, UpperCasePipe} from "@angular/common";
import {BenchmarkData} from '@app/types/pruning.types';

@Component({
  selector: 'app-benchmark-menu-left',
  imports: [
    ButtonDirective,
    FormInputSliderComponent,
    PruningSettingsComponent,
    UpperCasePipe,
    DecimalPipe,
    TitleCasePipe
  ],
  templateUrl: './benchmark-menu-left.component.html',
  styleUrl: './benchmark-menu-left.component.scss'
})
export class BenchmarkMenuLeftComponent {

  @Input() data!: BenchmarkData | undefined;

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
