import {Component, Input} from '@angular/core';
import {
  PruningSettingsComponent
} from '@app/pages/pruning-adjustments/components/pruning-menu-left/pruning-settings/pruning-settings.component';
import {PruneSettingsFormGroup} from '@app/types/pruning.types';
import {FormInputSliderComponent} from '@app/domains/ui/components/forms/form-input-slider/form-input-slider.component';
import {ButtonDirective} from '@app/domains/ui/directives/button/button.directive';
import {UpperCasePipe} from '@angular/common';
import {Router} from '@angular/router';
import {SettingsService} from '@app/services/settings.service';

@Component({
  selector: 'app-pruning-menu-left',
  imports: [
    PruningSettingsComponent,
    FormInputSliderComponent,
    ButtonDirective,
    UpperCasePipe
  ],
  templateUrl: './pruning-menu-left.component.html',
  styleUrl: './pruning-menu-left.component.scss'
})
export class PruningMenuLeftComponent {

  @Input() formGroup: PruneSettingsFormGroup;
  @Input() gpus: { value: string, label: string }[] = [];
  @Input() locations: { value: string, label: string }[] = [];
  @Input() metrics: { value: string, label: string }[] = [];

  constructor(
    private router: Router,
    private readonly settingsService: SettingsService,
  ) {
  }

  public runBenchmark() {
    this.settingsService.Gpu = this.formGroup.controls.gpu.value
    this.settingsService.Location = this.formGroup.controls.location.value;
    this.settingsService.Threshold = this.formGroup.controls.threshold.value || 0;

    this.router.navigate(['/loading-benchmark']);
  }

}
