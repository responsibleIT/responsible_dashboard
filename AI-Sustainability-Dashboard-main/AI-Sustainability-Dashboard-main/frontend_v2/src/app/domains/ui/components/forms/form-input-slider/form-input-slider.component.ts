import {Component, Input} from '@angular/core';
import {FormControl, ReactiveFormsModule} from "@angular/forms";

@Component({
  selector: 'app-form-input-slider',
    imports: [
        ReactiveFormsModule
    ],
  templateUrl: './form-input-slider.component.html',
  styleUrl: './form-input-slider.component.scss'
})
export class FormInputSliderComponent {

  @Input() label: string;
  @Input() control: FormControl;
  @Input() min: number;
  @Input() max: number;
  @Input() step: number;

}
