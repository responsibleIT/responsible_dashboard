import {Component, Input} from '@angular/core';
import {FormControl, ReactiveFormsModule} from "@angular/forms";
import {NgForOf, NgIf, TitleCasePipe} from '@angular/common';

@Component({
  selector: 'app-form-input-select',
  imports: [
    ReactiveFormsModule,
    NgIf,
    NgForOf,
    TitleCasePipe
  ],
  templateUrl: './form-input-select.component.html',
  styleUrl: './form-input-select.component.scss'
})
export class FormInputSelectComponent {

  public readonly id = `select-${Math.random().toString(36).substring(2, 15)}`;

  @Input() label: string;
  @Input() placeholder: string = 'Select an option';
  @Input() options: { value: string, label: string }[] = [];
  @Input() control: FormControl;
  @Input() required: boolean = false;

}
