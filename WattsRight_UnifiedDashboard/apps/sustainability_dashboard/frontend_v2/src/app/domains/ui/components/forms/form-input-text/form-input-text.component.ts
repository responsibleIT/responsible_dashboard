import {Component, Input} from '@angular/core';
import {FormControl, ReactiveFormsModule} from '@angular/forms';
import {NgIf} from '@angular/common';

@Component({
  selector: 'app-form-input-text',
  imports: [
    ReactiveFormsModule,
    NgIf
  ],
  templateUrl: './form-input-text.component.html',
  styleUrl: './form-input-text.component.scss'
})
export class FormInputTextComponent {

  @Input() type: 'text' | 'email' | 'password' = 'text';
  @Input() label: string | null = null;
  @Input() placeholder: string;
  @Input() control: FormControl;
  @Input() required: boolean = false;

}
