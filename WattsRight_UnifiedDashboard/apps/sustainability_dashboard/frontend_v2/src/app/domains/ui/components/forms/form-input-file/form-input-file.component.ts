import {Component, Input} from '@angular/core';
import {FormControl, FormsModule, ReactiveFormsModule} from "@angular/forms";
import {NgIf} from '@angular/common';

@Component({
  selector: 'app-form-input-file',
  imports: [
    FormsModule,
    ReactiveFormsModule,
    NgIf
  ],
  templateUrl: './form-input-file.component.html',
  styleUrl: './form-input-file.component.scss'
})
export class FormInputFileComponent {

  public readonly id = `file-upload-${Math.random().toString(36).substring(2, 15)}`;

  @Input() label: string | null = null;
  @Input() placeholder: string = 'Select a file';
  @Input() control!: FormControl;
  @Input() required: boolean = false;

  fileName: string = '';

  onFileSelected(event: any): void {
    const file = event.target.files[0];
    if (file) {
      this.fileName = file.name;
      this.control.setValue(file);
    }
  }

  triggerFileInput(): void {
    document.getElementById(this.id)?.click();
  }

}
