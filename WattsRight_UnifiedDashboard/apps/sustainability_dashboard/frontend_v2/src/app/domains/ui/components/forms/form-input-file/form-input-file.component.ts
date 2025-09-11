// apps/sustainability_dashboard/frontend_v2/src/app/domains/ui/components/forms/form-input-file/form-input-file.component.ts
import { Component, Input } from '@angular/core';
import { FormControl, FormsModule, ReactiveFormsModule } from '@angular/forms';
import { NgIf } from '@angular/common';

@Component({
  selector: 'app-form-input-file',
  standalone: true,
  imports: [FormsModule, ReactiveFormsModule, NgIf],
  templateUrl: './form-input-file.component.html',
  styleUrls: ['./form-input-file.component.scss'],
})
export class FormInputFileComponent {
  public readonly id = `file-upload-${Math.random().toString(36).slice(2)}`;

  @Input() label: string | null = null;
  @Input() placeholder = 'Select a file';
  @Input({ required: true }) control!: FormControl<File | null>;
  @Input() required = false;
  @Input() accept?: string; // optional accept filter e.g. ".csv,.txt"

  filename = '';

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0] ?? null;

    this.control.setValue(file);
    this.filename = file?.name ?? '';
  }

  triggerFileInput(): void {
    const el = document.getElementById(this.id) as HTMLInputElement | null;
    el?.click();
  }

  clear(): void {
    const el = document.getElementById(this.id) as HTMLInputElement | null;
    if (el) el.value = '';
    this.control.setValue(null);
    this.filename = '';
  }
}
