import { Component, OnDestroy } from '@angular/core';
import { ButtonDirective } from '@app/domains/ui/directives/button/button.directive';
import { FormInputFileComponent } from '@app/domains/ui/components/forms/form-input-file/form-input-file.component';
import { FormInputTextComponent } from '@app/domains/ui/components/forms/form-input-text/form-input-text.component';
import {
  AbstractControl,
  FormBuilder,
  FormControl,
  FormsModule,
  ReactiveFormsModule,
  ValidationErrors,
  Validators
} from '@angular/forms';
import { NgIf, UpperCasePipe, TitleCasePipe } from '@angular/common';
import { Router } from '@angular/router';
import { UploadService } from '@app/services/upload.service';
import { WebsocketService } from '@app/services/websocket.service';
import { firstValueFrom, map, throwError } from 'rxjs';
import { DialogRef } from '@angular/cdk/dialog';
import { HttpClient } from '@angular/common/http';
import { environment } from '@env/environment';

@Component({
  selector: 'app-upload',
  standalone: true,
  imports: [
    ButtonDirective,
    FormInputFileComponent,
    FormInputTextComponent,
    FormsModule,
    NgIf,
    ReactiveFormsModule,
    UpperCasePipe,
    TitleCasePipe
  ],
  templateUrl: './upload.modal.html',
  styleUrl: './upload.modal.scss'
})
export class UploadModal implements OnDestroy {
  // Only the target dropdown is needed
  csvColumns: string[] = [];

  private readonly apiBase = `${environment.api.schema}://${environment.api.hostname}`;

  constructor(
    private readonly dialogRef: DialogRef<UploadModal>,
    private readonly formBuilder: FormBuilder,
    private readonly router: Router,
    private readonly uploadService: UploadService,
    private readonly websocketService: WebsocketService,
    private readonly http: HttpClient
  ) {
    // When dataset changes, read header and populate dropdown
    this.uploadFormGroup.controls.dataset.valueChanges.subscribe(f => {
      if (f instanceof File) this.readCsvHeader(f);
    });
  }

  // either/or validator for HF URL vs H5 model
  private eitherOrValidator(controlNames: (keyof UploadFormControls)[]) {
    return (group: AbstractControl): ValidationErrors | null => {
      const values = controlNames.map(n => (group.get(n as string)?.value));
      const filled = values.filter(v =>
        v !== null && v !== undefined && v !== '' && !(v instanceof File && !v)
      ).length;
      return (filled === 1) ? null : { eitherOr: true };
    };
  }

  // form model (no textCol anymore)
  public uploadFormGroup = this.formBuilder.group({
    huggingfaceModel: this.formBuilder.control<string | null>(null),
    h5Model:          this.formBuilder.control<File | null>(null),
    dataset:          this.formBuilder.control<File | null>(null, [Validators.required]),
    targetCol:        this.formBuilder.control<string | null>(null),
  }, { validators: [this.eitherOrValidator(['huggingfaceModel', 'h5Model'])] });

  // convenience getters for error messages
  get neitherFieldFilledError(): boolean {
    const g = this.uploadFormGroup;
    const err = g.hasError('eitherOr') && (g.touched || g.dirty);
    return err && !g.controls.huggingfaceModel.value && !g.controls.h5Model.value;
  }
  get bothFieldsFilledError(): boolean {
    const g = this.uploadFormGroup;
    const err = g.hasError('eitherOr') && (g.touched || g.dirty);
    return err && !!g.controls.huggingfaceModel.value && !!g.controls.h5Model.value;
  }

  // enable submit only when: model provided, dataset chosen, target column selected
  canSubmit(): boolean {
    const g = this.uploadFormGroup;
    const hasModel = !!g.controls.huggingfaceModel.value || !!g.controls.h5Model.value;
    const hasDataset = g.controls.dataset.value instanceof File;
    const hasTarget = !!g.controls.targetCol.value;
    return hasModel && hasDataset && hasTarget && !g.hasError('eitherOr');
  }

  async submitForm(): Promise<void> {
    const g = this.uploadFormGroup;
    if (!this.canSubmit()) {
      g.markAllAsTouched();
      return;
    }

    const formData = new FormData();
    const huggingfaceUrl = g.controls.huggingfaceModel.value;
    const h5ModelFile = g.controls.h5Model.value;
    const datasetFile = g.controls.dataset.value;

    if (huggingfaceUrl) formData.append('huggingface_url', huggingfaceUrl);
    if (h5ModelFile) formData.append('model', h5ModelFile);
    if (datasetFile) formData.append('dataset', datasetFile);

    // 🔥 Add this block
   const target = this.uploadFormGroup.controls.targetCol.value;
    if (target) {
      formData.append('selected_columns', JSON.stringify({ target_column: target }));
    }
    // 🔥 End block

    try {
      const uploadId = await firstValueFrom(
        this.uploadService.uploadData(formData).pipe(map(d => d.upload_id))
      );

      this.websocketService.UploadId = uploadId;
      this.uploadService.UploadId = uploadId;
      this.uploadService.HuggingFaceUrl = huggingfaceUrl ?? null;
      this.uploadService.H5ModelFilename = h5ModelFile?.name ?? null;

      this.dialogRef.close();
      this.router.navigate(['/loading-upload']);
    } catch (err) {
      console.error('Upload failed:', err);
      throwError(() => new Error('Upload failed'));
    }
  }

  // read the first line of CSV to populate column names
  private readCsvHeader(file: File): void {
    const reader = new FileReader();
    reader.onload = () => {
      const text = (reader.result as string) ?? '';
      const firstLine = text.split(/\r?\n/)[0] ?? '';

      // strip BOM if present
      const line = firstLine.replace(/^\uFEFF/, '');

      // detect delimiter (comma, semicolon, or tab)
      const delimiters = [',', ';', '\t'];
      const best = delimiters
        .map(d => ({ d, parts: line.split(d).length }))
        .sort((a, b) => b.parts - a.parts)[0]?.d ?? ',';

      const cols = line
        .split(best)
        .map(c => c.trim().replace(/^"(.*)"$/, '$1'))
        .filter(Boolean);

      this.csvColumns = cols;

      // if current target not valid, reset
      const tgt = this.uploadFormGroup.controls.targetCol.value;
      if (!cols.includes(tgt ?? '')) {
        this.uploadFormGroup.controls.targetCol.setValue(null);
      }
    };
    reader.readAsText(file);
  }

  ngOnInit(): void {
    this.uploadFormGroup.controls.dataset.valueChanges.subscribe(file => {
      if (file instanceof File) this.readCsvHeader(file);
    });
  }

  ngOnDestroy(): void {
    this.websocketService.disconnect();
  }
}

type UploadFormControls = {
  huggingfaceModel: FormControl<string | null>;
  h5Model: FormControl<File | null>;
  dataset: FormControl<File | null>;
  targetCol: FormControl<string | null>;
};
