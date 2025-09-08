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
  // ------- backing lists for selects -------
  csvColumns: string[] = [];
  gpus: string[] = [];
  locations: string[] = [];
  metrics: string[] = [];

  private readonly apiBase = `${environment.api.schema}://${environment.api.hostname}`;

  constructor(
    private readonly dialogRef: DialogRef<UploadModal>,
    private readonly formBuilder: FormBuilder,
    private readonly router: Router,
    private readonly uploadService: UploadService,
    private readonly websocketService: WebsocketService,
    private readonly http: HttpClient
  ) {
    this.loadSettings();
    // watch dataset file to extract CSV header
    this.uploadFormGroup.controls.dataset.valueChanges.subscribe(f => {
      if (f instanceof File) this.readCsvHeader(f);
    });
  }

  // ------- either/or validator for HF URL vs H5 model -------
  private eitherOrValidator(controlNames: (keyof UploadFormControls)[]) {
    return (group: AbstractControl): ValidationErrors | null => {
      const values = controlNames.map(n => (group.get(n as string)?.value));
      const filled = values.filter(v =>
        v !== null && v !== undefined && v !== '' && !(v instanceof File && !v)
      ).length;
      return (filled === 1) ? null : { eitherOr: true };
    };
  }

  // ------- form model -------
  uploadFormGroup = this.formBuilder.group<UploadForm>({
    huggingfaceModel: new FormControl<string | null>(null),
    h5Model: new FormControl<File | null>(null),
    dataset: new FormControl<File | null>(null, { validators: [Validators.required] }),

    // NEW selects
    textCol: new FormControl<string | null>(null),
    targetCol: new FormControl<string | null>(null),
    gpu: new FormControl<string | null>(null),
    location: new FormControl<string | null>(null),
    metric: new FormControl<string | null>(null),
  }, {
    validators: [this.eitherOrValidator(['huggingfaceModel', 'h5Model'])]
  });

  // convenience getters for template/types
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

  // button enablement (also used by template)
  canSubmit(): boolean {
    const g = this.uploadFormGroup;
    if (g.invalid) return false;
    // need HF URL or H5
    const hasModel = !!g.controls.huggingfaceModel.value || !!g.controls.h5Model.value;
    // need dataset
    const hasDataset = !!g.controls.dataset.value;
    // need text + target columns
    const hasCols = !!g.controls.textCol.value && !!g.controls.targetCol.value;
    return hasModel && hasDataset && hasCols;
  }

  // ------- submit -------
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

    // store selections for later websocket flow
    this.uploadService.TextColumn       = g.controls.textCol.value ?? null;
    this.uploadService.TargetColumn     = g.controls.targetCol.value ?? null;
    this.uploadService.SelectedGPU      = g.controls.gpu.value ?? null;
    this.uploadService.SelectedLocation = g.controls.location.value ?? null;
    this.uploadService.SelectedMetric   = g.controls.metric.value ?? null;

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

  // ------- helpers -------
  private async loadSettings(): Promise<void> {
    try {
      const s = await firstValueFrom(this.http.get<{gpus:string[];locations:string[];metrics:string[]}>(`${this.apiBase}/settings`));
      this.gpus = s.gpus ?? [];
      this.locations = s.locations ?? [];
      this.metrics = s.metrics ?? [];
    } catch (e) {
      console.warn('Failed to load settings', e);
      this.gpus = [];
      this.locations = [];
      this.metrics = [];
    }
  }

  private readCsvHeader(file: File): void {
    const reader = new FileReader();
    reader.onload = () => {
      const text = (reader.result as string) || '';
      const firstLine = text.split(/\r?\n/)[0] ?? '';
      // very basic CSV split; your project may have a CSV parser you can use instead
      const cols = firstLine
        .split(',')
        .map(c => c.trim().replace(/^"(.*)"$/,'$1'))
        .filter(c => c.length > 0);
      this.csvColumns = cols;
      // reset the selects if they’re no longer valid
      const g = this.uploadFormGroup;
      if (!cols.includes(g.controls.textCol.value ?? '')) g.controls.textCol.setValue(null);
      if (!cols.includes(g.controls.targetCol.value ?? '')) g.controls.targetCol.setValue(null);
    };
    reader.readAsText(file);
  }

  ngOnDestroy(): void {
    this.websocketService.disconnect();
  }
}

/** Strongly-typed form model */
type UploadFormControls = {
  huggingfaceModel: FormControl<string | null>;
  h5Model: FormControl<File | null>;
  dataset: FormControl<File | null>;
  textCol: FormControl<string | null>;
  targetCol: FormControl<string | null>;
  gpu: FormControl<string | null>;
  location: FormControl<string | null>;
  metric: FormControl<string | null>;
};
type UploadForm = {
  [K in keyof UploadFormControls]: UploadFormControls[K];
};
