import { Component, OnDestroy, OnInit } from '@angular/core';
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
import { NgIf, NgFor, NgClass, UpperCasePipe, TitleCasePipe } from '@angular/common';
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
    NgFor,
    NgClass,
    ReactiveFormsModule,
    UpperCasePipe,
  ],
  templateUrl: './upload.modal.html',
  styleUrls: ['./upload.modal.scss']
})
export class UploadModal implements OnDestroy, OnInit {
  // Only the target dropdown is needed
  csvColumns: string[] = [];

  // Preset generative models (fetched from backend)
  presetModels: string[] = [];

  // Tab-based input method selection for generative flow
  activeInputMethod: 'preset' | 'huggingface' | 'upload' = 'huggingface';

  // HuggingFace validation feedback
  hfValidationMessage: string | null = null;
  hfValidationError = false;

  private readonly apiBase = `${environment.api.schema}://${environment.api.hostname}`;

  constructor(
    private readonly dialogRef: DialogRef<UploadModal>,
    private readonly formBuilder: FormBuilder,
    private readonly router: Router,
    private readonly uploadService: UploadService,
    private readonly websocketService: WebsocketService,
    private readonly http: HttpClient
  ) {}

  // form model (no textCol anymore)
  public uploadFormGroup = this.formBuilder.group({
    modelType:        this.formBuilder.control<'classification' | 'generative'>('classification'),
    presetModel:      this.formBuilder.control<string | null>(null),
    huggingfaceModel: this.formBuilder.control<string | null>(null),
    h5Model:          this.formBuilder.control<File | null>(null),
    dataset:          this.formBuilder.control<File | null>(null),
    targetCol:        this.formBuilder.control<string | null>(null),
  }, { validators: [this.modelSourceValidator()] });

  // Custom validator: preset OR (hf XOR h5)
  private modelSourceValidator() {
    return (group: AbstractControl): ValidationErrors | null => {
      const preset = group.get('presetModel')?.value;
      const hf = group.get('huggingfaceModel')?.value;
      const h5 = group.get('h5Model')?.value;
      const hasPreset = !!preset;
      const hasHf = !!hf;
      const hasH5 = h5 instanceof File;
      // At least one source, and not both hf+h5 at the same time
      const sources = [hasPreset, hasHf, hasH5].filter(Boolean).length;
      if (sources === 0) return { noModel: true };
      if (hasHf && hasH5) return { eitherOr: true };
      return null;
    };
  }

  // convenience getters for error messages
  get neitherFieldFilledError(): boolean {
    const g = this.uploadFormGroup;
    return g.hasError('noModel') && (g.touched || g.dirty);
  }
  get bothFieldsFilledError(): boolean {
    const g = this.uploadFormGroup;
    return g.hasError('eitherOr') && (g.touched || g.dirty);
  }

  // enable submit only when: model provided, dataset chosen, target column selected
  canSubmit(): boolean {
    const g = this.uploadFormGroup;
    const hasPreset = !!g.controls.presetModel.value;
    const hasModel = hasPreset || !!g.controls.huggingfaceModel.value || !!g.controls.h5Model.value;
    const isClassification = g.controls.modelType.value === 'classification';
    const hasDataset = !isClassification || g.controls.dataset.value instanceof File;
    const hasTarget = !!g.controls.targetCol.value;
    const targetRequirementMet = !isClassification || hasTarget;
    return hasModel && hasDataset && targetRequirementMet && !g.hasError('eitherOr') && !g.hasError('noModel');
  }

  async submitForm(): Promise<void> {
    const g = this.uploadFormGroup;
    if (!this.canSubmit()) {
      g.markAllAsTouched();
      return;
    }

    const modelType = g.controls.modelType.value;
    const formData = new FormData();
    const presetModel = g.controls.presetModel.value;
    const huggingfaceUrl = g.controls.huggingfaceModel.value;
    const h5ModelFile = g.controls.h5Model.value;
    const datasetFile = g.controls.dataset.value;

    if (presetModel) formData.append('preset_model', presetModel);
    if (huggingfaceUrl) formData.append('huggingface_url', huggingfaceUrl);
    if (h5ModelFile) formData.append('model', h5ModelFile);
    if (datasetFile) formData.append('dataset', datasetFile);

    const target = this.uploadFormGroup.controls.targetCol.value;
    if (modelType === 'classification' && target) {
      formData.append('selected_columns', JSON.stringify({ target_column: target }));
    }

    try {
      const uploadRequest = modelType === 'generative'
        ? this.uploadService.uploadGenerativeData(formData)
        : this.uploadService.uploadClassificationData(formData);

      const upload_id = await firstValueFrom(
        uploadRequest.pipe(map(d => d.upload_id))
      );

      this.websocketService.UploadId = upload_id;
      this.uploadService.UploadId = upload_id;
      this.uploadService.ModelType = modelType as 'classification' | 'generative';
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
      const raw = (reader.result as string) ?? '';
      if (!raw) { this.csvColumns = []; return; }

      // Pick the first non-empty line (skip BOM + blank lines)
      const lines = raw.split(/\r?\n/).map(l => l.replace(/^\uFEFF/, ''));
      const headerLine = (lines.find(l => l.trim().length > 0) || '').trim();
      if (!headerLine) { this.csvColumns = []; return; }

      // Robust delimiter detection ignoring quoted delimiters
      const candidates = [',', ';', '\t'];
      const countDelimiter = (delim: string): number => {
        let inQuotes = false, count = 0;
        for (let i = 0; i < headerLine.length; i++) {
          const ch = headerLine[i];
          if (ch === '"') inQuotes = !inQuotes;
          else if (!inQuotes && ch === delim) count++;
        }
        return count;
      };
      const best = candidates
        .map(d => ({ d, c: countDelimiter(d) }))
        .sort((a, b) => b.c - a.c)[0]?.d || ',';

      // Split respecting simple quoted fields (no embedded escaped quotes handling needed for headers)
      const cols: string[] = [];
      let buf = '', inQuotes = false;
      for (let i = 0; i < headerLine.length; i++) {
        const ch = headerLine[i];
        if (ch === '"') {
          inQuotes = !inQuotes; // toggle
        } else if (ch === best && !inQuotes) {
          cols.push(buf.trim().replace(/^"(.*)"$/, '$1'));
          buf = '';
        } else {
          buf += ch;
        }
      }
      if (buf.length) cols.push(buf.trim().replace(/^"(.*)"$/, '$1'));

      const filtered = cols.filter(c => c.length > 0);
      this.csvColumns = filtered;

      const current = this.uploadFormGroup.controls.targetCol.value;
      if (current) {
        const match = filtered.find(c => c.toLowerCase() === current.toLowerCase());
        if (match) {
          // Normalize exact casing
          this.uploadFormGroup.controls.targetCol.setValue(match, { emitEvent: false });
        } else {
          this.uploadFormGroup.controls.targetCol.setValue(null);
        }
      }
    };
    reader.readAsText(file);
  }

  ngOnInit(): void {
    this.uploadFormGroup.controls.dataset.valueChanges.subscribe(file => {
      if (file instanceof File) this.readCsvHeader(file);
      else this.csvColumns = [];
    });

    this.uploadFormGroup.controls.modelType.valueChanges.subscribe(modelType => {
      if (modelType === 'generative') {
        this.uploadFormGroup.controls.targetCol.setValue(null, { emitEvent: false });
        this.activeInputMethod = 'huggingface';
        // Fetch preset models when switching to generative
        this.fetchPresetModels();
      } else {
        this.presetModels = [];
        this.activeInputMethod = 'huggingface';
        this.uploadFormGroup.controls.presetModel.setValue(null, { emitEvent: false });
      }
    });

    // When a preset is selected, clear HF URL and H5 model
    this.uploadFormGroup.controls.presetModel.valueChanges.subscribe(preset => {
      if (preset) {
        this.uploadFormGroup.controls.huggingfaceModel.setValue(null, { emitEvent: false });
        this.uploadFormGroup.controls.h5Model.setValue(null, { emitEvent: false });
      }
    });

    // When HF URL is entered, clear preset
    this.uploadFormGroup.controls.huggingfaceModel.valueChanges.subscribe(url => {
      if (url) {
        this.uploadFormGroup.controls.presetModel.setValue(null, { emitEvent: false });
        // Simple HF model name validation
        this.validateHuggingFaceInput(url);
      } else {
        this.hfValidationMessage = null;
        this.hfValidationError = false;
      }
    });
  }

  private validateHuggingFaceInput(name: string): void {
    const trimmed = name.trim();
    if (!trimmed) {
      this.hfValidationMessage = null;
      this.hfValidationError = false;
      return;
    }
    // Basic format: should be org/model or just model name
    const validPattern = /^[a-zA-Z0-9_-]+\/[a-zA-Z0-9._-]+$|^[a-zA-Z0-9_-]+$/;
    if (!validPattern.test(trimmed)) {
      this.hfValidationMessage = 'Model name format should be "organization/model" (e.g. facebook/opt-125m)';
      this.hfValidationError = true;
    } else {
      this.hfValidationMessage = null;
      this.hfValidationError = false;
    }
  }

  private fetchPresetModels(): void {
    this.http.get<{ models: string[] }>(`${this.apiBase}/api/preset-models`).subscribe({
      next: (resp) => { this.presetModels = resp.models || []; },
      error: () => { this.presetModels = []; },
    });
  }

  ngOnDestroy(): void {
    this.websocketService.disconnect();
  }
}

type UploadFormControls = {
  presetModel: FormControl<string | null>;
  huggingfaceModel: FormControl<string | null>;
  h5Model: FormControl<File | null>;
  dataset: FormControl<File | null>;
  targetCol: FormControl<string | null>;
};
