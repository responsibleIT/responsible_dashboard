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
  ],
  templateUrl: './upload.modal.html',
  styleUrls: ['./upload.modal.scss']
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
  ) {}

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
      const upload_id = await firstValueFrom(
        this.uploadService.uploadData(formData).pipe(map(d => d.upload_id))
      );

      this.websocketService.UploadId = upload_id;
      this.uploadService.UploadId = upload_id;
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
