import {Component, OnDestroy} from '@angular/core';
import {ButtonDirective} from '@app/domains/ui/directives/button/button.directive';
import {FormInputFileComponent} from '@app/domains/ui/components/forms/form-input-file/form-input-file.component';
import {FormInputTextComponent} from '@app/domains/ui/components/forms/form-input-text/form-input-text.component';
import {
  AbstractControl,
  FormBuilder,
  FormsModule,
  ReactiveFormsModule,
  ValidationErrors,
  Validators
} from '@angular/forms';
import {NgIf, UpperCasePipe} from '@angular/common';
import {Router} from '@angular/router';
import {UploadService} from '@app/services/upload.service';
import {WebsocketService} from '@app/services/websocket.service';
import {firstValueFrom, map, throwError} from 'rxjs';
import {DialogRef} from '@angular/cdk/dialog';

@Component({
  selector: 'app-upload',
  imports: [
    ButtonDirective,
    FormInputFileComponent,
    FormInputTextComponent,
    FormsModule,
    NgIf,
    ReactiveFormsModule,
    UpperCasePipe
  ],
  templateUrl: './upload.modal.html',
  styleUrl: './upload.modal.scss'
})
export class UploadModal implements OnDestroy {

  private eitherOrValidator(controlNames: string[]): (group: AbstractControl) => ValidationErrors | null {
    return (group: AbstractControl): ValidationErrors | null => {
      const controls = controlNames.map(name => group.get(name));

      // Check if controls are valid
      if (controls.some(control => !control)) {
        return null; // Invalid configuration, should not happen
      }

      // Count how many fields are filled
      const filledCount = controls.filter(control =>
        control?.value !== null &&
        control?.value !== undefined &&
        control?.value !== ''
      ).length;

      // Either none are filled or more than one is filled
      if (filledCount === 0 || filledCount > 1) {
        return { eitherOr: true };
      }

      return null; // Valid - exactly one is filled
    };
  }

  public uploadFormGroup = this.formBuilder.group({
    huggingfaceModel: this.formBuilder.control<string | null>(null),
    h5Model: this.formBuilder.control<File | null>(null),
    dataset: this.formBuilder.control<string | null>(null, [Validators.required])
  }, {
    validators: [this.eitherOrValidator(['huggingfaceUrl', 'h5Model'])]
  });

  constructor(
    private readonly dialogRef: DialogRef<UploadModal>,
    private readonly formBuilder: FormBuilder,
    private readonly router: Router,
    private readonly uploadService: UploadService,
    private readonly websocketService: WebsocketService,
  ) {
  }

  public submitForm(): void {
    if (!this.formIsValid) {
      this.uploadFormGroup.markAllAsTouched();
      return;
    }

    const formData = new FormData();

    const huggingfaceUrl = this.uploadFormGroup.controls.huggingfaceModel.value;
    if (huggingfaceUrl) {
      formData.append('huggingface_url', huggingfaceUrl);
    }

    const h5ModelFile = this.uploadFormGroup.controls.h5Model.value;
    if (h5ModelFile) {
      formData.append('model', h5ModelFile);
    }

    const datasetFile = this.uploadFormGroup.controls.dataset?.value;
    if (datasetFile) {
      formData.append('dataset', datasetFile);
    }

    firstValueFrom(this.uploadService.uploadData(formData).pipe(
      map(data => data.upload_id),
    )).then(uploadId => {
      this.websocketService.UploadId = uploadId;
      
      this.uploadService.UploadId = uploadId;
      this.uploadService.HuggingFaceUrl = this.uploadFormGroup.controls.huggingfaceModel.value
      this.uploadService.H5ModelFilename = this.uploadFormGroup.controls.h5Model.value?.name || null;

      this.dialogRef.close()
      this.router.navigate(["/loading-upload"]);
    }).catch(error => {
      console.error('Upload failed:', error);
      return throwError(() => new Error('Upload failed'));
    })
  }

  get formIsValid(): boolean {
    return this.uploadFormGroup.valid;
  }

  get eitherOrError(): boolean {
    return this.uploadFormGroup.hasError('eitherOr') &&
      (this.uploadFormGroup.touched || this.uploadFormGroup.dirty);
  }

  get neitherFieldFilledError(): boolean {
    return this.eitherOrError &&
      !this.uploadFormGroup.controls.huggingfaceModel.value &&
      !this.uploadFormGroup.controls.h5Model!.value;
  }

  get bothFieldsFilledError(): boolean {
    return this.eitherOrError &&
      this.uploadFormGroup.controls.huggingfaceModel!.value !== null &&
      this.uploadFormGroup.controls.h5Model!.value !== null;
  }

  ngOnDestroy(): void {
    this.websocketService.disconnect();
  }

}
