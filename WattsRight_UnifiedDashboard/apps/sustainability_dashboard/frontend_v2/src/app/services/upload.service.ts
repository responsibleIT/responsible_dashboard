import { Injectable } from '@angular/core';
import { BehaviorSubject, catchError, map, Observable, throwError } from 'rxjs';
import { HttpClient } from '@angular/common/http';
import { UploadResponse } from '@app/types/upload.types';
import { environment } from '@env/environment';

type ModelType = 'classification' | 'generative';

@Injectable({
  providedIn: 'root'
})
export class UploadService {
  private readonly apiUrl = `${environment.api.schema}://${environment.api.hostname}`;

  // Existing subjects
  public upload_id = new BehaviorSubject<string | null>(null);
  public huggingFaceUrl = new BehaviorSubject<string | null>(null);
  public h5ModelFilename = new BehaviorSubject<string | null>(null);

  // NEW: user selections from the upload modal
  public textColumn = new BehaviorSubject<string | null>(null);
  public targetColumn = new BehaviorSubject<string | null>(null);
  public modelType = new BehaviorSubject<'classification' | 'generative'>('classification');
  public selectedGPU = new BehaviorSubject<string | null>(null);
  public selectedLocation = new BehaviorSubject<string | null>(null);
  public selectedMetric = new BehaviorSubject<string | null>(null);

  constructor(private http: HttpClient) {
    this.upload_id = new BehaviorSubject<string | null>(this.loadUploadIdFromLocalStorage());
    // Restore model type from localStorage
    const storedType = localStorage.getItem('model_type') as 'classification' | 'generative' | null;
    if (storedType === 'classification' || storedType === 'generative') {
      this.modelType = new BehaviorSubject<'classification' | 'generative'>(storedType);
    }
  }

  // ----- persistence helpers -----
  private loadUploadIdFromLocalStorage(): string | null {
    const storedUploadId = localStorage.getItem('upload_id');
    if (storedUploadId) {
      // initialize subject and return
      this.upload_id.next(storedUploadId);
      return storedUploadId;
    }
    return null;
  }

  private saveUploadIdToLocalStorage(id: string | null): void {
    if (id) {
      localStorage.setItem('upload_id', id);
    } else {
      localStorage.removeItem('upload_id');
    }
  }

  // ----- API -----
  uploadData(formData: FormData): Observable<UploadResponse> {
    return this.postUpload(formData);
  }

  uploadClassificationData(formData: FormData): Observable<UploadResponse> {
    return this.postUpload(this.withModelType(formData, 'classification'));
  }

  uploadGenerativeData(formData: FormData): Observable<UploadResponse> {
    return this.postUpload(this.withModelType(formData, 'generative'));
  }

  private postUpload(formData: FormData): Observable<UploadResponse> {
    return this.http.post(`${this.apiUrl}/upload`, formData).pipe(
      map(response => response as UploadResponse),
      catchError(error => throwError(() => new Error('Upload failed: ' + error.message)))
    );
  }

  private withModelType(formData: FormData, modelType: ModelType): FormData {
    const payload = new FormData();

    formData.forEach((value, key) => {
      payload.append(key, value);
    });

    payload.set('model_type', modelType);
    return payload;
  }

  // ----- setters (keep PascalCase names so your component code compiles) -----
  set UploadId(upload_id: string | null) {
    this.upload_id.next(upload_id);
    this.saveUploadIdToLocalStorage(upload_id);
  }

  set HuggingFaceUrl(huggingFaceUrl: string | null) {
    this.huggingFaceUrl.next(huggingFaceUrl);
  }

  set H5ModelFilename(h5ModelFilename: string | null) {
    this.h5ModelFilename.next(h5ModelFilename);
  }

  // NEW setters used by your component:
  set TargetColumn(col: string | null) {
    this.targetColumn.next(col);
  }
  set SelectedGPU(gpu: string | null) {
    this.selectedGPU.next(gpu);
  }
  set SelectedLocation(loc: string | null) {
    this.selectedLocation.next(loc);
  }
  set SelectedMetric(metric: string | null) {
    this.selectedMetric.next(metric);
  }

  set ModelType(type: 'classification' | 'generative') {
    this.modelType.next(type);
    localStorage.setItem('model_type', type);
  }

  get modelTypeValue(): 'classification' | 'generative' {
    return this.modelType.value;
  }

  // ----- getters / convenience -----
  get modelName(): string | null {
    if (this.huggingFaceUrl.value) {
      const parts = this.huggingFaceUrl.value.split('/');
      return parts[parts.length - 1] || null;
    } else if (this.h5ModelFilename.value) {
      return this.h5ModelFilename.value;
    }
    return null;
  }

  get uploadIdValue(): string | null {
    return this.upload_id.value;
  }

  // NEW: current values (optional helpers)
  get targetColumnValue(): string | null { return this.targetColumn.value; }
  get selectedGPUValue(): string | null { return this.selectedGPU.value; }
  get selectedLocationValue(): string | null { return this.selectedLocation.value; }
  get selectedMetricValue(): string | null { return this.selectedMetric.value; }

  public clearUploadId(): void {
    this.upload_id.next(null);
    this.saveUploadIdToLocalStorage(null);
    localStorage.removeItem('model_type');
  }
}
