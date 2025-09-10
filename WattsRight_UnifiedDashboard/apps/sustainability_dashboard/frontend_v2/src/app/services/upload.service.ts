import { Injectable } from '@angular/core';
import { BehaviorSubject, catchError, map, Observable, throwError } from 'rxjs';
import { HttpClient } from '@angular/common/http';
import { UploadResponse } from '@app/types/upload.types';
import { environment } from '@env/environment';

@Injectable({
  providedIn: 'root'
})
export class UploadService {
  private readonly apiUrl = `${environment.api.schema}://${environment.api.hostname}`;

  // Existing subjects
  public uploadId = new BehaviorSubject<string | null>(null);
  public huggingFaceUrl = new BehaviorSubject<string | null>(null);
  public h5ModelFilename = new BehaviorSubject<string | null>(null);

  // NEW: user selections from the upload modal
  public textColumn = new BehaviorSubject<string | null>(null);
  public targetColumn = new BehaviorSubject<string | null>(null);
  public selectedGPU = new BehaviorSubject<string | null>(null);
  public selectedLocation = new BehaviorSubject<string | null>(null);
  public selectedMetric = new BehaviorSubject<string | null>(null);

  constructor(private http: HttpClient) {
    this.uploadId = new BehaviorSubject<string | null>(this.loadUploadIdFromLocalStorage());
  }

  // ----- persistence helpers -----
  private loadUploadIdFromLocalStorage(): string | null {
    const storedUploadId = localStorage.getItem('uploadId');
    if (storedUploadId) {
      // initialize subject and return
      this.uploadId.next(storedUploadId);
      return storedUploadId;
    }
    return null;
  }

  private saveUploadIdToLocalStorage(id: string | null): void {
    if (id) {
      localStorage.setItem('uploadId', id);
    } else {
      localStorage.removeItem('uploadId');
    }
  }

  // ----- API -----
  uploadData(formData: FormData): Observable<UploadResponse> {
    return this.http.post(`${this.apiUrl}/upload`, formData).pipe(
      map(response => response as UploadResponse),
      catchError(error => throwError(() => new Error('Upload failed: ' + error.message)))
    );
  }

  // ----- setters (keep PascalCase names so your component code compiles) -----
  set UploadId(uploadId: string | null) {
    this.uploadId.next(uploadId);
    this.saveUploadIdToLocalStorage(uploadId);
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
    return this.uploadId.value;
  }

  // NEW: current values (optional helpers)
  get targetColumnValue(): string | null { return this.targetColumn.value; }
  get selectedGPUValue(): string | null { return this.selectedGPU.value; }
  get selectedLocationValue(): string | null { return this.selectedLocation.value; }
  get selectedMetricValue(): string | null { return this.selectedMetric.value; }

  public clearUploadId(): void {
    this.uploadId.next(null);
    this.saveUploadIdToLocalStorage(null);
  }
}
