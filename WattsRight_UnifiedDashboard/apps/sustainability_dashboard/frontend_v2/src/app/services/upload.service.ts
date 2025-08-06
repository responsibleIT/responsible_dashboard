import { Injectable } from '@angular/core';
import {BehaviorSubject, catchError, map, Observable, throwError} from 'rxjs';
import {HttpClient} from '@angular/common/http';
import {UploadResponse} from '@app/types/upload.types';
import {environment} from '@env/environment';

@Injectable({
  providedIn: 'root'
})
export class UploadService {

  private readonly apiUrl = `${environment.api.schema}://${environment.api.hostname}`;

  public uploadId = new BehaviorSubject<string | null>(null);
  public huggingFaceUrl = new BehaviorSubject<string | null>(null);
  public h5ModelFilename = new BehaviorSubject<string | null>(null);

  constructor(
    private http: HttpClient
  ) {
    this.uploadId = new BehaviorSubject<string | null>(this.loadUploadIdFromLocalStorage());
  }

  private loadUploadIdFromLocalStorage(): string | null {
    const storedUploadId = localStorage.getItem('uploadId');
    if (storedUploadId) {
      this.uploadId.next(storedUploadId);
      return storedUploadId;
    }
    return null;
  }

  uploadData(formData: FormData): Observable<UploadResponse> {
    return this.http.post(`${this.apiUrl}/upload`, formData).pipe(
      map(response => response as UploadResponse),
      catchError(error => throwError(() => new Error('Upload failed: ' + error.message)))
    )
  }

  set UploadId(uploadId: string | null) {
    this.uploadId.next(uploadId);
  }

  set HuggingFaceUrl(huggingFaceUrl: string | null) {
    this.huggingFaceUrl.next(huggingFaceUrl);
  }

  set H5ModelFilename(h5ModelFilename: string | null) {
    this.h5ModelFilename.next(h5ModelFilename);
  }

  get modelName(): string | null {
    if (this.huggingFaceUrl.value) {
      const urlParts = this.huggingFaceUrl.value.split('/', 2);
      return urlParts[urlParts.length - 1];
    } else if (this.h5ModelFilename.value) {
      return this.h5ModelFilename.value;
    }

    return null;
  }

  get uploadIdValue(): string | null {
    return this.uploadId.value;
  }

  public clearUploadId(): void {
    this.uploadId.next(null);
    localStorage.removeItem('uploadId');
  }
}
