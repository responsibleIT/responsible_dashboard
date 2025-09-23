import { Injectable } from '@angular/core';
import { BehaviorSubject, catchError, map, Observable, throwError } from 'rxjs';
import { BenchmarkData } from '@app/types/pruning.types';
import { HttpClient } from '@angular/common/http';
import { environment } from '@env/environment';

@Injectable({
  providedIn: 'root'
})
export class BenchmarkService {
  private readonly apiUrl = `${environment.api.schema}://${environment.api.hostname}`;
  private currentUploadId: string | null = null;

  public data$ = new BehaviorSubject<BenchmarkData | null>(null);

  constructor(private http: HttpClient) {}

  /** explicitly set the uploadId, so loader components can stash it */
  public setUploadId(id: string | null): void {
    this.currentUploadId = id;
  }

  public fetchData(upload_id: string): Observable<BenchmarkData> {
    // still keep the auto-stash behavior for safety
    this.currentUploadId = upload_id;

    return this.http.get<BenchmarkData>(`${this.apiUrl}/benchmark/${upload_id}`).pipe(
      map(response => response as BenchmarkData),
      catchError(error => throwError(() => new Error('Error fetching benchmark: ' + error.message)))
    );
  }

  public set Data(data: BenchmarkData | null) {
    this.data$.next(data);
  }

  public get Data(): BenchmarkData | null {
    return this.data$.getValue();
  }

  public get uploadId(): string | null {
    return this.currentUploadId;
  }
}