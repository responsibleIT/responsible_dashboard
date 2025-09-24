import { Injectable } from '@angular/core';
import { BehaviorSubject, catchError, map, Observable, throwError } from 'rxjs';
import { BenchmarkData } from '@app/types/pruning.types';
import { HttpClient } from '@angular/common/http';
import { environment } from '@env/environment';

@Injectable({ providedIn: 'root' })
export class BenchmarkService {
  private readonly apiUrl = `${environment.api.schema}://${environment.api.hostname}`;
  private currentUploadId: string | null = null;

  public data$ = new BehaviorSubject<BenchmarkData | null>(null);

  constructor(private http: HttpClient) {}

  fetchData(upload_id: string): Observable<BenchmarkData> {
    this.currentUploadId = upload_id;
    return this.http.get<BenchmarkData>(`${this.apiUrl}/benchmark/${upload_id}`);
  }

  public exportModel(uploadId: string): void {
    const url = `${this.apiUrl}/api/export/${uploadId}`;
    window.open(url, '_blank'); // triggers browser download
  }

  setUploadId(id: string) { this.currentUploadId = id; }   // 👈 add this

  set Data(data: BenchmarkData | null) { this.data$.next(data); }
  get Data(): BenchmarkData | null     { return this.data$.getValue(); }
  get uploadId(): string | null        { return this.currentUploadId; }
}