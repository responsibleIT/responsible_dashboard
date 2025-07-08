import { Injectable } from '@angular/core';
import {BehaviorSubject, catchError, map, Observable, throwError} from 'rxjs';
import {BenchmarkData} from '@app/types/pruning.types';
import {HttpClient} from '@angular/common/http';
import {environment} from '@env/environment';

@Injectable({
  providedIn: 'root'
})
export class BenchmarkService {

  private readonly apiUrl = `${environment.api.schema}://${environment.api.hostname}`;

  public data$ = new BehaviorSubject<BenchmarkData | null>(null);

  constructor(
    private http: HttpClient
  ) { }

  public fetchData(uploadId: string): Observable<BenchmarkData> {
    return this.http.get(`${this.apiUrl}/benchmark/${uploadId}`).pipe(
      map(response => response as BenchmarkData),
      catchError(error => throwError(() => new Error('Error fetching benchmark: ' + error.message)))
    )
  }

  public set Data(data: BenchmarkData | null) {
    this.data$.next(data);
  }

  public get Data(): BenchmarkData | null {
    return this.data$.getValue();
  }
}
