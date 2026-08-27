import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, of, catchError, tap, map } from 'rxjs';
import { environment } from '@env/environment';
import { GenerativeDashboardData } from '@app/types/generative.types';

@Injectable({ providedIn: 'root' })
export class GenerativeDataService {
  private readonly apiUrl = `${environment.api.schema}://${environment.api.hostname}`;
  private data$ = new BehaviorSubject<GenerativeDashboardData | null>(null);

  constructor(private http: HttpClient) {}

  get dashboardData$(): Observable<GenerativeDashboardData | null> {
    return this.data$.asObservable();
  }

  get currentData(): GenerativeDashboardData | null {
    return this.data$.getValue();
  }

  /**
   * Fetch generative dashboard data from the backend.
   * Falls back to null on error (caller can use GenerativeMockDataService instead).
   */
  fetchData(uploadId: string): Observable<GenerativeDashboardData | null> {
    return this.http.get<GenerativeDashboardData>(`${this.apiUrl}/generative/${uploadId}`).pipe(
      tap((data) => this.data$.next(data)),
      catchError((err) => {
        console.error('[GenerativeDataService] fetch failed:', err);
        return of(null);
      }),
    );
  }

  clear(): void {
    this.data$.next(null);
  }
}
