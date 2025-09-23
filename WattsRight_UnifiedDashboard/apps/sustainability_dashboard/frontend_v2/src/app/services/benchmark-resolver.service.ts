import { Injectable } from '@angular/core';
import { Resolve, ActivatedRouteSnapshot } from '@angular/router';
import { Observable, of } from 'rxjs';
import { catchError, tap } from 'rxjs/operators';
import { BenchmarkService } from './benchmark.service';
import { BenchmarkData } from '@app/types/pruning.types';

@Injectable({ providedIn: 'root' })
export class BenchmarkResolver implements Resolve<BenchmarkData | null> {
  constructor(private benchmarkService: BenchmarkService) {}

  resolve(route: ActivatedRouteSnapshot): Observable<BenchmarkData | null> {
    // Prefer query param; fall back to service-stashed id
    const fromQuery = route.queryParamMap.get('upload_id');
    const uploadId = fromQuery || this.benchmarkService.uploadId;

    console.debug('[BenchmarkResolver] Resolving benchmark for upload_id', uploadId);

    if (!uploadId) return of(null); // never undefined

    // keep it in the service for later
    this.benchmarkService.setUploadId(uploadId);

    return this.benchmarkService.fetchData(uploadId).pipe(
      tap(d => this.benchmarkService.Data = d),
      catchError(err => {
        console.error('[BenchmarkResolver] fetch failed:', err);
        return of(null); // never throw here; let component decide
      })
    );
  }
}
