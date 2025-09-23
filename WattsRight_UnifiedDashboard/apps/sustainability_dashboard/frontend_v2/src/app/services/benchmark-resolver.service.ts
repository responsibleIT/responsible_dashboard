// apps/sustainability_dashboard/frontend_v2/src/app/services/benchmark-resolver.service.ts
import { Injectable } from '@angular/core';
import { Resolve, ActivatedRouteSnapshot, RouterStateSnapshot } from '@angular/router';
import { Observable, of } from 'rxjs';
import { BenchmarkService } from './benchmark.service';
import { BenchmarkData } from '@app/types/pruning.types';

@Injectable({ providedIn: 'root' })
export class BenchmarkResolver implements Resolve<BenchmarkData | null> {
  constructor(private benchmarkService: BenchmarkService) {}

  resolve(route: ActivatedRouteSnapshot, state: RouterStateSnapshot): Observable<BenchmarkData | null> {
    // 1. Prefer query param
    const uploadIdFromRoute = route.queryParamMap.get('upload_id');

    // 2. Fall back to service value
    const uploadId = uploadIdFromRoute || this.benchmarkService.uploadId;

    if (!uploadId) {
      console.error('[BenchmarkResolver] No upload_id available');
      return of(null);
    }

    console.log('[BenchmarkResolver] Resolving benchmark for upload_id', uploadId);
    return this.benchmarkService.fetchData(uploadId);
  }
}
