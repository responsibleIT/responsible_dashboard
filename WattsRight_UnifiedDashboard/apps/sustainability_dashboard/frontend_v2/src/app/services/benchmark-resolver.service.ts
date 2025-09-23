import { Injectable } from '@angular/core';
import { Resolve, ActivatedRouteSnapshot, RouterStateSnapshot } from '@angular/router';
import { Observable, of } from 'rxjs';
import { BenchmarkService } from './benchmark.service';
import { BenchmarkData } from '@app/types/pruning.types';

@Injectable({ providedIn: 'root' })
export class BenchmarkResolver implements Resolve<BenchmarkData | null> {
  constructor(private benchmarkService: BenchmarkService) {}

  resolve(route: ActivatedRouteSnapshot, state: RouterStateSnapshot): Observable<BenchmarkData | null> {
    const uploadId = this.benchmarkService.uploadId;
    if (!uploadId) {
      return of(null); // fallback → could redirect instead
    }
    return this.benchmarkService.fetchData(uploadId);
  }
}
