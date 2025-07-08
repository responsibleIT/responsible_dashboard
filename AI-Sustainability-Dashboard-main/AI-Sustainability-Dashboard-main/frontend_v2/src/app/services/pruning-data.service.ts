import { Injectable } from '@angular/core';
import {BehaviorSubject, catchError, map, Observable, throwError} from 'rxjs';
import {PruningPlaygroundData, PruningSettings} from '@app/types/pruning.types';
import {HttpClient} from '@angular/common/http';
import {environment} from '@env/environment';

@Injectable({
  providedIn: 'root'
})
export class PruningDataService {

  private readonly apiUrl = `${environment.api.schema}://${environment.api.hostname}`;
  public data$ = new BehaviorSubject<PruningPlaygroundData | null>(null);

  constructor(
    private http: HttpClient
  ) { }

  public fetchSettings(): Observable<PruningSettings> {
    return this.http.get(`${this.apiUrl}/settings`).pipe(
      map(response => response as PruningSettings),
      catchError(error => throwError(() => new Error('Error fetching settings: ' + error.message)))
    )
  }

  public fetchData(uploadId: string, gpu: string, location: string, metric: string): Observable<PruningPlaygroundData> {
    return this.http.get(`${this.apiUrl}/chart-data/${uploadId}/${gpu}/${location}`).pipe(
      map(response => response as PruningPlaygroundData),
      catchError(error => throwError(() => new Error('Error fetching playground data: ' + error.message)))
    )
  }

  public set Data(data: PruningPlaygroundData | null) {
    this.data$.next(data);
  }

  public get Data(): PruningPlaygroundData | null {
    return this.data$.getValue();
  }
}
