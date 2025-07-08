import { Injectable } from '@angular/core';
import {BehaviorSubject} from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class SettingsService {

  public gpu = new BehaviorSubject<string | null>(null);
  public location = new BehaviorSubject<string | null>(null);
  public threshold = new BehaviorSubject<number>(0);

  public set Gpu(gpu: string | null) {
    this.gpu.next(gpu);
  }

  public get Gpu(): string | null {
    return this.gpu.getValue();
  }

  public set Location(location: string | null) {
    this.location.next(location);
  }

  public get Location(): string | null {
    return this.location.getValue();
  }

  public set Threshold(threshold: number) {
    this.threshold.next(threshold);
  }

  public get Threshold(): number {
    return this.threshold.getValue();
  }
}
