import { Routes } from '@angular/router';
import {PruningAdjustmentsComponent} from './pages/pruning-adjustments/pruning-adjustments.component';
import {UploadLoaderComponent} from '@app/pages/loaders/upload-loader/upload-loader.component';
import {LandingPageComponent} from '@app/pages/landing-page/landing-page.component';
import {BenchmarkLoaderComponent} from '@app/pages/loaders/benchmark-loader/benchmark-loader.component';
import {BenchmarkResultsComponent} from '@app/pages/benchmark-results/benchmark-results.component';

export const routes: Routes = [
  {
    path: '',
    pathMatch: 'full',
    component: LandingPageComponent,
  },
  {
    path: 'loading-upload',
    component: UploadLoaderComponent,
  },
  {
    path: 'pruning-adjustments',
    component: PruningAdjustmentsComponent,
  },
  {
    path: 'loading-benchmark',
    component: BenchmarkLoaderComponent,
  },
  {
    path: 'benchmark-results',
    component: BenchmarkResultsComponent,
  },
]
