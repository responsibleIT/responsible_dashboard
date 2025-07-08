import { bootstrapApplication } from '@angular/platform-browser';
import { appConfig } from '@app/app.config';
import { AppComponent } from '@app/app.component';
import {environment} from '@env/environment';
import {enableProdMode} from '@angular/core';

if (environment.productionMode) {
  enableProdMode()
}

bootstrapApplication(AppComponent, appConfig)
  .catch((err) => console.error(err));
