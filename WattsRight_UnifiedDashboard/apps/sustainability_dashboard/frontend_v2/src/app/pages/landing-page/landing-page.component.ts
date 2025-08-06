import { Component } from '@angular/core';
import {UpperCasePipe} from '@angular/common';
import {ButtonDirective} from '@app/domains/ui/directives/button/button.directive';
import {Router} from '@angular/router';
import {ModalService} from '@app/services/modal.service';
import {UploadModal} from '@app/pages/landing-page/modals/upload/upload.modal';

@Component({
  selector: 'app-landing-page',
  imports: [
    UpperCasePipe,
    ButtonDirective
  ],
  templateUrl: './landing-page.component.html',
  styleUrl: './landing-page.component.scss'
})
export class LandingPageComponent {

  constructor(
    private router: Router,
    private modalService: ModalService,
  ) {
  }

  public optimizeClick() {
    this.modalService.openDialog({}, UploadModal)
  }

}
