import {inject, Injectable} from '@angular/core';
import {MatDialog} from "@angular/material/dialog";
import {ComponentType} from "@angular/cdk/overlay";

@Injectable({
  providedIn: 'root'
})
export class ModalService {

  matDialog = inject(MatDialog);

  openDialog<T>(data : any, component: ComponentType<T>) {
    return this.matDialog.open(component,{
      data : data,
      disableClose: false,
      width: '650px',
    })
  }

}
