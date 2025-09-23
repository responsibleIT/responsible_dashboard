// apps/sustainability_dashboard/frontend_v2/src/app/pages/loaders/upload-loader/upload-loader.component.ts
import { Component, OnDestroy, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { Subscription, Observable, of } from 'rxjs';
import { filter, take } from 'rxjs/operators';

import { ButtonDirective } from '@app/domains/ui/directives/button/button.directive';
import { WebsocketService } from '@app/services/websocket.service';
import { UploadService } from '@app/services/upload.service';

type ConnStatus = 'connecting' | 'connected' | 'disconnected' | 'error';
type WsMsg = { type?: string; message?: string; [k: string]: any };

@Component({
  selector: 'app-upload-loader',
  standalone: true,
  imports: [ButtonDirective],
  templateUrl: './upload-loader.component.html',
  styleUrls: ['./upload-loader.component.scss'],
})
export class UploadLoaderComponent implements OnInit, OnDestroy {
  public message = 'Uploading…';
  private sub = new Subscription();

  constructor(
    private readonly router: Router,
    private readonly websocketService: WebsocketService,
    private readonly uploadService: UploadService
  ) {}

  ngOnInit(): void {
    const upload_id = this.uploadService.uploadIdValue;
    if (!upload_id) {
      // No context — return to start page
      this.router.navigate(['/']);
      return;
    }

    // 1) Open socket
    this.websocketService.connect(upload_id);

    // 2) Get a connection-status stream that works for either service shape
    const connection$: Observable<ConnStatus> =
      // newer public observable
      (this.websocketService as any).connection$
        ? (this.websocketService as any).connection$
        // older service API method
        : (this.websocketService as any).getConnectionStatus?.() ?? of('connecting');

    // 3) When connected, fire the 'start' event
    this.sub.add(
      connection$
        .pipe(
          filter((s: ConnStatus) => s === 'connected'),
          take(1)
        )
        .subscribe(() => {
          // Support both send() and sendMessage()
          const svc: any = this.websocketService;
          if (typeof svc.send === 'function') {
            svc.send('join',  { upload_id: upload_id });
            svc.send('start', { upload_id: upload_id });
          } else {
            svc.sendMessage({ event: 'join',  data: { upload_id: upload_id } });
            svc.sendMessage({ event: 'start', data: { upload_id: upload_id } });
          }
        })
    );

    // 4) Subscribe to backend progress/messages (supports both shapes)
    const messages$: Observable<WsMsg> =
      (this.websocketService as any).messages$
        ? (this.websocketService as any).messages$
        : (this.websocketService as any).getMessages?.() ?? of({});

    this.sub.add(
      messages$.subscribe((msg: WsMsg) => {
        if (msg?.message) this.message = msg.message;

        // Only treat "upload-complete" as success in the upload loader
        if (msg?.type === 'upload-complete') {
          console.log('[upload-loader] upload complete, moving to pruning adjustments!');
          this.router.navigate(['/pruning-adjustments']);
        } else if (msg?.type === 'error') {
          this.message = msg.message ?? 'An error occurred.';
        }
      })
    );
  }

  onCancel(): void {
    this.router.navigate(['/']);
  }

  ngOnDestroy(): void {
    this.sub.unsubscribe();
  }
}
