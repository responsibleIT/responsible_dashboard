import {Component, ElementRef, OnDestroy, OnInit, ViewChild} from '@angular/core';
import {ButtonDirective} from '@app/domains/ui/directives/button/button.directive';
import {Router} from '@angular/router';
import {WebsocketService} from '@app/services/websocket.service';
import {Subscription} from 'rxjs';

@Component({
  selector: 'app-upload-loader',
  imports: [
    ButtonDirective
  ],
  templateUrl: './upload-loader.component.html',
  styleUrl: './upload-loader.component.scss'
})
export class UploadLoaderComponent implements OnInit, OnDestroy {

  @ViewChild('progressCircle', {static: true}) progressCircle!: ElementRef;

  public message: string = 'Uploading...';

  private subscription: Subscription = new Subscription();

  constructor(
    private readonly router: Router,
    private readonly websocketService: WebsocketService,
  ) {}

  ngOnInit() {
    this.websocketService.connect()
    this.subscription.add(
      this.websocketService.getConnectionStatus().subscribe(status => {
        if (status === 'connected') {
          this.websocketService.sendMessage({
            type: 'start',
            gpu: null,
            location: null,
          })
        } else if (status === 'disconnected') {
          console.log('WebSocket disconnected');
        } else if (status === 'error') {
          console.error('WebSocket error occurred');
        }
      })
    )
    this.subscription.add(
      this.websocketService.getMessages().subscribe(message => {
        if (message.type === 'complete') {
          this.message = message.message;
          this.websocketService.disconnect();
          setTimeout(() => {
            this.router.navigateByUrl('/pruning-adjustments', { replaceUrl: true });
          }, 1000);
        } else if (message.type === 'loading') {
          this.message = message.message;
        }
      })
    );

    // this.simulateProgress();
  }

  simulateProgress() {
    setTimeout(() => {
        this.router.navigate(['/pruning-adjustments']);
    }, 2000);
  }

  onCancel(): void {
    this.router.navigate(["/"])
  }

  ngOnDestroy() {
    this.subscription.unsubscribe();
  }

}
