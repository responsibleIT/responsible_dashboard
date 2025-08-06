import {Component, ElementRef, OnDestroy, OnInit, ViewChild} from '@angular/core';
import {ButtonDirective} from "@app/domains/ui/directives/button/button.directive";
import {Subscription} from 'rxjs';
import {Router} from '@angular/router';
import {WebsocketService} from '@app/services/websocket.service';
import {SettingsService} from '@app/services/settings.service';

@Component({
  selector: 'app-benchmark-loader',
    imports: [
        ButtonDirective
    ],
  templateUrl: './benchmark-loader.component.html',
  styleUrl: './benchmark-loader.component.scss'
})
export class BenchmarkLoaderComponent implements OnInit, OnDestroy {

  @ViewChild('progressCircle', {static: true}) progressCircle!: ElementRef;

  public message: string = 'Model is being loaded...';

  private subscription: Subscription = new Subscription();

  constructor(
    private readonly router: Router,
    private readonly websocketService: WebsocketService,
    private readonly settingsService: SettingsService
  ) {}

  ngOnInit() {
    const threshold = this.settingsService.Threshold;
    const gpu = this.settingsService.Gpu;
    const location = this.settingsService.Location;

    this.websocketService.connect()
    this.subscription.add(
      this.websocketService.getConnectionStatus().subscribe(status => {
        if (status === 'connected') {
          setTimeout(() => {
            this.websocketService.sendMessage({
              type: 'validate',
              threshold: threshold,
              gpu: gpu,
              location: location
            });
          }, 500);
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
            this.router.navigateByUrl('/benchmark-results', { replaceUrl: true });
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
      this.router.navigate(['/benchmark-results']);
    }, 2000);
  }

  onCancel(): void {
    this.router.navigate(["/"])
  }

  ngOnDestroy() {
    this.subscription.unsubscribe();
  }

}
