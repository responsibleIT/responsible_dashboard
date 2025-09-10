import { Component, ElementRef, OnDestroy, OnInit, ViewChild } from '@angular/core';
import { ButtonDirective } from '@app/domains/ui/directives/button/button.directive';
import { Subscription } from 'rxjs';
import { Router } from '@angular/router';
import { WebsocketService } from '@app/services/websocket.service';
import { SettingsService } from '@app/services/settings.service';

@Component({
  selector: 'app-benchmark-loader',
  imports: [ButtonDirective],
  templateUrl: './benchmark-loader.component.html',
  styleUrl: './benchmark-loader.component.scss'
})
export class BenchmarkLoaderComponent implements OnInit, OnDestroy {
  @ViewChild('progressCircle', { static: true }) progressCircle!: ElementRef;

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

    const uploadId = this.websocketService.getUploadId() || 'benchmark';
    this.websocketService.connect(uploadId);

    // When connected, send the *real* benchmark command
    this.subscription.add(
      this.websocketService.getConnectionStatus().subscribe(status => {
        if (status === 'connected') {
          // small delay to ensure the room join finished
          setTimeout(() => {
            this.websocketService.sendMessage({
              event: 'benchmark_real',
              data: {
                upload_id: uploadId,
                threshold: threshold,
                gpu: gpu,           // optional (backend ignores if not needed)
                location: location  // optional
              }
            });
          }, 300);
        } else if (status === 'disconnected') {
          console.log('WebSocket disconnected');
        } else if (status === 'error') {
          console.error('WebSocket error occurred');
        }
      })
    );

    // Listen for backend progress/completion
    this.subscription.add(
      this.websocketService.getMessages().subscribe(message => {
        // backend sends: emit("status", {"message": "...", "type": "complete" | "error" | "loading"})
        if (message.type === 'complete') {
          this.message = message.message;
          this.websocketService.disconnect();
          setTimeout(() => {
            this.router.navigateByUrl('/benchmark-results', { replaceUrl: true });
          }, 800);
        } else if (message.type === 'loading') {
          this.message = message.message;
        } else if (message.type === 'error') {
          this.message = message.message || 'Benchmark failed';
          console.error('[benchmark_real] error:', message);
        }
      })
    );
  }

  onCancel(): void {
    this.router.navigate(['/']);
  }

  ngOnDestroy() {
    this.subscription.unsubscribe();
  }
}
