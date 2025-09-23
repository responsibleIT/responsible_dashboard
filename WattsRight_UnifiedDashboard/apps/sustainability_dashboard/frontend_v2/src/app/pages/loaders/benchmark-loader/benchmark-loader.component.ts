import { Component, ElementRef, OnDestroy, OnInit, ViewChild } from '@angular/core';
import { ButtonDirective } from '@app/domains/ui/directives/button/button.directive';
import { Subscription } from 'rxjs';
import { Router } from '@angular/router';
import { WebsocketService } from '@app/services/websocket.service';
import { SettingsService } from '@app/services/settings.service';
import { UploadService } from '@app/services/upload.service';
import { BenchmarkService } from '@app/services/benchmark.service';

@Component({
  selector: 'app-benchmark-loader',
  standalone: true,
  imports: [ButtonDirective],
  templateUrl: './benchmark-loader.component.html',
  styleUrls: ['./benchmark-loader.component.scss']
})
export class BenchmarkLoaderComponent implements OnInit, OnDestroy {
  @ViewChild('progressCircle', { static: true }) progressCircle!: ElementRef;

  public message: string = 'Model is being loaded...';
  private subscription: Subscription = new Subscription();

  constructor(
    private readonly router: Router,
    private readonly websocketService: WebsocketService,
    private readonly uploadService: UploadService,
    private readonly settingsService: SettingsService,
    private readonly benchmarkService: BenchmarkService
  ) {}

  ngOnInit() {
    // 1. Resolve upload_id from (a) query param, (b) UploadService, (c) localStorage via UploadService
    const fromQuery = this.router.parseUrl(this.router.url).queryParams['upload_id'];
    const uploadId = fromQuery
      || this.uploadService?.uploadIdValue     // preferred
      || null;

    if (!uploadId) {
      console.error('[benchmark-loader] Missing upload_id, going back to start');
      this.router.navigate(['/']);
      return;
    }

    // ✅ save globally for resolver
    this.benchmarkService.setUploadId(uploadId); // 👈 stash it for the resolver

    // 2. Open websocket with explicit id
    this.websocketService.connect(uploadId);

    // 3. Pull the other settings (they should already be set earlier in the flow)
    const threshold = this.settingsService.Threshold;
    const gpu = this.settingsService.Gpu;
    const location = this.settingsService.Location;

    // 4. Subscribe to connection + send command
    this.subscription.add(
      this.websocketService.getConnectionStatus().subscribe(status => {
        if (status === 'connected') {
          setTimeout(() => {
            this.websocketService.sendMessage({
              event: 'benchmark_real',
              data: { upload_id: uploadId, threshold, gpu, location }
            });
          }, 200);
        }
      })
    );

    // 5. Listen for messages
    this.subscription.add(
      this.websocketService.getMessages().subscribe(msg => {
        if (msg.type === 'loading' && msg.message) {
          this.message = msg.message;
        } else if (msg?.type === 'benchmark-complete') {
          this.message = msg.message || 'Benchmark complete.';
          setTimeout(() => {
            this.router.navigate(['/benchmark-results'], {
              replaceUrl: true,
              queryParams: { upload_id: uploadId }
            });
          }, 300);
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
