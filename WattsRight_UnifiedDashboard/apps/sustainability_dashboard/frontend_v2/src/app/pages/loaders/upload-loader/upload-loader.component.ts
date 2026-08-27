// apps/sustainability_dashboard/frontend_v2/src/app/pages/loaders/upload-loader/upload-loader.component.ts
import { Component, OnDestroy, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { Subscription, Observable, of, interval } from 'rxjs';
import { filter, take } from 'rxjs/operators';
import { NgFor, NgIf, NgClass } from '@angular/common';

import { ButtonDirective } from '@app/domains/ui/directives/button/button.directive';
import { WebsocketService } from '@app/services/websocket.service';
import { UploadService } from '@app/services/upload.service';

type ConnStatus = 'connecting' | 'connected' | 'disconnected' | 'error';
type WsMsg = { type?: string; message?: string; [k: string]: any };

interface LoadingStage {
  label: string;
  description: string;
  status: 'pending' | 'active' | 'completed';
}

@Component({
  selector: 'app-upload-loader',
  standalone: true,
  imports: [ButtonDirective, NgFor, NgIf, NgClass],
  templateUrl: './upload-loader.component.html',
  styleUrls: ['./upload-loader.component.scss'],
})
export class UploadLoaderComponent implements OnInit, OnDestroy {
  public message = 'Connecting…';
  public currentHint = '';
  public isGenerative = false;
  private sub = new Subscription();
  private hintIndex = 0;

  public stages: LoadingStage[] = [];

  private readonly generativeStages: LoadingStage[] = [
    { label: 'Loading model', description: 'Downloading and initializing the model architecture', status: 'pending' },
    { label: 'Benchmarking baseline (unpruned model)', description: 'Evaluating token predictions on benchmark data', status: 'pending' },
    { label: 'Benchmarking threshold 10 (max pruning)', description: 'Evaluating pruned model at maximum threshold', status: 'pending' },
    { label: 'Generating surrogate predictions', description: 'Building probabilistic pruning trade-off curves', status: 'pending' },
    { label: 'Post-processing results', description: 'Computing sustainability metrics and uncertainty estimates', status: 'pending' },
  ];

  private readonly classificationStages: LoadingStage[] = [
    { label: 'Loading model', description: 'Initializing model', status: 'pending' },
    { label: 'Running evaluation', description: 'Evaluating model performance', status: 'pending' },
    { label: 'Generating predictions', description: 'Computing pruning curves', status: 'pending' },
  ];

  private readonly educationalHints: string[] = [
    'Perplexity measures how well a model predicts the next token lower is better.',
    'Surrogate modelling reduces benchmarking cost by predicting outcomes without running full evaluations.',
    'FLOPs (Floating Point Operations) indicate computational workload fewer FLOPs mean less energy consumption.',
    'Uncertainty estimates help you understand how confident the predictions are at each pruning level.',
    'Magnitude pruning removes weights with the smallest absolute values, which often contribute least to model output.',
    'The knee point in the trade-off curve suggests where further pruning yields diminishing returns.',
    'CO₂ emissions from AI inference depend on the energy grid mix of your deployment location.',
  ];

  constructor(
    private readonly router: Router,
    private readonly websocketService: WebsocketService,
    private readonly uploadService: UploadService
  ) {}

  ngOnInit(): void {
    const upload_id = this.uploadService.uploadIdValue;
    if (!upload_id) {
      this.router.navigate(['/']);
      return;
    }

    this.isGenerative = this.uploadService.modelTypeValue === 'generative';
    this.stages = this.isGenerative
      ? this.generativeStages.map(s => ({ ...s }))
      : this.classificationStages.map(s => ({ ...s }));

    // Set first stage as active
    if (this.stages.length) {
      this.stages[0].status = 'active';
    }

    // Rotate educational hints during loading
    if (this.isGenerative) {
      this.currentHint = this.educationalHints[0];
      this.sub.add(
        interval(6000).subscribe(() => {
          this.hintIndex = (this.hintIndex + 1) % this.educationalHints.length;
          this.currentHint = this.educationalHints[this.hintIndex];
        })
      );
    }

    // 1) Open socket
    this.websocketService.connect(upload_id);

    // 2) Get a connection-status stream that works for either service shape
    const connection$: Observable<ConnStatus> =
      (this.websocketService as any).connection$
        ? (this.websocketService as any).connection$
        : (this.websocketService as any).getConnectionStatus?.() ?? of('connecting');

    // 3) When connected, fire the 'start' event
    this.sub.add(
      connection$
        .pipe(
          filter((s: ConnStatus) => s === 'connected'),
          take(1)
        )
        .subscribe(() => {
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

    // 4) Subscribe to backend progress/messages
    const messages$: Observable<WsMsg> =
      (this.websocketService as any).messages$
        ? (this.websocketService as any).messages$
        : (this.websocketService as any).getMessages?.() ?? of({});

    this.sub.add(
      messages$.subscribe((msg: WsMsg) => {
        if (msg?.message) {
          this.message = msg.message;
          this.updateStageFromMessage(msg.message);
        }

        if (msg?.type === 'upload-complete') {
          // Complete remaining stages one by one with delay, then navigate
          this.completeStagesSequentially().then(() => {
            setTimeout(() => {
              this.router.navigate(['/pruning-adjustments']);
            }, 1000);
          });
        } else if (msg?.type === 'error') {
          this.message = msg.message ?? 'An error occurred.';
        }
      })
    );
  }

  private currentStageIndex = 0;

  private updateStageFromMessage(message: string): void {
    const lower = message.toLowerCase();
    let stage = this.currentStageIndex;

    // Match from highest stage first (most specific) to prevent regression
    if (lower.includes('post-process') || lower.includes('saving') || lower.includes('building dashboard') || lower.includes('finaliz')) {
      stage = 4;
    } else if (lower.includes('surrogate') || lower.includes('predicting perplexity') || lower.includes('sweep')) {
      stage = 3;
    } else if (lower.includes('threshold 10') || lower.includes('max pruning') || lower.includes('pruning at threshold')) {
      stage = 2;
    } else if (lower.includes('baseline') || lower.includes('computing flops') || lower.includes('evaluating unpruned')) {
      stage = 1;
    } else if (lower.includes('loading') || lower.includes('downloading') || lower.includes('initializ')) {
      stage = 0;
    }

    // Only advance forward, never go backward
    if (stage > this.currentStageIndex) {
      this.currentStageIndex = stage;
      this.activateStage(stage);
    } else if (this.currentStageIndex === 0 && stage === 0) {
      this.activateStage(0);
    }
  }

  private activateStage(index: number): void {
    if (index >= this.stages.length) return;

    for (let i = 0; i < this.stages.length; i++) {
      if (i < index) {
        this.stages[i].status = 'completed';
      } else if (i === index) {
        this.stages[i].status = 'active';
      } else {
        this.stages[i].status = 'pending';
      }
    }
  }

  get completedStages(): number {
    return this.stages.filter(s => s.status === 'completed').length;
  }

  get progressPercent(): number {
    if (!this.stages.length) return 0;
    const completed = this.completedStages;
    const activeBonus = this.stages.some(s => s.status === 'active') ? 0.5 : 0;
    return Math.min(100, ((completed + activeBonus) / this.stages.length) * 100);
  }

  onCancel(): void {
    this.router.navigate(['/']);
  }

  private completeStagesSequentially(): Promise<void> {
    return new Promise<void>((resolve) => {
      let idx = this.currentStageIndex;
      const completeNext = () => {
        if (idx < this.stages.length) {
          this.stages[idx].status = 'completed';
          idx++;
          if (idx < this.stages.length) {
            this.stages[idx].status = 'active';
          }
          setTimeout(completeNext, 1000);
        } else {
          resolve();
        }
      };
      completeNext();
    });
  }

  ngOnDestroy(): void {
    this.sub.unsubscribe();
  }
}
