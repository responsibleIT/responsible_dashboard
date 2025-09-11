// apps/sustainability_dashboard/frontend_v2/src/app/services/websocket.service.ts
import { Injectable } from '@angular/core';
import { io, Socket } from 'socket.io-client';
import { Subject, Observable } from 'rxjs';
import { environment } from '@env/environment';
import { UploadService } from '@app/services/upload.service';

type ConnStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

@Injectable({ providedIn: 'root' })
export class WebsocketService {
  private socket?: Socket;
  private uploadId: string | null = null;

  private connection$ = new Subject<ConnStatus>();
  private messages$ = new Subject<any>();

  constructor(private readonly uploadService: UploadService) {
    // keep in sync with UploadService so we always have the latest id
    this.uploadService.uploadId.subscribe(id => (this.uploadId = id));
  }

  private baseUrl(): string {
    return (window?.location?.origin ?? `${environment.api.schema}://${environment.api.hostname}`);
  }

  connect(uploadId: string): void {
    // IMPORTANT: remember it
    this.uploadId = uploadId;

    this.socket = io(this.baseUrl(), {
      transports: ['websocket'],
      withCredentials: false,
    });

    this.connection$.next('connecting');

    this.socket.on('connect', () => {
      this.connection$.next('connected');
      // join the room for this upload
      this.socket!.emit('join', { upload_id: this.uploadId });
    });

    this.socket.on('status', (msg) => this.messages$.next(msg));
    this.socket.on('disconnect', () => this.connection$.next('disconnected'));
    this.socket.on('connect_error', (err) => {
      this.connection$.next('error');
      this.messages$.next({ type: 'error', message: String(err) });
    });
  }

  getConnectionStatus(): Observable<ConnStatus> {
    return this.connection$.asObservable();
  }

  getMessages(): Observable<any> {
    return this.messages$.asObservable();
  }

  getUploadId(): string | null {
    return this.uploadId;
  }

  /**
   * Generic sender used by some existing components:
   * sendMessage({ event: 'start', data: {...} })
   */
  sendMessage(msg: { event?: string; type?: string; data?: any } & Record<string, any>): void {
    if (!this.socket) return;

    const event = (msg.event ?? msg.type) as string;
    if (!event) return;

    const { event: _e, type: _t, ...rest } = msg;
    const payload = (msg.data ?? rest) ?? {};

    // Auto-attach upload_id if not provided
    if (this.uploadId && payload.upload_id == null) {
      payload.upload_id = this.uploadId;
    }

    this.socket.emit(event, payload);
  }

  // Convenience methods
  start(): void {
    if (!this.socket) return;
    this.socket.emit('start', { upload_id: this.uploadId });
  }

  validate(threshold: number): void {
    if (!this.socket) return;
    this.socket.emit('validate', { upload_id: this.uploadId, threshold });
  }

  disconnect(): void {
    this.socket?.disconnect();
    this.socket = undefined;
    this.connection$.next('disconnected');
  }

  // Setter kept for compatibility if you set it manually elsewhere
  set UploadId(id: string | null) { this.uploadId = id; }
  get UploadId(): string | null { return this.uploadId; }
}
