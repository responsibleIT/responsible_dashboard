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

  // subjects expected by legacy components
  private connection$ = new Subject<ConnStatus>();
  private messages$ = new Subject<any>();

  constructor(private readonly uploadService: UploadService) {}

  // --- helpers ---------------------------------------------------------------
  private baseUrl(): string {
    return `${environment.api.schema}://${environment.api.hostname}`;
  }

  // --- API expected by existing components -----------------------------------
  connect(uploadId: string): void {
    this.socket = io(this.baseUrl(), {
      transports: ['websocket'],   // avoid long-polling churn
      withCredentials: false,      // CORS simplicity for local
    });

    this.socket.on('connect', () => {
      this.connection$.next('connected');
      this.socket!.emit('join', { upload_id: uploadId });
    });

    this.socket.on('status', (msg) => this.messages$.next(msg));
    this.socket.on('disconnect', () => this.connection$.next('disconnected'));
    this.socket.on('connect_error', (err) => this.messages$.next({type:'error', message: String(err)}));
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

    // Determine the event name (prefer `event`, fallback to legacy `type`)
    const event = (msg.event ?? msg.type) as string;
    if (!event) return;

    // Build payload:
    // - If `data` is provided, use it.
    // - Otherwise, use all properties except `event` / `type`.
    const { event: _e, type: _t, ...rest } = msg;
    const payload = msg.data ?? rest;

    this.socket.emit(event, payload);
  }

  // --- Convenience methods used elsewhere ------------------------------------
  set UploadId(id: string | null) {
    this.uploadId = id;
  }
  get UploadId(): string | null {
    return this.uploadId;
  }

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
}
