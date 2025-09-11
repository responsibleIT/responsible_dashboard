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
  private upload_id: string | null = null;
  private hardClosed = false; // <— NEW

  private connection$ = new Subject<ConnStatus>();
  private messages$   = new Subject<any>();

  constructor(private readonly uploadService: UploadService) {}

  private baseUrl(): string {
    return `${environment.api.schema}://${environment.api.hostname}`;
  }

  connect(upload_id: string): void {
    this.upload_id = upload_id;  // <-- make sure internal state matches
    // guard: if we previously hard-closed, reset flag so this connect is explicit
    this.hardClosed = false;

    // create a brand-new socket with reconnection disabled by default
    this.socket = io(this.baseUrl(), {
      transports: ['websocket'],
      withCredentials: false,
      reconnection: false,     // <— IMPORTANT: no auto-reconnect
      autoConnect: true,
    });

    this.socket.on('connect', () => {
      this.connection$.next('connected');
      this.socket!.emit('join', { upload_id: upload_id });
    });

    this.socket.on('status', (msg) => this.messages$.next(msg));
    this.socket.on('disconnect', () => this.connection$.next('disconnected'));
    this.socket.on('connect_error', (err) =>
      this.messages$.next({ type: 'error', message: String(err) })
    );
  }

  getConnectionStatus(): Observable<ConnStatus> {
    return this.connection$.asObservable();
  }
  getMessages(): Observable<any> {
    return this.messages$.asObservable();
  }

  getUploadId(): string | null {
    return this.upload_id;
  }

  sendMessage(msg: { event?: string; type?: string; data?: any } & Record<string, any>): void {
    if (!this.socket) return;
    const event = (msg.event ?? msg.type) as string;
    if (!event) return;
    const { event: _e, type: _t, ...rest } = msg;
    const payload = msg.data ?? rest;
    this.socket.emit(event, payload);
  }

  set UploadId(id: string | null) { this.upload_id = id; }
  get UploadId(): string | null { return this.upload_id; }

  start(): void {
    if (!this.socket) return;
    this.socket.emit('start', { upload_id: this.upload_id });
  }

  validate(threshold: number): void {
    if (!this.socket) return;
    this.socket.emit('validate', { upload_id: this.upload_id, threshold });
  }

  // *** HARD teardown: prevents “invalid frame header” reconnect churn ***
  disconnect(): void {
    this.hardClosed = true;

    if (this.socket) {
      try {
        // stop any re-connect behaviour on this instance
        // (defensive: some builds expose as io.opts or via methods)
        // @ts-ignore
        if (this.socket.io?.reconnection) this.socket.io.reconnection(false);

        this.socket.removeAllListeners();   // remove handlers
        this.socket.disconnect();           // logical disconnect
        // @ts-ignore
        if (this.socket.close) this.socket.close(); // physical close if available
      } catch { /* noop */ }
    }

    this.socket = undefined;
    this.connection$.next('disconnected');
  }
}
