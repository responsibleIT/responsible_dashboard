import { Injectable } from '@angular/core';
import {BehaviorSubject, Observable, Subject} from 'rxjs';
import {WebsocketMessageRequest, WebsocketResponse} from '@app/types/websocket.types';
import {environment} from '@env/environment';

@Injectable({
  providedIn: 'root'
})
export class WebsocketService {

  private readonly apiUrl = `${environment.api.websocketProtocol}://${environment.api.hostname}`;

  public uploadId = new BehaviorSubject<string | null>(null);
  private websocket: WebSocket | null = null;
  private messagesSubject = new Subject<WebsocketResponse>();
  private connectionStatusSubject = new BehaviorSubject<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');

  constructor(
    private readonly localStorage: Storage,
  ) {
    const storedUploadId = this.localStorage.getItem('uploadId');
    if (!storedUploadId) {
      return
    }

    this.uploadId.next(storedUploadId);
  }

  connect(): Observable<WebsocketResponse> {

    if (this.websocket) {
      this.websocket.close();
    }

    const uploadId = this.uploadId.value;
    if (!uploadId) {
      throw new Error('Upload ID is not set. Please set the UploadId before connecting.');
    }

    const wsUrl = `${this.apiUrl}/ws/${uploadId}`;

    this.connectionStatusSubject.next('connecting');
    this.websocket = new WebSocket(wsUrl);

    this.websocket.onopen = (event) => {
      this.connectionStatusSubject.next('connected');
    };

    this.websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.messagesSubject.next(data);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    this.websocket.onclose = (event) => {
      this.connectionStatusSubject.next('disconnected');
      this.websocket = null;
    };

    this.websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.connectionStatusSubject.next('error');
    };

    return this.messagesSubject.asObservable();
  }

  sendMessage(message: WebsocketMessageRequest): void {
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected. Message not sent:', message);
    }
  }

  getMessages(): Observable<WebsocketResponse> {
    return this.messagesSubject.asObservable();
  }

  disconnect(): void {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
      this.connectionStatusSubject.next('disconnected');
    }
  }

  getConnectionStatus(): Observable<'connecting' | 'connected' | 'disconnected' | 'error'> {
    return this.connectionStatusSubject.asObservable();
  }

  get isConnected(): boolean {
    return this.websocket?.readyState === WebSocket.OPEN;
  }

  set UploadId(uploadId: string) {
    this.uploadId.next(uploadId);
    this.localStorage.setItem('uploadId', uploadId);
  }
}
