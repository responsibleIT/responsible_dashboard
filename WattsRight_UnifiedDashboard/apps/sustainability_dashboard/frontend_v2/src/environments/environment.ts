// src/environments/environment(.production).ts
import { Environment, EnvironmentVariables } from "@app/types/environment.types";

// Compute from the current page so we stay same-origin (localhost:8000 when served by Flask)
const isHttps = window.location.protocol === "https:";
const runtimeHost = window.location.host;
const runtimePort = window.location.port;
const host = runtimePort === "4200"
  ? `${window.location.hostname}:8000`
  : runtimeHost;

export const environment: EnvironmentVariables = {
  productionMode: false,
  name: Environment.Development,
  api: {
    schema:            isHttps ? "https" : "http",
    hostname:          host,
    websocketProtocol: isHttps ? "wss" : "ws",
  },
};