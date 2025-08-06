import {
  Environment,
  EnvironmentVariables,
} from "@app/types/environment.types";

export const environment: EnvironmentVariables = {
  productionMode: false,
  name: Environment.Development,

  api: {
    schema: "http",
    hostname: "localhost:8000",
    websocketProtocol: "ws",
  },
};

/*
 * For easier debugging in development mode, you can import the following file
 * to ignore zone related error stack frames such as `zone.run`, `zoneDelegate.invokeTask`.
 *
 * This import should be commented out in production mode because it will have a negative impact
 * on performance if an error is thrown.
 */
import "zone.js/plugins/zone-error"; // Included with Angular CLI.
