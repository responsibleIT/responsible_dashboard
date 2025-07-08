import {
  Environment,
  EnvironmentVariables,
} from "@app/types/environment.types";

export const environment: EnvironmentVariables = {
  productionMode: true,
  name: Environment.Production,
  api: {
    schema: "https",
    hostname: "dashboard2-api.alexandervreeswijk.com",
    websocketProtocol: "wss",
  },
};
