export const enum Environment {
  Development = "development",
  Production = "production",
}

export interface EnvironmentVariables {
  readonly productionMode: boolean;
  readonly name: Environment;

  /**
   * The main GraphQL API URL and credentials, used for queries and mutations.
   */
  readonly api: {
    readonly schema: "http" | "https";
    readonly hostname: string;
    readonly websocketProtocol: "ws" | "wss";
  };
}
