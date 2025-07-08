export interface MetricColorScheme {
  [key: string]: MetricColor;
}

export interface MetricColor {
  header: string;
  value: string;
  background: string;
}

export interface MetricBarColorScheme {
  [key: string]: string;
}
