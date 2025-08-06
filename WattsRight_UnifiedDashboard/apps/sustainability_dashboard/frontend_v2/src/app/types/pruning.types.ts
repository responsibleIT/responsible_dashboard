import {FormControl, FormGroup} from '@angular/forms';

interface PruneSettingsFormControls {
  gpu: FormControl<string | null>;
  location: FormControl<string | null>;
  metric: FormControl<string | null>;
  threshold: FormControl<number | null>;
}

export type PruneSettingsFormGroup = FormGroup<PruneSettingsFormControls>;

export type PruningMetricCardList = {
  [key in 'power' | 'performance' | 'emissions' | 'compute']: PruningMetricCard;
};

export type BenchmarkMetricCardList = {
  [key in 'power' | 'performance' | 'emissions' | 'compute']: BenchmarkMetricCard;
};

export interface BenchmarkMetricCard {
  title: string;
  unit: string;
  original: number;
  pruned: number;
  change: number;
}

export interface PruningMetricCard {
  title: string;
  unit: string;
  values: Record<number, number>
}

export interface PruningClassPerformance {
  className: string;
  unit: string | null;
  original: number;
  pruned: {
    [threshold: number]: number;
  }
}

export type PruningTab = 'Charts' | 'Performance per Class';


export interface PruningSettings {
  gpus: string[];
  locations: string[];
  metrics: string[];
}

export interface PruningPlaygroundData {
  tflops: Record<number, number>;
  power: Record<number, number>;
  emissions: Record<number, number>;
  performance: Record<number, number>;
}

export interface MetricData {
  original: number;
  pruned: number;
}

export interface ClassificationData {
  accuracy: MetricData;
  precision: MetricData;
  recall: MetricData;
  f1Score: MetricData;
}

export interface RawBenchmarkMetrics {
  power: MetricData;
  performance: MetricData;
  emissions: MetricData;
  compute: MetricData;
}

export interface BenchmarkData {
  model: string;
  threshold: number;
  gpu: string;
  location: string;
  overall: ClassificationData;
  perClass: Record<string, ClassificationData>;
  originalFlops: number;
  prunedFlops: number;
  reductionPercentage: number;
  originalParameters: number;
  prunedParameters: number;
  metricCards: RawBenchmarkMetrics;
}

export interface ClassPerformance {
  className: string;
  performance: ClassificationData
}
