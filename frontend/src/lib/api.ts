const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export type JobPostingInput = {
  title: string;
  location?: string | null;
  department?: string | null;
  salary_range?: string | null;
  company_profile?: string | null;
  description?: string | null;
  requirements?: string | null;
  benefits?: string | null;
  telecommuting?: number | null;
  has_company_logo?: number | null;
  has_questions?: number | null;
  employment_type?: string | null;
  required_experience?: string | null;
  required_education?: string | null;
  industry?: string | null;
  function?: string | null;
};

export type PredictionMeta = {
  model_type?: string | null;
  model_name?: string | null;
};

export type FeatureContribution = {
  feature: string;
  source: string;
  contribution: number;
};

export type PredictionExplanation = {
  top_positive: FeatureContribution[];
  top_negative: FeatureContribution[];
  intercept?: number | null;
  summary?: string | null;
};

export type PredictionResponse = {
  probability_fraud: number;
  binary_label: number;
  decision: string;
  threshold: number;
  gray_zone: {
    width: number;
    lower: number;
    upper: number;
    positive_label?: string;
    negative_label?: string;
    review_label?: string;
  };
  meta: PredictionMeta;
  explanation: PredictionExplanation;
};

export type PredictionBatchResponse = {
  predictions: PredictionResponse[];
};

export type GrayZonePolicy = {
  width: number;
  lower: number;
  upper: number;
  positive_label: string;
  negative_label: string;
  review_label: string;
};

export type MetricSet = {
  f1?: number;
  precision?: number;
  recall?: number;
  roc_auc?: number;
  pr_auc?: number;
  brier?: number;
};

export type MetadataResponse = {
  model_name: string;
  model_type: string;
  feature_type: string;
  calibration_method?: string | null;
  threshold: number;
  gray_zone: GrayZonePolicy;
  val_metrics: MetricSet;
  test_metrics: MetricSet;
  test_ece?: number | null;
};

export type TokenImportanceResponse = {
  positive: { term: string; weight: number }[];
  negative: { term: string; weight: number }[];
};

export type TokenFrequencyResponse = {
  items: {
    token: string;
    positive_count: number;
    negative_count: number;
    difference: number;
  }[];
};

export type ThresholdMetricPoint = {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
};

export type ThresholdMetricsResponse = {
  points: ThresholdMetricPoint[];
};

export type LatencySummaryPoint = {
  batch_size: number;
  latency_p50_ms: number;
  latency_p95_ms: number;
  throughput_rps: number;
};

export type LatencySummaryResponse = {
  items: LatencySummaryPoint[];
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with status ${response.status}`);
  }

  return (await response.json()) as T;
}

export async function fetchMetadata(): Promise<MetadataResponse> {
  return request<MetadataResponse>("/metadata");
}

export async function fetchTokenImportance(limit = 20): Promise<TokenImportanceResponse> {
  return request<TokenImportanceResponse>(`/insights/token-importance?limit=${limit}`);
}

export async function fetchTokenFrequency(limit = 20): Promise<TokenFrequencyResponse> {
  return request<TokenFrequencyResponse>(`/insights/token-frequency?limit=${limit}`);
}

export async function fetchThresholdMetrics(limit = 50): Promise<ThresholdMetricsResponse> {
  return request<ThresholdMetricsResponse>(`/insights/threshold-metrics?limit=${limit}`);
}

export async function fetchLatencySummary(): Promise<LatencySummaryResponse> {
  return request<LatencySummaryResponse>("/insights/latency");
}

export async function predictBatch(instances: JobPostingInput[]): Promise<PredictionBatchResponse> {
  return request<PredictionBatchResponse>("/predict", {
    method: "POST",
    body: JSON.stringify({ instances }),
  });
}

export async function predictSingle(instance: JobPostingInput): Promise<PredictionResponse> {
  const { predictions } = await predictBatch([instance]);
  if (!predictions.length) {
    throw new Error("No predictions returned by API");
  }
  return predictions[0];
}

export async function fetchHealth() {
  return request<{ status: string; model_type: string; threshold: number }>("/health");
}

export function getApiBaseUrl(): string {
  return API_BASE_URL;
}
