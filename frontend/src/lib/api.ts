import * as MockData from "./mock-data";
import { useBackendStatus, useDemoReviewQueue } from "./backend-status";
import { API_BASE_URL, getApiBaseUrl } from "./config";

// Function to check if we should use mock data
function shouldUseMockData(): boolean {
  if (typeof window === "undefined") return false;

  try {
    const status = useBackendStatus.getState();
    return !status.isConnected;
  } catch {
    return false;
  }
}

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
  request_id: string;
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

export type ModelSummary = {
  model_name: string;
  model_type: string;
  calibration_method?: string | null;
  threshold?: number | null;
  timestamp?: string | null;
  validation: MetricSet;
  test: MetricSet;
};

export type ModelsResponse = {
  items: ModelSummary[];
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

export type SliceMetric = {
  slice: string;
  category: string;
  count: number;
  f1?: number | null;
  precision?: number | null;
  recall?: number | null;
};

export type SliceMetricsResponse = {
  items: SliceMetric[];
};

export type ReviewCase = {
  request_id: string;
  created_at: string;
  probability: number;
  predicted_label: string;
  model_version: string;
  threshold?: number | null;
  text_hash: string;
  features_hash: string;
  payload: {
    title?: string | null;
    company_profile?: string | null;
    description?: string | null;
    requirements?: string | null;
    benefits?: string | null;
    location?: string | null;
    employment_type?: string | null;
    required_experience?: string | null;
    required_education?: string | null;
    industry?: string | null;
    function?: string | null;
  };
  explanation: PredictionExplanation;
};

export type CasesResponse = {
  total_pending: number;
  items: ReviewCase[];
};

export type FeedbackPayload = {
  request_id: string;
  model_version: string;
  proba: number;
  predicted_label: "fraud" | "legit" | "review";
  reviewer_label: "fraud" | "legit" | "unsure";
  text_hash: string;
  features_hash: string;
  rationale?: string | null;
  notes?: string | null;
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
  if (shouldUseMockData()) {
    return MockData.mockFetchMetadata();
  }
  return request<MetadataResponse>("/metadata");
}

export async function fetchModelSummaries(limit = 20): Promise<ModelsResponse> {
  if (shouldUseMockData()) {
    return MockData.mockFetchModelSummaries(limit);
  }
  return request<ModelsResponse>(`/models?limit=${limit}`);
}

export async function fetchTokenImportance(limit = 20): Promise<TokenImportanceResponse> {
  if (shouldUseMockData()) {
    return MockData.mockFetchTokenImportance(limit);
  }
  return request<TokenImportanceResponse>(`/insights/token-importance?limit=${limit}`);
}

export async function fetchTokenFrequency(limit = 20): Promise<TokenFrequencyResponse> {
  if (shouldUseMockData()) {
    return MockData.mockFetchTokenFrequency(limit);
  }
  return request<TokenFrequencyResponse>(`/insights/token-frequency?limit=${limit}`);
}

export async function fetchThresholdMetrics(limit = 50): Promise<ThresholdMetricsResponse> {
  if (shouldUseMockData()) {
    return MockData.mockFetchThresholdMetrics(limit);
  }
  return request<ThresholdMetricsResponse>(`/insights/threshold-metrics?limit=${limit}`);
}

export async function fetchLatencySummary(): Promise<LatencySummaryResponse> {
  if (shouldUseMockData()) {
    return MockData.mockFetchLatencySummary();
  }
  return request<LatencySummaryResponse>("/insights/latency");
}

export async function fetchSliceMetrics(limit = 6): Promise<SliceMetricsResponse> {
  if (shouldUseMockData()) {
    return MockData.mockFetchSliceMetrics(limit);
  }
  return request<SliceMetricsResponse>(`/insights/slice-metrics?limit=${limit}`);
}

export async function fetchReviewCases(
  limit = 25,
  policy = "gray-zone",
  offset = 0
): Promise<CasesResponse> {
  if (shouldUseMockData()) {
    return MockData.mockFetchReviewCases(limit, policy, offset);
  }
  return request<CasesResponse>(`/cases?policy=${policy}&limit=${limit}&offset=${offset}`);
}

export async function fetchReviewCount(): Promise<number> {
  const result = await fetchReviewCases(1);
  return result.total_pending;
}

export async function submitFeedback(records: FeedbackPayload[]): Promise<void> {
  if (shouldUseMockData()) {
    await new Promise((resolve) => setTimeout(resolve, 300));

    if (typeof window !== "undefined") {
      try {
        const demoQueue = useDemoReviewQueue.getState();
        records.forEach((record) => {
          demoQueue.removeCase(record.request_id);
        });
      } catch (e) {
        console.error("Failed to remove cases from demo queue:", e);
      }
    }

    return;
  }
  await request<{ inserted: number }>("/feedback", {
    method: "POST",
    body: JSON.stringify(records),
  });
}

export async function predictBatch(instances: JobPostingInput[]): Promise<PredictionBatchResponse> {
  if (shouldUseMockData()) {
    return MockData.mockPredictBatch(instances);
  }
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

  const prediction = predictions[0];

  if (shouldUseMockData() && prediction.decision === "review") {
    if (typeof window !== "undefined") {
      try {
        const demoQueue = useDemoReviewQueue.getState();
        demoQueue.addCase(prediction, instance);
      } catch (e) {
        console.error("Failed to add case to demo queue:", e);
      }
    }
  }

  return prediction;
}

export async function fetchHealth() {
  if (shouldUseMockData()) {
    return MockData.mockFetchHealth();
  }
  return request<{ status: string; model_type: string; threshold: number }>("/health");
}

export { getApiBaseUrl };

export type ChatContext = {
  request_id?: string | null;
  job_posting?: JobPostingInput | null;
  prediction?: PredictionResponse | null;
};

export type ChatMessage = {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp?: string;
};

export type ChatRequest = {
  message: string;
  context?: ChatContext | null;
  session_id?: string | null;
  history?: ChatMessage[] | null;
};

export type ChatStreamChunk = {
  chunk: string;
  done: boolean;
};

export async function streamChat(
  request: ChatRequest,
  onChunk: (chunk: string) => void,
  onComplete: () => void,
  onError: (error: Error) => void
): Promise<void> {
  if (shouldUseMockData()) {
    return MockData.mockStreamChat(request.message, onChunk, onComplete, onError);
  }

  try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Chat API error: ${response.status} - ${errorText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error("No response body reader available");
    }

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const jsonStr = line.slice(6);
            const data: ChatStreamChunk = JSON.parse(jsonStr);
            if (data.chunk) {
              onChunk(data.chunk);
            }
            if (data.done) {
              onComplete();
              return;
            }
          } catch (e) {
            console.error("Failed to parse SSE data:", e);
          }
        }
      }
    }

    onComplete();
  } catch (error) {
    onError(error as Error);
  }
}
