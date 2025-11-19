"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";
import { getApiBaseUrl } from "./config";
import type { ReviewCase, PredictionResponse, JobPostingInput } from "./api";

export type BackendStatus = {
  isConnected: boolean;
  isChecking: boolean;
  lastChecked: Date | null;
  error: string | null;
};

type BackendStatusStore = BackendStatus & {
  checkConnection: () => Promise<void>;
  setConnected: (connected: boolean) => void;
};

type DemoReviewQueueStore = {
  cases: ReviewCase[];
  addCase: (prediction: PredictionResponse, jobPosting: JobPostingInput) => void;
  removeCase: (requestId: string) => void;
  getCases: () => ReviewCase[];
  getCount: () => number;
};

export const useBackendStatus = create<BackendStatusStore>((set, get) => ({
  isConnected: false,
  isChecking: false,
  lastChecked: null,
  error: null,

  checkConnection: async () => {
    const { isChecking } = get();
    if (isChecking) return;

    set({ isChecking: true, error: null });

    try {
      const apiBaseUrl = getApiBaseUrl();
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);

      const response = await fetch(`${apiBaseUrl}/health`, {
        method: "GET",
        signal: controller.signal,
        cache: "no-store",
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        set({
          isConnected: true,
          isChecking: false,
          lastChecked: new Date(),
          error: null,
        });
      } else {
        set({
          isConnected: false,
          isChecking: false,
          lastChecked: new Date(),
          error: `Backend returned status ${response.status}`,
        });
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      set({
        isConnected: false,
        isChecking: false,
        lastChecked: new Date(),
        error: errorMessage.includes("aborted") ? "Connection timeout" : errorMessage,
      });
    }
  },

  setConnected: (connected: boolean) => {
    set({ isConnected: connected });
  },
}));

if (typeof window !== "undefined") {
  useBackendStatus.getState().checkConnection();

  setInterval(
    () => {
      useBackendStatus.getState().checkConnection();
    },
    5 * 60 * 1000
  );
}

export const useDemoReviewQueue = create<DemoReviewQueueStore>()(
  persist(
    (set, get) => ({
      cases: [],

      addCase: (prediction: PredictionResponse, jobPosting: JobPostingInput) => {
        if (prediction.decision !== "review") return;

        const reviewCase: ReviewCase = {
          request_id: prediction.request_id,
          created_at: new Date().toISOString(),
          probability: prediction.probability_fraud,
          predicted_label: prediction.decision,
          model_version: prediction.meta.model_name || "demo-model",
          threshold: prediction.threshold,
          text_hash: `hash-${prediction.request_id}`,
          features_hash: `features-${prediction.request_id}`,
          payload: {
            title: jobPosting.title || null,
            company_profile: jobPosting.company_profile || null,
            description: jobPosting.description || null,
            requirements: jobPosting.requirements || null,
            benefits: jobPosting.benefits || null,
            location: jobPosting.location || null,
            employment_type: jobPosting.employment_type || null,
            required_experience: jobPosting.required_experience || null,
            required_education: jobPosting.required_education || null,
            industry: jobPosting.industry || null,
            function: jobPosting.function || null,
          },
          explanation: prediction.explanation,
        };

        set((state) => ({
          cases: [reviewCase, ...state.cases],
        }));
      },

      removeCase: (requestId: string) => {
        set((state) => ({
          cases: state.cases.filter((c) => c.request_id !== requestId),
        }));
      },

      getCases: () => {
        return get().cases;
      },

      getCount: () => {
        return get().cases.length;
      },
    }),
    {
      name: "demo-review-queue-storage",
    }
  )
);
