"use client";

import { useEffect, useState } from "react";
import useSWR, { mutate as mutateGlobal } from "swr";
import {
  fetchReviewCases,
  submitFeedback,
  type ReviewCase,
} from "@/lib/api";
import TopNav from "@/components/top-nav";
import {
  formatContribution,
  formatMetric,
  ContributionColumn,
} from "@/components/home-page";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertTriangle, CheckCircle2, ChevronLeft, ChevronRight, Loader2 } from "lucide-react";

function deriveHeadline(payload: ReviewCase["payload"]) {
  return payload.title ?? "Untitled job posting";
}

function formatUtcTimestamp(value: string): string {
  if (!value) {
    return "-";
  }
  const hasZone = /[zZ]|[+-]\d\d:\d\d$/.test(value);
  const normalized = hasZone ? value : `${value}Z`;
  const date = new Date(normalized);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

export default function ReviewQueue() {
  const PAGE_SIZE = 5;
  const [selectedNotes, setSelectedNotes] = useState<Record<string, string>>({});
  const [selectedRationale, setSelectedRationale] = useState<Record<string, string>>({});
  const [submittingId, setSubmittingId] = useState<string | null>(null);
  const [toast, setToast] = useState<{ type: "success" | "error"; message: string } | null>(null);
  const [page, setPage] = useState(0);

  const {
    data,
    isLoading,
    mutate,
    error,
  } = useSWR(["review-cases", PAGE_SIZE, page], ([, limit, pageIndex]) =>
    fetchReviewCases(limit as number, "gray-zone", (pageIndex as number) * (limit as number)),
    { keepPreviousData: true }
  );

  const cases = data?.items ?? [];
  const totalPending = data?.total_pending ?? 0;
  const totalPages = totalPending > 0 ? Math.ceil(totalPending / PAGE_SIZE) : 1;
  const currentPage = Math.min(page, totalPages - 1);

  useEffect(() => {
    if (page !== currentPage) {
      setPage(currentPage);
    }
  }, [currentPage, page]);

  const startIndex = totalPending === 0 ? 0 : currentPage * PAGE_SIZE + 1;
  const endIndex = totalPending === 0 ? 0 : Math.min(totalPending, currentPage * PAGE_SIZE + cases.length);

  const goPrev = () => {
    setPage((prev) => Math.max(prev - 1, 0));
  };

  const goNext = () => {
    setPage((prev) => Math.min(prev + 1, totalPages - 1));
  };

  const handleSubmit = async (
    item: ReviewCase,
    reviewerLabel: "fraud" | "legit" | "unsure"
  ) => {
    setSubmittingId(item.request_id);
    setToast(null);
    try {
      await submitFeedback([
        {
          request_id: item.request_id,
          model_version: item.model_version,
          proba: item.probability,
          predicted_label: item.predicted_label as "fraud" | "legit" | "review",
          reviewer_label: reviewerLabel,
          text_hash: item.text_hash,
          features_hash: item.features_hash,
          rationale: selectedRationale[item.request_id] ?? "",
          notes: selectedNotes[item.request_id] ?? "",
        },
      ]);

      const optimistic = {
        total_pending: Math.max(0, totalPending - 1),
        items: cases.filter((c) => c.request_id !== item.request_id),
      };
      mutate(optimistic, { revalidate: true });
      const nextCount = Math.max(0, totalPending - 1);
      mutateGlobal("review-count", nextCount, { revalidate: true });

      setToast({
        type: "success",
        message: "Feedback submitted. Case removed from queue.",
      });
      setSelectedNotes((prev) => {
        const next = { ...prev };
        delete next[item.request_id];
        return next;
      });
      setSelectedRationale((prev) => {
        const next = { ...prev };
        delete next[item.request_id];
        return next;
      });
    } catch (err) {
      setToast({
        type: "error",
        message:
          err instanceof Error
            ? err.message
            : "Failed to submit feedback. Please retry.",
      });
    } finally {
      setSubmittingId(null);
    }
  };

  const emptyState = !isLoading && !cases.length && totalPending === 0;

  return (
    <div className="min-h-screen bg-gradient-to-b from-background via-background/90 to-background">
      <TopNav />
      <main className="mx-auto flex w-full max-w-5xl flex-col gap-8 px-4 py-10 sm:px-8">
        <header className="flex flex-col gap-3">
          <Badge variant="secondary" className="w-fit">
            Human-in-the-loop Queue
          </Badge>
          <h1 className="text-foreground text-3xl font-semibold tracking-tight sm:text-4xl">
            Review uncertain job postings
          </h1>
          <p className="text-muted-foreground max-w-3xl text-base">
            Confirm or override the classifier&apos;s call on the most ambiguous listings.
            Feedback is logged for retraining and calibration monitoring.
          </p>
          <div className="text-sm text-muted-foreground">
            Pending cases:{" "}
            <span className="font-semibold text-foreground">{totalPending}</span>
          </div>
          {toast ? (
            <Alert variant={toast.type === "error" ? "destructive" : "default"}>
              {toast.type === "error" ? <AlertTriangle className="size-4" /> : <CheckCircle2 className="size-4" />}
              <AlertTitle>{toast.type === "error" ? "Submission failed" : "Feedback captured"}</AlertTitle>
              <AlertDescription>{toast.message}</AlertDescription>
            </Alert>
          ) : null}
          {error ? (
            <Alert variant="destructive">
              <AlertTriangle className="size-4" />
              <AlertTitle>Unable to load queue</AlertTitle>
              <AlertDescription>
                {error instanceof Error ? error.message : "An unexpected error occurred while loading review cases."}
              </AlertDescription>
            </Alert>
          ) : null}
        </header>

        {isLoading ? (
          <div className="grid gap-4">
            <Skeleton className="h-48 w-full rounded-xl" />
            <Skeleton className="h-48 w-full rounded-xl" />
            <Skeleton className="h-48 w-full rounded-xl" />
          </div>
        ) : emptyState ? (
          <div className="flex flex-col items-center gap-3 rounded-xl border border-border/60 bg-card/50 px-8 py-16 text-center">
            <CheckCircle2 className="size-10 text-chart-2" />
            <h2 className="text-lg font-semibold text-foreground">Queue is clear</h2>
            <p className="text-sm text-muted-foreground">
              No gray-zone predictions waiting for review right now. Come back after the next batch of predictions.
            </p>
          </div>
        ) : (
          <div className="grid gap-6">
            {cases.map((item) => {
              const headline = deriveHeadline(item.payload);
              const isSubmitting = submittingId === item.request_id;
              const localizedCreatedAt = formatUtcTimestamp(item.created_at);
              return (
                <Card key={item.request_id} className="border border-border/70 bg-card/80">
                  <CardHeader className="gap-2 pb-4">
                    <CardTitle className="flex flex-col gap-1 text-lg font-semibold">
                      <span className="text-sm uppercase tracking-wide text-muted-foreground">
                        request #{item.request_id.slice(0, 8)}
                      </span>
                      {headline}
                    </CardTitle>
                    <CardDescription className="grid gap-1 text-xs sm:grid-cols-3 sm:gap-3">
                      <span>
                        Probability:{" "}
                        <strong className="text-foreground">
                          {formatMetric(item.probability, {
                            style: "percent",
                            maximumFractionDigits: 1,
                          })}
                        </strong>
                      </span>
                      <span>
                        Model decision:{" "}
                        <strong className="capitalize text-foreground">{item.predicted_label}</strong>
                      </span>
                      <span>
                        Model version:{" "}
                        <strong className="text-foreground">{item.model_version}</strong>
                      </span>
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="flex flex-col gap-6">
                    <section className="grid gap-3 text-sm">
                      <h3 className="text-sm font-semibold text-foreground">Listing snapshot</h3>
                      <div className="grid gap-2 text-muted-foreground sm:grid-cols-2">
                        <FieldPresenter label="Location" value={item.payload.location} />
                        <FieldPresenter label="Employment type" value={item.payload.employment_type} />
                      </div>
                      <FieldPresenter label="Description" value={item.payload.description} multiline />
                      <FieldPresenter label="Requirements" value={item.payload.requirements} multiline />
                    </section>

                    <section className="flex flex-col gap-4">
                      <h3 className="text-sm font-semibold text-foreground">Model rationale</h3>
                      <div className="grid gap-4 sm:grid-cols-2">
                        <ContributionColumn
                          title="Signals toward fraud"
                          items={item.explanation.top_positive}
                          emptyLabel="No strong fraud drivers logged."
                        />
                        <ContributionColumn
                          title="Signals toward legit"
                          direction="negative"
                          items={item.explanation.top_negative}
                          emptyLabel="No strong legit counter signals."
                        />
                      </div>
                      {typeof item.explanation.intercept === "number" ? (
                        <p className="text-xs text-muted-foreground">
                          Model intercept: {formatContribution(item.explanation.intercept)}
                        </p>
                      ) : null}
                    </section>

                    <section className="grid gap-3 sm:grid-cols-2">
                      <div className="flex flex-col gap-2">
                        <label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                          Reviewer rationale (optional)
                        </label>
                        <Textarea
                          rows={3}
                          placeholder="Summarize why you agree or disagree with the model."
                          value={selectedRationale[item.request_id] ?? ""}
                          onChange={(event) =>
                            setSelectedRationale((prev) => ({
                              ...prev,
                              [item.request_id]: event.target.value,
                            }))
                          }
                          className="bg-background/60"
                        />
                      </div>
                      <div className="flex flex-col gap-2">
                        <label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                          Private notes (optional)
                        </label>
                        <Textarea
                          rows={3}
                          placeholder="Internal notes, redactions applied automatically."
                          value={selectedNotes[item.request_id] ?? ""}
                          onChange={(event) =>
                            setSelectedNotes((prev) => ({
                              ...prev,
                              [item.request_id]: event.target.value,
                            }))
                          }
                          className="bg-background/60"
                        />
                      </div>
                    </section>
                  </CardContent>
                  <CardFooter className="flex flex-col gap-3 border-t border-border/60 bg-card/40 px-6 py-4 sm:flex-row sm:items-center sm:justify-between">
                    <div className="text-xs text-muted-foreground">
                      Logged at{" "}
                      <span className="font-medium text-foreground">
                        {localizedCreatedAt}
                      </span>
                      . Reviewer overrides feed the next calibration pass.
                    </div>
                    <div className="flex gap-2">
                      <Button
                        variant="ghost"
                        disabled={isSubmitting}
                        onClick={() => handleSubmit(item, "unsure")}
                      >
                        Unsure
                      </Button>
                      <Button
                        variant="outline"
                        className="border-chart-2 text-chart-2 hover:bg-chart-2/10"
                        disabled={isSubmitting}
                        onClick={() => handleSubmit(item, "legit")}
                      >
                        {isSubmitting ? (
                          <Loader2 className="mr-2 size-4 animate-spin" />
                        ) : null}
                        Confirm Legit
                      </Button>
                      <Button
                        className="bg-destructive text-white hover:bg-destructive/90"
                        disabled={isSubmitting}
                        onClick={() => handleSubmit(item, "fraud")}
                      >
                        {isSubmitting ? (
                          <Loader2 className="mr-2 size-4 animate-spin" />
                        ) : null}
                        Confirm Fraud
                      </Button>
                    </div>
                  </CardFooter>
                </Card>
              );
            })}
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border/60 bg-card/60 px-4 py-3 text-xs text-muted-foreground">
              <span>
                {totalPending
                  ? `Showing ${startIndex}-${endIndex} of ${totalPending}`
                  : "No queued cases"}
              </span>
              <div className="inline-flex items-center gap-2">
                <span>
                  Page {totalPending ? currentPage + 1 : 0} of {totalPending ? totalPages : 0}
                </span>
                <div className="flex items-center gap-1">
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={goPrev}
                    disabled={isLoading || currentPage === 0 || totalPending === 0}
                    aria-label="Previous page"
                  >
                    <ChevronLeft className="size-4" />
                  </Button>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={goNext}
                    disabled={
                      isLoading ||
                      totalPending === 0 ||
                      currentPage >= totalPages - 1
                    }
                    aria-label="Next page"
                  >
                    <ChevronRight className="size-4" />
                  </Button>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

function FieldPresenter({
  label,
  value,
  multiline,
}: {
  label: string;
  value?: string | null;
  multiline?: boolean;
}) {
  if (!value) {
    return (
      <div className="rounded-lg border border-dashed border-border/60 bg-card/40 px-3 py-2 text-xs text-muted-foreground">
        <span className="font-semibold">{label}:</span> <span>Not provided</span>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-border/60 bg-card/50 px-3 py-2 text-sm text-foreground">
      <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {label}
      </span>
      <p className={`mt-1 ${multiline ? "whitespace-pre-wrap" : ""}`}>{value}</p>
    </div>
  );
}
