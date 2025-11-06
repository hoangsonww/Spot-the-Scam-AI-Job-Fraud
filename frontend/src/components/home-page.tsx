"use client"

import { useCallback, useId, useMemo, useState } from "react"
import useSWR from "swr"
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { Skeleton } from "@/components/ui/skeleton"
import TopNav from "@/components/top-nav"
import {
  fetchMetadata,
  fetchTokenFrequency,
  fetchTokenImportance,
  fetchThresholdMetrics,
  fetchLatencySummary,
  fetchSliceMetrics,
  fetchModelSummaries,
  getApiBaseUrl,
  predictSingle,
  type JobPostingInput,
  type MetadataResponse,
  type ModelsResponse,
  type PredictionResponse,
  type TokenFrequencyResponse,
  type TokenImportanceResponse,
  type ThresholdMetricsResponse,
  type LatencySummaryResponse,
  type FeatureContribution,
  type SliceMetricsResponse,
} from "@/lib/api"
import {
  Activity,
  AlertTriangle,
  AlignVerticalSpaceBetween,
  ArrowRight,
  BarChart4,
  Brain,
  Flame,
  LineChart as LineChartIcon,
  ShieldCheck,
  Target,
} from "lucide-react"

type MetricKey = keyof NonNullable<MetadataResponse["val_metrics"]>

type BadgeTone = "default" | "destructive" | "secondary" | "outline"

type ToggleField = "telecommuting" | "has_company_logo" | "has_questions"

type FormFields = Pick<
  JobPostingInput,
  | "title"
  | "company_profile"
  | "description"
  | "requirements"
  | "benefits"
  | "location"
  | "employment_type"
  | "required_experience"
  | "required_education"
  | "industry"
  | "function"
> & {
  telecommuting: boolean
  has_company_logo: boolean
  has_questions: boolean
}

const samplePosting: FormFields = {
  title: "Remote Accounts Payable Specialist (Immediate Start)",
  location: "Remote",
  employment_type: "Contract",
  required_experience: "2+ years",
  required_education: "Associate Degree",
  industry: "Accounting",
  function: "Finance",
  company_profile:
    "Nimbus Finance is a regional outsourcing firm supporting mid-market clients with day-to-day operations.",
  description:
    "We are expanding our remote accounting pod to support new enterprise contracts. You will process invoices, maintain vendor ledgers, and support monthly close for North America clients.",
  requirements:
    "2+ years of accounts payable experience • Familiar with QuickBooks or NetSuite • Comfortable working remote with verified identity • Must pass background screening at hire.",
  benefits:
    "Competitive hourly pay • 401(k) match after 90 days • Remote stipend • Bonus eligibility based on accuracy and throughput.",
  telecommuting: true,
  has_company_logo: true,
  has_questions: false,
}

export const metricLabels: Record<MetricKey, string> = {
  f1: "F1",
  precision: "Precision",
  recall: "Recall",
  roc_auc: "ROC AUC",
  pr_auc: "PR AUC",
  brier: "Brier",
}

export function formatMetric(value?: number | null, options?: Intl.NumberFormatOptions) {
  if (value === undefined || value === null) {
    return "-"
  }

  const formatter = new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 3,
    minimumFractionDigits: options?.style === "percent" ? 1 : 2,
    ...options,
  })
  return formatter.format(value)
}

export function formatContribution(value?: number | null) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "0.000"
  }
  const formatter = new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 3,
    maximumFractionDigits: 3,
  })
  const formatted = formatter.format(value)
  return value >= 0 ? `+${formatted}` : formatted
}

function normalizeTextField(value?: string | null): string | null {
  if (value === undefined || value === null) {
    return null
  }
  const trimmed = value.trim()
  return trimmed.length ? trimmed : null
}

function metricToPercent(value?: number | null) {
  if (value === undefined || value === null) {
    return 0
  }
  return Math.min(100, Math.max(0, Math.round(value * 100)))
}

function formatTimestamp(value?: string | null) {
  if (!value) {
    return "-"
  }
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) {
    return "-"
  }
  return parsed.toLocaleString()
}

export default function HomePage() {
  const [form, setForm] = useState<FormFields>(samplePosting)
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const apiBaseUrl = useMemo(() => getApiBaseUrl(), [])

  const {
    data: metadata,
    error: metadataError,
    isLoading: isLoadingMetadata,
  } = useSWR<MetadataResponse>("metadata", fetchMetadata, {
    revalidateOnFocus: false,
  })

  const {
    data: modelsResponse,
    error: modelsError,
    isLoading: isLoadingModels,
  } = useSWR<ModelsResponse>(
    "models",
    () => fetchModelSummaries(50),
    { revalidateOnFocus: false }
  )

  const {
    data: tokenImportance,
    isLoading: isLoadingImportance,
  } = useSWR<TokenImportanceResponse>(
    "token-importance",
    () => fetchTokenImportance(12),
    { revalidateOnFocus: false }
  )

  const {
    data: tokenFrequency,
    isLoading: isLoadingFrequency,
  } = useSWR<TokenFrequencyResponse>(
    "token-frequency",
    () => fetchTokenFrequency(12),
    { revalidateOnFocus: false }
  )

  const {
    data: thresholdMetrics,
    isLoading: isLoadingThresholds,
  } = useSWR<ThresholdMetricsResponse>(
    "threshold-metrics",
    () => fetchThresholdMetrics(40),
    { revalidateOnFocus: false }
  )

  const {
    data: latencySummary,
    isLoading: isLoadingLatency,
  } = useSWR<LatencySummaryResponse>(
    "latency-summary",
    () => fetchLatencySummary(),
    { revalidateOnFocus: false }
  )

  const {
    data: sliceMetrics,
    error: sliceMetricsError,
    isLoading: isLoadingSliceMetrics,
  } = useSWR<SliceMetricsResponse>(
    "slice-metrics",
    () => fetchSliceMetrics(6),
    { revalidateOnFocus: false }
  )

  const handleChange = useCallback(
    (field: keyof FormFields) =>
      (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        const value = event.target.value
        setForm((prev) => ({ ...prev, [field]: value }))
      },
    []
  )

  const handleToggleChange = useCallback(
    (field: ToggleField) => (event: React.ChangeEvent<HTMLInputElement>) => {
      setForm((prev) => ({ ...prev, [field]: event.target.checked }))
    },
    []
  )

  const resetToSample = useCallback(() => {
    setForm(() => ({ ...samplePosting }))
    setPrediction(null)
    setError(null)
  }, [])

  const handlePredict = useCallback(
    async (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault()
      setIsSubmitting(true)
      setError(null)

      const payload: JobPostingInput = {
        title: form.title.trim(),
        location: normalizeTextField(form.location),
        company_profile: normalizeTextField(form.company_profile),
        description: normalizeTextField(form.description),
        requirements: normalizeTextField(form.requirements),
        benefits: normalizeTextField(form.benefits),
        employment_type: normalizeTextField(form.employment_type),
        required_experience: normalizeTextField(form.required_experience),
        required_education: normalizeTextField(form.required_education),
        industry: normalizeTextField(form.industry),
        function: normalizeTextField(form.function),
        telecommuting: form.telecommuting ? 1 : 0,
        has_company_logo: form.has_company_logo ? 1 : 0,
        has_questions: form.has_questions ? 1 : 0,
      }

      try {
        const result = await predictSingle(payload)
        setPrediction(result)
      } catch (predictError) {
        const message =
          predictError instanceof Error
            ? predictError.message
            : "Unable to score the job posting."
        setError(message)
        setPrediction(null)
      } finally {
        setIsSubmitting(false)
      }
    },
    [form]
  )

  const metrics = useMemo(() => {
    if (!metadata) {
      return []
    }
    const entries: {
      label: string
      key: MetricKey
      validation?: number | null
      testing?: number | null
    }[] = []

    ;(Object.keys(metricLabels) as MetricKey[]).forEach((metric) => {
      const validation = metadata.val_metrics?.[metric]
      const testing = metadata.test_metrics?.[metric]
      if (validation !== undefined || testing !== undefined) {
        entries.push({
          label: metricLabels[metric],
          key: metric,
          validation,
          testing,
        })
      }
    })

    return entries
  }, [metadata])

  const modelSummaries = useMemo(() => modelsResponse?.items ?? [], [modelsResponse])

  const thresholdSeries = useMemo(() => {
    if (!thresholdMetrics?.points?.length) {
      return []
    }
    return thresholdMetrics.points.map((point) => ({
      x: point.threshold,
      y: point.f1,
    }))
  }, [thresholdMetrics])

  const latencyBars = useMemo(() => {
    if (!latencySummary?.items?.length) {
      return []
    }

    return latencySummary.items.map((item) => ({
      batchSize: item.batch_size,
      p50: item.latency_p50_ms,
      p95: item.latency_p95_ms,
      throughput: item.throughput_rps,
    }))
  }, [latencySummary])

  const grayZoneDetails = useMemo(() => {
    if (!metadata) {
      return null
    }

    const policy = metadata.gray_zone
    return [
      { label: "Width", value: formatMetric(policy.width, { maximumFractionDigits: 2 }) },
      { label: "Lower bound", value: formatMetric(policy.lower, { maximumFractionDigits: 2 }) },
      { label: "Upper bound", value: formatMetric(policy.upper, { maximumFractionDigits: 2 }) },
      { label: "Labels", value: `${policy.negative_label} → ${policy.review_label} → ${policy.positive_label}` },
    ]
  }, [metadata])

  const predictionBadge = useMemo((): { tone: BadgeTone; label: string } => {
    if (!prediction) {
      return { tone: "outline", label: "Awaiting submission" }
    }

    const normalizedDecision = prediction.decision.toLowerCase()
    let tone: BadgeTone

    if (normalizedDecision === "fraud") {
      tone = "destructive"
    } else if (normalizedDecision === "review") {
      tone = "secondary"
    } else {
      tone = "default"
    }

    return { tone, label: prediction.decision }
  }, [prediction])

  return (
    <div className="min-h-screen bg-gradient-to-b from-background via-background/90 to-background">
      <TopNav />
      <main className="mx-auto flex w-full max-w-6xl flex-col gap-10 px-4 py-12 sm:px-8">
        <header className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <Badge variant="secondary" className="w-fit">
              <AlignVerticalSpaceBetween className="size-4" />
              Model Risk Dashboard
            </Badge>
            <h1 className="text-foreground text-3xl font-semibold tracking-tight sm:text-4xl">
              Spot questionable job postings before they hit applicants.
            </h1>
            <p className="text-muted-foreground max-w-3xl text-base">
              Score listings using the calibrated pipeline, inspect the features that drive
              decisions, and keep an eye on model health metrics - all in one vertical flow.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3 text-sm text-muted-foreground">
            <span className="inline-flex items-center gap-1 rounded-full border border-border/60 bg-card px-3 py-1">
              <ShieldCheck className="size-4 text-primary" />
              Serving threshold:{" "}
              <strong className="text-foreground">
                {metadata ? formatMetric(metadata.threshold, { maximumFractionDigits: 3 }) : "…"}
              </strong>
            </span>
            <span className="inline-flex items-center gap-1 rounded-full border border-border/60 bg-card px-3 py-1">
              <Activity className="size-4 text-chart-2" />
              API: <code className="font-mono text-xs">{apiBaseUrl}</code>
            </span>
            {metadata?.calibration_method ? (
              <span className="inline-flex items-center gap-1 rounded-full border border-border/60 bg-card px-3 py-1">
                <Brain className="size-4 text-chart-3" />
                Calibration: {metadata.calibration_method}
              </span>
            ) : null}
          </div>
        </header>

        <section className="grid gap-6 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
          <div className="flex flex-col gap-6">
            <Card className="backdrop-blur">
              <CardHeader>
                <CardTitle className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
                  <span>Score a job posting</span>
                  <Badge variant={predictionBadge.tone} className="uppercase tracking-wide">
                    {predictionBadge.label}
                  </Badge>
                </CardTitle>
                <CardDescription>
                  Paste details from a listing. We keep the text local until you submit for scoring.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form className="flex flex-col gap-4" onSubmit={handlePredict}>
                  <div className="flex flex-col gap-2">
                    <Label htmlFor="title">Job title *</Label>
                    <Input
                      id="title"
                      required
                      value={form.title}
                      onChange={handleChange("title")}
                      placeholder="Fraud Prevention Specialist"
                    />
                  </div>
                  <Tabs defaultValue="description" className="w-full">
                    <TabsList>
                      <TabsTrigger value="description">Description</TabsTrigger>
                      <TabsTrigger value="requirements">Requirements</TabsTrigger>
                      <TabsTrigger value="company">Company profile</TabsTrigger>
                      <TabsTrigger value="benefits">Benefits</TabsTrigger>
                    </TabsList>
                    <TabsContent value="description" className="mt-2 flex flex-col gap-2">
                      <Label htmlFor="description">Role overview</Label>
                      <Textarea
                        id="description"
                        rows={8}
                        value={form.description ?? ""}
                        onChange={handleChange("description")}
                        placeholder="Enter the main body of the job posting..."
                      />
                    </TabsContent>
                    <TabsContent value="requirements" className="mt-2 flex flex-col gap-2">
                      <Label htmlFor="requirements">Requirements</Label>
                      <Textarea
                        id="requirements"
                        rows={6}
                        value={form.requirements ?? ""}
                        onChange={handleChange("requirements")}
                        placeholder="Experience expectations, background checks, etc."
                      />
                    </TabsContent>
                    <TabsContent value="company" className="mt-2 flex flex-col gap-2">
                      <Label htmlFor="company_profile">Company profile</Label>
                      <Textarea
                        id="company_profile"
                        rows={5}
                        value={form.company_profile ?? ""}
                        onChange={handleChange("company_profile")}
                        placeholder="How the company presents itself in the listing."
                      />
                    </TabsContent>
                    <TabsContent value="benefits" className="mt-2 flex flex-col gap-2">
                      <Label htmlFor="benefits">Benefits</Label>
                      <Textarea
                        id="benefits"
                        rows={4}
                        value={form.benefits ?? ""}
                        onChange={handleChange("benefits")}
                        placeholder="Perks, compensation, or promises made in the posting."
                      />
                    </TabsContent>
                  </Tabs>
                  <Separator className="my-1" />
                  <div className="grid gap-4 sm:grid-cols-2">
                    <div className="flex flex-col gap-2">
                      <Label htmlFor="location">Location</Label>
                      <Input
                        id="location"
                        value={form.location ?? ""}
                        onChange={handleChange("location")}
                        placeholder="Remote · Austin, TX · Hybrid"
                      />
                    </div>
                    <div className="flex flex-col gap-2">
                      <Label htmlFor="employment_type">Employment type</Label>
                      <Input
                        id="employment_type"
                        value={form.employment_type ?? ""}
                        onChange={handleChange("employment_type")}
                        placeholder="Full-time, Contract, Internship…"
                      />
                    </div>
                    <div className="flex flex-col gap-2">
                      <Label htmlFor="required_experience">Required experience</Label>
                      <Input
                        id="required_experience"
                        value={form.required_experience ?? ""}
                        onChange={handleChange("required_experience")}
                        placeholder="e.g., 2+ years, Entry level"
                      />
                    </div>
                    <div className="flex flex-col gap-2">
                      <Label htmlFor="required_education">Required education</Label>
                      <Input
                        id="required_education"
                        value={form.required_education ?? ""}
                        onChange={handleChange("required_education")}
                        placeholder="e.g., Bachelor's, High School Diploma"
                      />
                    </div>
                    <div className="flex flex-col gap-2">
                      <Label htmlFor="industry">Industry</Label>
                      <Input
                        id="industry"
                        value={form.industry ?? ""}
                        onChange={handleChange("industry")}
                        placeholder="Industry context (Finance, Healthcare…)"
                      />
                    </div>
                    <div className="flex flex-col gap-2">
                      <Label htmlFor="function">Function</Label>
                      <Input
                        id="function"
                        value={form.function ?? ""}
                        onChange={handleChange("function")}
                        placeholder="Functional area (Sales, Engineering…)"
                      />
                    </div>
                  </div>
                  <div className="rounded-lg border border-border/60 bg-muted/10 p-4">
                    <div className="mb-3 flex flex-col gap-1">
                      <span className="text-sm font-medium text-foreground">Metadata flags</span>
                      <span className="text-xs text-muted-foreground">
                        These booleans align with the training data and directly influence the model&apos;s tabular features.
                      </span>
                    </div>
                    <div className="grid gap-3 sm:grid-cols-3">
                      <label className="flex items-start gap-3 rounded-md border border-border/60 bg-background px-3 py-2 text-sm">
                        <input
                          type="checkbox"
                          className="mt-1 size-4 accent-primary"
                          checked={form.has_company_logo}
                          onChange={handleToggleChange("has_company_logo")}
                        />
                        <span className="flex flex-col gap-0.5">
                          <span className="font-medium text-foreground">Displays company logo</span>
                          <span className="text-xs text-muted-foreground">
                            Mark when the listing shows an authentic employer logo.
                          </span>
                        </span>
                      </label>
                      <label className="flex items-start gap-3 rounded-md border border-border/60 bg-background px-3 py-2 text-sm">
                        <input
                          type="checkbox"
                          className="mt-1 size-4 accent-primary"
                          checked={form.telecommuting}
                          onChange={handleToggleChange("telecommuting")}
                        />
                        <span className="flex flex-col gap-0.5">
                          <span className="font-medium text-foreground">Telecommuting / remote</span>
                          <span className="text-xs text-muted-foreground">
                            Set when the job can be performed remotely.
                          </span>
                        </span>
                      </label>
                      <label className="flex items-start gap-3 rounded-md border border-border/60 bg-background px-3 py-2 text-sm">
                        <input
                          type="checkbox"
                          className="mt-1 size-4 accent-primary"
                          checked={form.has_questions}
                          onChange={handleToggleChange("has_questions")}
                        />
                        <span className="flex flex-col gap-0.5">
                          <span className="font-medium text-foreground">Screening questions included</span>
                          <span className="text-xs text-muted-foreground">
                            Enable when applicants must answer custom questions.
                          </span>
                        </span>
                      </label>
                    </div>
                  </div>
                  {error ? (
                    <Alert variant="destructive" className="mt-2">
                      <AlertTriangle className="size-4" />
                      <AlertTitle>Scoring failed</AlertTitle>
                      <AlertDescription>{error}</AlertDescription>
                    </Alert>
                  ) : null}
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between sm:gap-2">
                    <div className="text-muted-foreground text-xs">
                      Fields marked * are required. We automatically apply your gray-zone policy.
                    </div>
                    <div className="flex gap-2">
                      <Button
                        type="button"
                        variant="ghost"
                        onClick={resetToSample}
                        disabled={isSubmitting}
                      >
                        Load sample
                      </Button>
                      <Button type="submit" disabled={isSubmitting}>
                        Score posting
                        <ArrowRight className="size-4" />
                      </Button>
                    </div>
                  </div>
                </form>
              </CardContent>
              <CardFooter className="flex flex-col gap-4 border-t border-border/60 pt-6">
                <div className="grid gap-3 sm:grid-cols-3">
                  <MetricCallout
                    label="Fraud probability"
                    value={
                      prediction
                        ? formatMetric(prediction.probability_fraud, {
                            style: "percent",
                            maximumFractionDigits: 1,
                          })
                        : "-"
                    }
                    accent={prediction ? "text-chart-1" : undefined}
                  />
                  <MetricCallout
                    label="Binary label"
                    value={
                      prediction
                        ? prediction.binary_label === 1
                          ? "Fraud"
                          : "Legit"
                        : "Pending"
                    }
                    accent={
                      prediction
                        ? prediction.binary_label === 1
                          ? "text-destructive"
                          : "text-chart-2"
                        : undefined
                    }
                  />
                  <MetricCallout
                    label="Threshold applied"
                    value={
                      prediction
                        ? formatMetric(prediction.threshold, { maximumFractionDigits: 3 })
                        : metadata
                          ? formatMetric(metadata.threshold, { maximumFractionDigits: 3 })
                          : "-"
                    }
                    accent={prediction ? "text-chart-3" : undefined}
                  />
                </div>
                {prediction ? (
                  <div className="text-muted-foreground text-xs">
                    Gray-zone band:{" "}
                    <span className="font-medium text-foreground">
                      {formatMetric(prediction.gray_zone.lower, { maximumFractionDigits: 3 })} -{" "}
                      {formatMetric(prediction.gray_zone.upper, { maximumFractionDigits: 3 })}
                    </span>{" "}
                    ({prediction.gray_zone.negative_label} → {prediction.gray_zone.review_label} →{" "}
                    {prediction.gray_zone.positive_label})
                  </div>
                ) : (
                  <div className="grid gap-2 text-xs text-muted-foreground">
                    <div>
                      Gray-zone band:{" "}
                      <span className="font-medium text-foreground">
                        {metadata
                          ? `${formatMetric(metadata.gray_zone.lower, { maximumFractionDigits: 3 })} - ${formatMetric(metadata.gray_zone.upper, { maximumFractionDigits: 3 })}`
                          : "-"}
                      </span>
                    </div>
                    <ul className="grid gap-1 pl-4 marker:text-primary list-disc">
                      <li>Text goes through the same TF-IDF + tabular pipeline used during training.</li>
                      <li>
                        Calibrated probabilities are compared to the current threshold and gray-zone policy.
                      </li>
                      <li>
                        Predictions return decision labels along with metadata so you can log or review them.
                      </li>
                    </ul>
                  </div>
                )}
            </CardFooter>
          </Card>

            {prediction ? (
              <Card>
                <CardHeader className="pb-4">
                  <CardTitle className="flex items-center gap-2">
                    <AlignVerticalSpaceBetween className="size-5 text-chart-3" />
                    Decision rationale
                  </CardTitle>
                  <CardDescription>
                    Highlights of what pushed this posting toward or away from fraud.
                  </CardDescription>
                </CardHeader>
                <CardContent className="flex flex-col gap-4">
                  <p className="text-sm text-muted-foreground">
                    {prediction.explanation.summary ??
                      "Positive contributions increase the fraud score, while negative ones back the legit decision."}
                  </p>
                  <div className="grid gap-4 sm:grid-cols-2">
                    <ContributionColumn
                      title="Signals toward fraud"
                      emptyLabel="No strong fraud drivers detected."
                      items={prediction.explanation.top_positive}
                    />
                    <ContributionColumn
                      title="Signals toward legit"
                      emptyLabel="No strong legit counter signals."
                      items={prediction.explanation.top_negative}
                      direction="negative"
                    />
                  </div>
                  {typeof prediction.explanation.intercept === "number" ? (
                    <div className="text-xs text-muted-foreground">
                      Model intercept: {formatContribution(prediction.explanation.intercept)}
                    </div>
                  ) : null}
                </CardContent>
              </Card>
            ) : null}

            <Card>
              <CardHeader className="pb-4">
                <CardTitle className="flex items-center gap-2">
                  <LineChartIcon className="size-5 text-chart-1" />
                  Performance trends
                </CardTitle>
                <CardDescription>
                  Threshold tuning and latency envelopes from the latest training benchmark.
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-8">
                <div className="flex flex-col gap-3">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-sm font-semibold text-foreground">
                      F1 vs. threshold
                    </span>
                    <span className="text-xs text-muted-foreground">
                      Latest F1:{" "}
                      <strong className="text-foreground">
                        {thresholdSeries.length
                          ? formatMetric(thresholdSeries[thresholdSeries.length - 1].y, {
                              maximumFractionDigits: 3,
                            })
                          : "-"}
                      </strong>
                    </span>
                  </div>
                  {isLoadingThresholds ? (
                    <Skeleton className="h-36 w-full" />
                  ) : thresholdSeries.length ? (
                    <LineChart
                      points={thresholdSeries}
                      color="var(--chart-1)"
                      minY={0}
                      maxY={1}
                      formatX={(value) =>
                        formatMetric(value, { maximumFractionDigits: 2 })
                      }
                      formatY={(value) => formatMetric(value, { maximumFractionDigits: 2 })}
                    />
                  ) : (
                    <Alert>
                      <AlertTitle>No threshold sweep found</AlertTitle>
                      <AlertDescription>
                        Run the training pipeline to regenerate threshold metrics and plots.
                      </AlertDescription>
                    </Alert>
                  )}
                </div>

                <Separator />

                <div className="flex flex-col gap-3">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-sm font-semibold text-foreground">
                      Latency envelopes
                    </span>
                    <span className="text-xs text-muted-foreground">
                      Throughput @ batch 32:{" "}
                      <strong className="text-foreground">
                        {latencyBars.length
                          ? formatMetric(
                              latencyBars.find((bar) => bar.batchSize === 32)?.throughput,
                              { maximumFractionDigits: 0, minimumFractionDigits: 0 }
                            )
                          : "-"}{" "}
                        rps
                      </strong>
                    </span>
                  </div>
                  {isLoadingLatency ? (
                    <Skeleton className="h-36 w-full" />
                  ) : latencyBars.length ? (
                    <LatencyChart bars={latencyBars} />
                  ) : (
                    <Alert>
                      <AlertTitle>No latency benchmark</AlertTitle>
                      <AlertDescription>
                        Execute the benchmark suite to populate latency statistics.
                      </AlertDescription>
                    </Alert>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="flex flex-col gap-6">
            <Card>
              <CardHeader className="pb-4">
                <CardTitle className="flex items-center gap-2">
                  <BarChart4 className="size-5 text-primary" />
                  Model snapshot
                </CardTitle>
                <CardDescription>
                  Validation and held-out test scores from the most recent training run.
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-4">
                {isLoadingMetadata ? (
                  <div className="grid gap-3">
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-11/12" />
                    <Skeleton className="h-4 w-10/12" />
                  </div>
                ) : metadataError ? (
                  <Alert variant="destructive">
                    <AlertTriangle className="size-4" />
                    <AlertTitle>Metadata unavailable</AlertTitle>
                    <AlertDescription>
                      {metadataError instanceof Error
                        ? metadataError.message
                        : "Failed to load model metrics."}
                    </AlertDescription>
                  </Alert>
                ) : metadata ? (
                  <>
                    <div className="flex flex-col gap-1">
                      <h2 className="text-lg font-semibold">{metadata.model_name}</h2>
                      <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                        <span className="inline-flex items-center gap-1 rounded-full border border-border/60 px-2 py-1">
                          <Flame className="size-4 text-chart-4" />
                          {metadata.model_type} · {metadata.feature_type}
                        </span>
                        <span className="inline-flex items-center gap-1 rounded-full border border-border/60 px-2 py-1">
                          <Activity className="size-4 text-chart-2" />
                          Test ECE: {formatMetric(metadata.test_ece, { maximumFractionDigits: 3 })}
                        </span>
                      </div>
                    </div>
                    <div className="grid gap-3">
                      {metrics.map((metric) => (
                        <div key={metric.key} className="flex flex-col gap-1.5">
                          <div className="flex items-end justify-between text-xs uppercase text-muted-foreground">
                            <span>{metric.label}</span>
                            <span>
                              Val:{" "}
                              <strong className="text-foreground">
                                {formatMetric(metric.validation)}
                              </strong>{" "}
                              · Test:{" "}
                              <strong className="text-foreground">
                                {formatMetric(metric.testing)}
                              </strong>
                            </span>
                          </div>
                          <div className="relative h-2 rounded-full bg-muted">
                            <div
                              className="absolute inset-y-0 left-0 rounded-full bg-primary/80 transition-all"
                              style={{ width: `${metricToPercent(metric.testing) || 0}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                ) : null}
              </CardContent>
              {grayZoneDetails ? (
                <>
                  <Separator className="mt-2" />
                  <CardFooter className="flex flex-col gap-2 pt-4">
                    <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                      Gray-zone policy
                    </div>
                    <dl className="grid gap-1 text-sm">
                      {grayZoneDetails.map((item) => (
                        <div
                          key={item.label}
                          className="flex items-center justify-between gap-2 text-muted-foreground"
                        >
                          <dt>{item.label}</dt>
                          <dd className="font-medium text-foreground">{item.value}</dd>
                        </div>
                      ))}
                    </dl>
                  </CardFooter>
                </>
              ) : null}
            </Card>

            <Card>
              <CardHeader className="pb-4">
                <CardTitle className="flex items-center gap-2">
                  <AlignVerticalSpaceBetween className="size-5 text-chart-1" />
                  Model leaderboard
                </CardTitle>
                <CardDescription>
                  Recent training runs captured in the lightweight tracker. Sorted by test F1.
                </CardDescription>
              </CardHeader>
              <CardContent>
                {isLoadingModels ? (
                  <div className="grid gap-2">
                    <Skeleton className="h-5 w-full" />
                    <Skeleton className="h-5 w-11/12" />
                    <Skeleton className="h-5 w-10/12" />
                  </div>
                ) : modelsError ? (
                  <Alert variant="destructive">
                    <AlertTriangle className="size-4" />
                    <AlertTitle>Model leaderboard unavailable</AlertTitle>
                    <AlertDescription>
                      {modelsError instanceof Error
                        ? modelsError.message
                        : "Failed to load model records."}
                    </AlertDescription>
                  </Alert>
                ) : modelSummaries.length ? (
                  <div className="overflow-x-auto">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Model</TableHead>
                          <TableHead className="text-right">Val F1</TableHead>
                          <TableHead className="text-right">Test F1</TableHead>
                          <TableHead className="text-right">Test Precision</TableHead>
                          <TableHead className="text-right">Test Recall</TableHead>
                          <TableHead className="text-right">Threshold</TableHead>
                          <TableHead>Updated</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {modelSummaries.map((summary, index) => (
                          <TableRow
                            key={`${summary.model_name}-${summary.timestamp ?? index}`}
                            className={
                              metadata?.model_name === summary.model_name
                                ? "bg-muted/60"
                                : undefined
                            }
                          >
                            <TableCell>
                              <div className="flex flex-col gap-1">
                                <div className="flex items-center gap-2">
                                  <span className="font-medium text-foreground">
                                    {summary.model_name}
                                  </span>
                                  {metadata?.model_name === summary.model_name ? (
                                    <Badge variant="secondary" className="text-[10px] uppercase">
                                      Serving
                                    </Badge>
                                  ) : null}
                                </div>
                                <span className="text-xs text-muted-foreground">
                                  {summary.model_type}
                                  {summary.calibration_method
                                    ? ` · ${summary.calibration_method}`
                                    : ""}
                                </span>
                              </div>
                            </TableCell>
                            <TableCell className="text-right text-sm font-medium">
                              {formatMetric(summary.validation?.f1)}
                            </TableCell>
                            <TableCell className="text-right text-sm font-medium">
                              {formatMetric(summary.test?.f1)}
                            </TableCell>
                            <TableCell className="text-right text-xs text-muted-foreground">
                              {formatMetric(summary.test?.precision)}
                            </TableCell>
                            <TableCell className="text-right text-xs text-muted-foreground">
                              {formatMetric(summary.test?.recall)}
                            </TableCell>
                            <TableCell className="text-right text-xs text-muted-foreground">
                              {summary.threshold !== undefined && summary.threshold !== null
                                ? formatMetric(summary.threshold, { maximumFractionDigits: 3 })
                                : "-"}
                            </TableCell>
                            <TableCell className="text-xs text-muted-foreground">
                              {formatTimestamp(summary.timestamp)}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                ) : (
                  <div className="text-sm text-muted-foreground">
                    No tracked model runs yet. Train a model to populate this leaderboard.
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-4">
                <CardTitle className="flex items-center gap-2">
                  <Brain className="size-5 text-chart-5" />
                  Feature signals
                </CardTitle>
                <CardDescription>
                  Inspect how tokens influence the calibration stack across fraud (positive) and
                  legitimate (negative) labels.
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-4">
                <Tabs defaultValue="importance">
                  <TabsList>
                    <TabsTrigger value="importance">Token importance</TabsTrigger>
                    <TabsTrigger value="frequency">Token frequency</TabsTrigger>
                  </TabsList>
                  <TabsContent value="importance" className="mt-3">
                    {isLoadingImportance ? (
                      <Skeleton className="h-36 w-full" />
                    ) : tokenImportance ? (
                      <div className="grid gap-4 lg:grid-cols-2">
                        <TokenList
                          title="Fraud-leaning tokens"
                          tone="text-destructive"
                          items={tokenImportance.positive}
                        />
                        <TokenList
                          title="Legit-leaning tokens"
                          tone="text-chart-2"
                          items={tokenImportance.negative}
                        />
                      </div>
                    ) : (
                      <Alert>
                        <AlertTitle>No token importance available</AlertTitle>
                        <AlertDescription>
                          The current artifacts do not expose token weights. Re-run training with
                          reporting enabled.
                        </AlertDescription>
                      </Alert>
                    )}
                  </TabsContent>
                  <TabsContent value="frequency" className="mt-3">
                    {isLoadingFrequency ? (
                      <Skeleton className="h-36 w-full" />
                    ) : tokenFrequency ? (
                      <div className="rounded-xl border">
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>Token</TableHead>
                              <TableHead className="text-right">Fraud count</TableHead>
                              <TableHead className="text-right">Legit count</TableHead>
                              <TableHead className="text-right">Δ</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {tokenFrequency.items.map((item) => (
                              <TableRow key={item.token}>
                                <TableCell className="font-medium">{item.token}</TableCell>
                                <TableCell className="text-right">
                                  {item.positive_count}
                                </TableCell>
                                <TableCell className="text-right">
                                  {item.negative_count}
                                </TableCell>
                                <TableCell
                                  className={`text-right font-semibold ${
                                    item.difference >= 0 ? "text-chart-1" : "text-chart-2"
                                  }`}
                                >
                                  {item.difference >= 0 ? "+" : ""}
                                  {item.difference}
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </div>
                    ) : (
                      <Alert>
                        <AlertTitle>No frequency breakdown</AlertTitle>
                        <AlertDescription>
                          Run the training pipeline with insights enabled to populate frequency
                          slices.
                        </AlertDescription>
                      </Alert>
                    )}
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-4">
                <CardTitle className="flex items-center gap-2">
                  <Target className="size-5 text-chart-4" />
                  Slices to review
                </CardTitle>
                <CardDescription>
                  Lowest F1 slices from the latest evaluation so you can focus manual audits where the
                  model struggles most.
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-4">
                {isLoadingSliceMetrics ? (
                  <div className="grid gap-3">
                    <Skeleton className="h-20 w-full" />
                    <Skeleton className="h-20 w-full" />
                    <Skeleton className="h-20 w-full" />
                  </div>
                ) : sliceMetricsError ? (
                  <Alert variant="destructive">
                    <AlertTriangle className="size-4" />
                    <AlertTitle>Slice metrics unavailable</AlertTitle>
                    <AlertDescription>
                      {sliceMetricsError instanceof Error
                        ? sliceMetricsError.message
                        : "Failed to load slice-level performance."}
                    </AlertDescription>
                  </Alert>
                ) : sliceMetrics && sliceMetrics.items.length ? (
                  <ul className="grid gap-3">
                    {sliceMetrics.items.map((item) => (
                      <li
                        key={`${item.slice}-${item.category}`}
                        className="rounded-xl border border-border/60 bg-card/70 px-4 py-3"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex flex-col">
                            <span className="text-sm font-semibold text-foreground">
                              {item.category && item.category !== "<missing>" ? item.category : "Missing value"}
                            </span>
                            <span className="text-xs uppercase tracking-wide text-muted-foreground">
                              {item.slice}
                            </span>
                          </div>
                          <Badge variant="outline">n={item.count}</Badge>
                        </div>
                        <div className="mt-3 grid gap-2 text-xs text-muted-foreground sm:grid-cols-3">
                          <div className="flex items-center justify-between gap-2">
                            <span>F1</span>
                            <span className="font-medium text-foreground">
                              {formatMetric(item.f1, { maximumFractionDigits: 3 })}
                            </span>
                          </div>
                          <div className="flex items-center justify-between gap-2">
                            <span>Precision</span>
                            <span className="font-medium text-foreground">
                              {formatMetric(item.precision, { maximumFractionDigits: 3 })}
                            </span>
                          </div>
                          <div className="flex items-center justify-between gap-2">
                            <span>Recall</span>
                            <span className="font-medium text-foreground">
                              {formatMetric(item.recall, { maximumFractionDigits: 3 })}
                            </span>
                          </div>
                        </div>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    Slice metrics are not available yet. Re-run the evaluation suite to surface
                    segment-level performance.
                  </p>
                )}
              </CardContent>
            </Card>
          </div>
        </section>
      </main>
    </div>
  )
}

function MetricCallout({
  label,
  value,
  accent,
}: {
  label: string
  value: string
  accent?: string
}) {
  return (
    <div className="rounded-lg border border-border/60 bg-card px-4 py-3 text-sm">
      <div className="text-muted-foreground text-xs uppercase tracking-wide">{label}</div>
      <div className={`text-lg font-semibold ${accent ?? "text-foreground"}`}>{value}</div>
    </div>
  )
}

type TokenWeightItem = TokenImportanceResponse["positive"][number]

function TokenList({
  title,
  tone,
  items,
}: {
  title: string
  tone: string
  items: TokenWeightItem[]
}) {
  return (
    <div className="flex min-w-0 flex-col rounded-xl border border-border/60 bg-card/70 p-4">
      <div className="mb-3 flex items-center justify-between">
        <span className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          {title}
        </span>
        <span className={`inline-flex items-center gap-1 text-xs font-medium ${tone}`}>
          <ArrowRight className="size-4" />
          Influence
        </span>
      </div>
      <ul className="grid gap-2">
        {items.slice(0, 12).map((token) => (
          <li
            key={`${title}-${token.term}`}
            className="flex w-full min-w-0 items-start justify-between gap-3 rounded-lg border border-border/60 bg-background/60 px-3 py-2 text-sm"
          >
            <span className="flex-1 min-w-0 break-words font-medium leading-snug text-foreground">
              {token.term}
            </span>
            <span className={`${tone} shrink-0 text-right font-semibold tabular-nums`}>
              {formatMetric(token.weight, { maximumFractionDigits: 3 })}
            </span>
          </li>
        ))}
        {items.length === 0 ? (
          <li className="text-muted-foreground text-sm">No tokens surfaced.</li>
        ) : null}
      </ul>
    </div>
  )
}

export function ContributionColumn({
  title,
  items,
  direction = "positive",
  emptyLabel,
}: {
  title: string
  items: FeatureContribution[]
  direction?: "positive" | "negative"
  emptyLabel: string
}) {
  if (!items.length) {
    return (
      <div className="flex flex-col gap-2">
        <div className="text-sm font-semibold text-foreground">{title}</div>
        <p className="text-xs text-muted-foreground">{emptyLabel}</p>
      </div>
    )
  }

  return (
    <div className="flex min-w-0 flex-col gap-2">
      <div className="text-sm font-semibold text-foreground">{title}</div>
      <ul className="grid gap-1.5">
        {items.map((item) => (
          <li
            key={`${title}-${item.feature}-${item.source}`}
            className="border-border/70 bg-card/60 flex w-full min-w-0 items-start justify-between gap-3 rounded-lg border px-3 py-2 shadow-sm transition-colors hover:bg-card"
          >
            <div className="flex min-w-0 flex-col">
              <span className="text-sm font-medium text-foreground break-words leading-snug">
                {item.feature}
              </span>
              <span className="text-xs text-muted-foreground capitalize">{item.source}</span>
            </div>
            <span
              className={
                direction === "negative"
                  ? "text-chart-2 shrink-0 text-right text-sm font-semibold tabular-nums"
                  : "text-destructive shrink-0 text-right text-sm font-semibold tabular-nums"
              }
            >
              {formatContribution(item.contribution)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}

type ChartPoint = { x: number; y: number }

function LineChart({
  points,
  color,
  minY,
  maxY,
  formatX,
  formatY,
}: {
  points: ChartPoint[]
  color: string
  minY?: number
  maxY?: number
  formatX?: (value: number) => string
  formatY?: (value: number) => string
}) {
  const gradientId = useId()
  if (!points.length) {
    return null
  }

  const width = 420
  const height = 160
  const padding = 16

  const xs = points.map((point) => point.x)
  const ys = points.map((point) => point.y)

  const minX = Math.min(...xs)
  let maxX = Math.max(...xs)
  let minDataY = Math.min(...ys)
  let maxDataY = Math.max(...ys)

  if (typeof minY === "number") {
    minDataY = Math.min(minDataY, minY)
  }
  if (typeof maxY === "number") {
    maxDataY = Math.max(maxDataY, maxY)
  }

  if (Math.abs(maxX - minX) < 1e-9) {
    maxX = minX + 1
  }
  if (Math.abs(maxDataY - minDataY) < 1e-9) {
    maxDataY = minDataY + 1
  }

  const scaleX = (value: number) =>
    padding + ((value - minX) / (maxX - minX)) * (width - padding * 2)
  const scaleY = (value: number) =>
    height - padding - ((value - minDataY) / (maxDataY - minDataY)) * (height - padding * 2)

  const pathData = points
    .map((point, index) => {
      const x = scaleX(point.x)
      const y = scaleY(point.y)
      return `${index === 0 ? "M" : "L"} ${x} ${y}`
    })
    .join(" ")

  const xFormatter = formatX ?? ((value: number) => value.toFixed(2))
  const yFormatter = formatY ?? ((value: number) => value.toFixed(2))

  return (
    <div className="rounded-xl border border-border/60 bg-gradient-to-b from-background/60 to-background/20 p-4">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="h-40 w-full"
        aria-hidden="true"
        role="img"
      >
        <defs>
          <linearGradient id={gradientId} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.45" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>
        <path
          d={`${pathData} L ${scaleX(points[points.length - 1].x)} ${scaleY(minDataY)} L ${scaleX(points[0].x)} ${scaleY(minDataY)} Z`}
          fill={`url(#${gradientId})`}
          stroke="none"
        />
        <path
          d={pathData}
          fill="none"
          stroke={color}
          strokeWidth={3}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <div className="mt-2 flex justify-between text-xs text-muted-foreground">
        <span>{xFormatter(minX)}</span>
        <span>{xFormatter(maxX)}</span>
      </div>
      <div className="mt-1 flex justify-end text-xs text-muted-foreground">
        <span>
          Top:{" "}
          <strong className="text-foreground">
            {yFormatter(points.reduce((highest, point) => Math.max(highest, point.y), minDataY))}
          </strong>
        </span>
      </div>
    </div>
  )
}

type LatencyBar = {
  batchSize: number
  p50: number
  p95: number
  throughput: number
}

function LatencyChart({ bars }: { bars: LatencyBar[] }) {
  if (!bars.length) {
    return null
  }

  const maxLatency = bars.reduce((max, bar) => Math.max(max, bar.p95), 0) || 1

  return (
    <div className="rounded-xl border border-border/60 bg-gradient-to-b from-background/60 to-background/20 p-6">
      <div className="mb-4 flex flex-wrap items-center gap-4 text-xs text-muted-foreground">
        <span className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-chart-2" />
          p50 latency
        </span>
        <span className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-destructive/70" />
          p95 latency
        </span>
      </div>
      <div className="flex h-48 items-end justify-between gap-4">
        {bars.map((bar) => {
          const p95Height = Math.max(6, (bar.p95 / maxLatency) * 100)
          const p50Height = Math.max(4, (bar.p50 / maxLatency) * 100)
          return (
            <div
              key={bar.batchSize}
              className="flex min-w-[56px] flex-1 flex-col items-center gap-2"
            >
              <div className="relative flex h-32 w-full max-w-[72px] items-end justify-center">
                <div
                  className="absolute bottom-0 w-3/5 max-w-[36px] rounded-md bg-destructive/70"
                  style={{ height: `${p95Height}%` }}
                />
                <div
                  className="absolute bottom-0 w-2/5 max-w-[24px] rounded-md bg-chart-2"
                  style={{ height: `${p50Height}%` }}
                />
              </div>
              <div className="text-xs font-medium text-foreground">×{bar.batchSize}</div>
              <div className="text-[11px] text-muted-foreground">
                {formatMetric(bar.throughput, {
                  maximumFractionDigits: 0,
                  minimumFractionDigits: 0,
                })}{" "}
                rps
              </div>
            </div>
          )
        })}
      </div>
      <div className="mt-3 flex w-full justify-between text-xs text-muted-foreground">
        <span>0 ms</span>
        <span>{formatMetric(maxLatency, { maximumFractionDigits: 0, minimumFractionDigits: 0 })} ms</span>
      </div>
    </div>
  )
}
