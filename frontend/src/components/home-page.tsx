"use client";

import { useMemo, useState } from "react";
import useSWR from "swr";
import {
  fetchMetadata,
  fetchTokenFrequency,
  fetchTokenImportance,
  getApiBaseUrl,
  JobPostingInput,
  MetadataResponse,
  PredictionResponse,
  predictSingle,
  TokenFrequencyResponse,
  TokenImportanceResponse,
} from "@/lib/api";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { Toggle } from "@/components/ui/toggle";
import {
  AlertTriangle,
  ClipboardList,
  Loader2,
  Sparkles,
  TrendingDown,
  TrendingUp,
} from "lucide-react";

const defaultPosting: JobPostingInput = {
  title: "",
  description: "",
  requirements: "",
  company_profile: "",
  benefits: "",
  employment_type: "",
  required_experience: "",
  required_education: "",
  industry: "",
  function: "",
  telecommuting: 0,
  has_company_logo: 1,
  has_questions: 0,
};

const samplePosting: JobPostingInput = {
  title: "Remote Data Entry Specialist",
  company_profile:
    "Global Processing Partners claims to provide remote clerical support for Fortune 500 clients. Employees are expected to handle sensitive documents and payments.",
  description:
    "We are urgently hiring remote data entry specialists to process client payments. Work from home, full-time hours, competitive weekly pay. Applicants will be required to purchase a laptop from our approved vendor, which will be reimbursed with your first paycheck.",
  requirements:
    "Must be detail oriented and able to work without supervision. Comfortable handling large volumes of transactions. Familiarity with payment platforms and quick to learn company systems.",
  benefits:
    "Weekly payouts via wire transfer. Flexible hours. Career advancement opportunities.",
  employment_type: "Full-time",
  required_experience: "No experience",
  required_education: "Not required",
  industry: "Financial Services",
  function: "Administrative",
  telecommuting: 1,
  has_company_logo: 0,
  has_questions: 0,
};

const metricLabels: Record<keyof Required<MetadataResponse>["val_metrics"], string> = {
  f1: "F1",
  precision: "Precision",
  recall: "Recall",
  roc_auc: "ROC AUC",
  pr_auc: "PR AUC",
  brier: "Brier",
};

function formatPercent(value?: number) {
  if (value === undefined || value === null) return "--";
  return `${(value * 100).toFixed(1)}%`;
}

function MetricsTable({ metadata }: { metadata?: MetadataResponse }) {
  if (!metadata) {
    return (
      <div className="space-y-2">
        {[...Array(3)].map((_, idx) => (
          <Skeleton key={idx} className="h-5 w-full" />
        ))}
      </div>
    );
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Split</TableHead>
          {Object.values(metricLabels).map((label) => (
            <TableHead key={label} className="text-right">
              {label}
            </TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {["val_metrics", "test_metrics"].map((split) => {
          const row = metadata[split as keyof MetadataResponse] as MetadataResponse["val_metrics"];
          const label = split === "val_metrics" ? "Validation" : "Test";
          return (
            <TableRow key={split}>
              <TableCell className="font-medium">{label}</TableCell>
              {Object.keys(metricLabels).map((metricKey) => {
                const value = row?.[metricKey as keyof MetricSet];
                const displayValue = metricKey === "brier" ? value?.toFixed(3) ?? "--" : formatPercent(value);
                return (
                  <TableCell key={metricKey} className="text-right">
                    {displayValue}
                  </TableCell>
                );
              })}
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}

type MetricSet = MetadataResponse["val_metrics"];

function DecisionBadge({ decision }: { decision?: string }) {
  if (!decision) return null;
  const normalized = decision.toLowerCase();
  const variant = normalized === "fraud" ? "destructive" : normalized === "review" ? "outline" : "secondary";
  return <Badge variant={variant}>{decision}</Badge>;
}

function BooleanToggle({
  label,
  description,
  value,
  onChange,
}: {
  label: string;
  description?: string;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <Label className="text-sm font-medium leading-none">{label}</Label>
        {description ? <p className="text-sm text-muted-foreground">{description}</p> : null}
      </div>
      <Toggle
        pressed={value === 1}
        onPressedChange={(pressed) => onChange(pressed ? 1 : 0)}
        aria-label={label}
        className="min-w-[72px]"
      >
        {value === 1 ? "Yes" : "No"}
      </Toggle>
    </div>
  );
}

function TokenTable({ title, items, emptyMessage }: { title: string; items?: { [key: string]: string | number }[]; emptyMessage: string }) {
  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="text-base font-semibold">{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {!items || items.length === 0 ? (
          <p className="text-sm text-muted-foreground">{emptyMessage}</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                {Object.keys(items[0]).map((key) => (
                  <TableHead key={key} className="capitalize">
                    {key.replace(/_/g, " ")}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {items.map((row, idx) => (
                <TableRow key={`${title}-${idx}`}>
                  {Object.values(row).map((value, cellIdx) => (
                    <TableCell key={cellIdx}>{typeof value === "number" ? value.toLocaleString() : value}</TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}

function PredictionSummary({ prediction, metadata }: { prediction?: PredictionResponse | null; metadata?: MetadataResponse }) {
  const band = prediction?.gray_zone;
  const probability = prediction?.probability_fraud ?? 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-xl font-semibold">
          <ShieldIcon decision={prediction?.decision} />
          {prediction ? "Latest Prediction" : "Run the first prediction"}
        </CardTitle>
        <CardDescription>
          {prediction
            ? `This posting is classified as ${prediction.decision.toUpperCase()} with ${(probability * 100).toFixed(1)}% fraud probability.`
            : "Submit a job posting description to receive a calibrated decision."}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {prediction ? (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-3">
              <DecisionBadge decision={prediction.decision} />
              <Badge variant="outline">Probability: {(probability * 100).toFixed(2)}%</Badge>
              <Badge variant="outline">Threshold: {(prediction.threshold * 100).toFixed(2)}%</Badge>
            </div>
            {band ? (
              <div className="rounded-md border bg-muted/30 p-3 text-sm">
                <p className="font-medium">Gray-zone policy</p>
                <p className="text-muted-foreground">
                  Review whenever probability is between {(band.lower * 100).toFixed(1)}% and {(band.upper * 100).toFixed(1)}%. Outside this band the
                  decision is {band.positive_label?.toUpperCase()} / {band.negative_label?.toUpperCase()}.
                </p>
              </div>
            ) : null}
            {metadata?.test_metrics ? (
              <Alert>
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>Calibration quality</AlertTitle>
                <AlertDescription>
                  Test ECE: {metadata?.test_ece !== undefined && metadata?.test_ece !== null ? metadata.test_ece.toFixed(3) : "--"}. Lower is better; consider retraining if this grows.
                </AlertDescription>
              </Alert>
            ) : null}
          </div>
        ) : (
          <div className="space-y-2 text-sm text-muted-foreground">
            <p>
              Provide a job posting title and description. The model will score the posting, apply the calibrated threshold, and surface a gray-zone decision for
              human triage.
            </p>
            <p>Need inspiration? Use the sample posting to see how a suspicious advert is handled.</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ShieldIcon({ decision }: { decision?: string }) {
  if (!decision) {
    return <ClipboardList className="h-5 w-5" />;
  }
  const normalized = decision.toLowerCase();
  if (normalized === "fraud") {
    return <AlertTriangle className="h-5 w-5 text-destructive" />;
  }
  if (normalized === "review") {
    return <Sparkles className="h-5 w-5 text-amber-500" />;
  }
  return <TrendingUp className="h-5 w-5 text-emerald-500" />;
}

export function HomePage() {
  const [form, setForm] = useState<JobPostingInput>(defaultPosting);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const { data: metadata, isLoading: metadataLoading } = useSWR<MetadataResponse>("metadata", fetchMetadata, {
    revalidateOnFocus: false,
  });
  const { data: tokenImportance } = useSWR<TokenImportanceResponse>("token-importance", () => fetchTokenImportance(15), {
    revalidateOnFocus: false,
  });
  const { data: tokenFrequency } = useSWR<TokenFrequencyResponse>("token-frequency", () => fetchTokenFrequency(15), {
    revalidateOnFocus: false,
  });

  const apiBase = useMemo(() => getApiBaseUrl(), []);

  const updateField = <K extends keyof JobPostingInput>(key: K, value: JobPostingInput[K]) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const result = await predictSingle(form);
      setPrediction(result);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setSubmitting(false);
    }
  };

  const resetForm = () => {
    setForm(defaultPosting);
    setPrediction(null);
  };

  const loadSample = () => {
    setForm(samplePosting);
  };

  return (
    <div className="container mx-auto max-w-6xl space-y-6 py-8">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">Spot the Scam Control Panel</h1>
          <p className="text-muted-foreground">
            Submit job postings, inspect calibrated metrics, and explore top warning signals surfaced by the classifier.
          </p>
          <p className="text-xs text-muted-foreground">API base: {apiBase}</p>
        </div>
        <Button variant="outline" onClick={loadSample} className="w-full gap-2 lg:w-auto">
          <Sparkles className="h-4 w-4" />
          Load sample scam posting
        </Button>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle>Score a job posting</CardTitle>
            <CardDescription>Provide key details and the model will return a calibrated fraud probability.</CardDescription>
          </CardHeader>
          <form onSubmit={handleSubmit}>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="title">Job title</Label>
                <Input
                  id="title"
                  value={form.title}
                  onChange={(event) => updateField("title", event.target.value)}
                  required
                  placeholder="e.g. Senior Financial Analyst"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="company_profile">Company overview</Label>
                <Textarea
                  id="company_profile"
                  value={form.company_profile ?? ""}
                  onChange={(event) => updateField("company_profile", event.target.value)}
                  rows={3}
                  placeholder="What does the company say about itself?"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="description">Job description</Label>
                <Textarea
                  id="description"
                  value={form.description ?? ""}
                  onChange={(event) => updateField("description", event.target.value)}
                  required
                  rows={6}
                  placeholder="Core responsibilities, duties, and promises made in the posting."
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="requirements">Requirements</Label>
                <Textarea
                  id="requirements"
                  value={form.requirements ?? ""}
                  onChange={(event) => updateField("requirements", event.target.value)}
                  rows={4}
                  placeholder="Summarize requirements bullet points."
                />
              </div>

              <Separator />

              <Button
                type="button"
                variant="ghost"
                className="w-full justify-start"
                onClick={() => setShowAdvanced((prev) => !prev)}
              >
                {showAdvanced ? "Hide advanced fields" : "Show advanced fields"}
              </Button>

              {showAdvanced ? (
                <div className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Benefits</Label>
                      <Textarea
                        value={form.benefits ?? ""}
                        onChange={(event) => updateField("benefits", event.target.value)}
                        rows={3}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Employment type</Label>
                      <Input
                        value={form.employment_type ?? ""}
                        onChange={(event) => updateField("employment_type", event.target.value)}
                        placeholder="Full-time, contract, ..."
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Required experience</Label>
                      <Input
                        value={form.required_experience ?? ""}
                        onChange={(event) => updateField("required_experience", event.target.value)}
                        placeholder="e.g. 3+ years"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Required education</Label>
                      <Input
                        value={form.required_education ?? ""}
                        onChange={(event) => updateField("required_education", event.target.value)}
                        placeholder="e.g. Bachelor's"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Industry</Label>
                      <Input
                        value={form.industry ?? ""}
                        onChange={(event) => updateField("industry", event.target.value)}
                        placeholder="e.g. Information Technology"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Function</Label>
                      <Input
                        value={form.function ?? ""}
                        onChange={(event) => updateField("function", event.target.value)}
                        placeholder="e.g. Sales"
                      />
                    </div>
                  </div>

                  <Separator />

                  <div className="space-y-3">
                    <BooleanToggle
                      label="Telecommuting allowed"
                      description="Remote or work-from-home role."
                      value={form.telecommuting ?? 0}
                      onChange={(value) => updateField("telecommuting", value)}
                    />
                    <BooleanToggle
                      label="Company logo present"
                      description="Job advert displays a corporate logo."
                      value={form.has_company_logo ?? 0}
                      onChange={(value) => updateField("has_company_logo", value)}
                    />
                    <BooleanToggle
                      label="Has application questions"
                      description="Posting includes screening questions."
                      value={form.has_questions ?? 0}
                      onChange={(value) => updateField("has_questions", value)}
                    />
                  </div>
                </div>
              ) : null}
            </CardContent>
            <CardFooter className="flex flex-col gap-2 sm:flex-row sm:justify-between">
              <Button type="button" variant="outline" onClick={resetForm} className="w-full sm:w-auto">
                Reset
              </Button>
              <Button type="submit" className="w-full sm:w-auto" disabled={submitting}>
                {submitting ? (
                  <span className="flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Scoring...
                  </span>
                ) : (
                  "Score posting"
                )}
              </Button>
            </CardFooter>
          </form>
        </Card>

        <div className="space-y-6">
          <PredictionSummary prediction={prediction} metadata={metadata} />

          <Card>
            <CardHeader>
              <CardTitle>Model snapshot</CardTitle>
              <CardDescription>
                {metadataLoading
                  ? "Loading metrics"
                  : metadata
                  ? `${metadata.model_name} (${metadata.model_type}, ${metadata.feature_type}) calibrated via ${metadata.calibration_method ?? "--"}`
                  : "Start the API to load metadata."}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {metadata ? (
                <>
                  <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
                    <Badge variant="outline">Threshold {(metadata.threshold * 100).toFixed(2)}%</Badge>
                    <Badge variant="outline">Gray-zone ±{(metadata.gray_zone.width * 100 / 2).toFixed(1)}%</Badge>
                  </div>
                  <MetricsTable metadata={metadata} />
                </>
              ) : (
                <Skeleton className="h-24 w-full" />
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Token insights</CardTitle>
              <CardDescription>Top signals learned from the training data.</CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="positive" className="w-full">
                <TabsList className="grid grid-cols-3">
                  <TabsTrigger value="positive" className="flex items-center gap-1">
                    <TrendingUp className="h-4 w-4" />
                    Fraud cues
                  </TabsTrigger>
                  <TabsTrigger value="negative" className="flex items-center gap-1">
                    <TrendingDown className="h-4 w-4" />
                    Legit cues
                  </TabsTrigger>
                  <TabsTrigger value="frequency" className="flex items-center gap-1">
                    <ClipboardList className="h-4 w-4" />
                    Frequency
                  </TabsTrigger>
                </TabsList>
                <TabsContent value="positive">
                  <TokenTable
                    title="Fraud-skewed n-grams"
                    items={tokenImportance?.positive?.map((row) => ({ term: row.term, weight: row.weight.toFixed(4) }))}
                    emptyMessage="Train the classical model to populate token importances."
                  />
                </TabsContent>
                <TabsContent value="negative">
                  <TokenTable
                    title="Legitimate n-grams"
                    items={tokenImportance?.negative?.map((row) => ({ term: row.term, weight: row.weight.toFixed(4) }))}
                    emptyMessage="Train the classical model to populate token importances."
                  />
                </TabsContent>
                <TabsContent value="frequency">
                  <TokenTable
                    title="Token frequency gap"
                    items={tokenFrequency?.items?.map((row) => ({
                      token: row.token,
                      positive_count: row.positive_count,
                      negative_count: row.negative_count,
                      difference: row.difference,
                    }))}
                    emptyMessage="No frequency analysis available."
                  />
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </div>
      </div>

      {error ? (
        <Alert variant="destructive">
          <AlertTitle>Prediction failed</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      ) : null}
    </div>
  );
}

export default HomePage;
