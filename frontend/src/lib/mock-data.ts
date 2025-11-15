import type {
  PredictionResponse,
  PredictionBatchResponse,
  MetadataResponse,
  ModelsResponse,
  TokenImportanceResponse,
  TokenFrequencyResponse,
  ThresholdMetricsResponse,
  LatencySummaryResponse,
  SliceMetricsResponse,
  CasesResponse,
  JobPostingInput,
  ChatStreamChunk,
} from "./api";

// Simulated delay for realistic API behavior
const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

// Generate realistic prediction based on job posting content
function generateRealisticPrediction(input: JobPostingInput): PredictionResponse {
  // Analyze input for fraud indicators
  const fraudIndicators: string[] = [];
  const legitIndicators: string[] = [];

  const text = [input.title, input.description, input.requirements, input.company_profile]
    .join(" ")
    .toLowerCase();

  // Check for fraud signals
  if (text.includes("earn") && text.includes("from home")) fraudIndicators.push("work from home");
  if (text.includes("no experience")) fraudIndicators.push("no experience required");
  if (text.includes("urgent") || text.includes("immediate")) fraudIndicators.push("urgency");
  if (text.includes("$$$") || text.includes("money")) fraudIndicators.push("money focus");
  if (!input.company_profile || input.company_profile.length < 20)
    fraudIndicators.push("minimal company info");
  if (!input.requirements || input.requirements.length < 30)
    fraudIndicators.push("vague requirements");

  // Check for legit signals
  if (input.company_profile && input.company_profile.length > 100)
    legitIndicators.push("detailed company info");
  if (input.requirements && input.requirements.length > 100)
    legitIndicators.push("specific requirements");
  if (input.benefits && input.benefits.length > 50) legitIndicators.push("clear benefits");
  if (input.required_education) legitIndicators.push("education requirements");
  if (input.required_experience) legitIndicators.push("experience requirements");
  if (input.industry) legitIndicators.push("industry specified");

  // Calculate probability based on indicators
  const fraudScore = fraudIndicators.length * 0.15;
  const legitScore = legitIndicators.length * 0.1;
  let probability_fraud = Math.max(0.1, Math.min(0.9, 0.5 + fraudScore - legitScore));

  // Add some randomness for variety
  probability_fraud += (Math.random() - 0.5) * 0.1;
  probability_fraud = Math.max(0.05, Math.min(0.95, probability_fraud));

  const threshold = 0.5;
  const grayZoneWidth = 0.2;
  const grayZoneLower = threshold - grayZoneWidth / 2;
  const grayZoneUpper = threshold + grayZoneWidth / 2;

  let decision: string;
  if (probability_fraud < grayZoneLower) {
    decision = "legit";
  } else if (probability_fraud > grayZoneUpper) {
    decision = "fraud";
  } else {
    decision = "review";
  }

  // Generate feature contributions
  const topPositive = [
    { feature: "no_company_logo", source: "has_company_logo", contribution: 0.23 },
    { feature: "work from home", source: "description", contribution: 0.19 },
    { feature: "urgent hiring", source: "title", contribution: 0.15 },
    { feature: "no experience required", source: "requirements", contribution: 0.12 },
    { feature: "high salary promise", source: "salary_range", contribution: 0.08 },
  ].slice(0, Math.min(5, fraudIndicators.length + 2));

  const topNegative = [
    { feature: "detailed company profile", source: "company_profile", contribution: -0.18 },
    { feature: "specific job requirements", source: "requirements", contribution: -0.14 },
    { feature: "benefits package listed", source: "benefits", contribution: -0.11 },
    { feature: "education requirements", source: "required_education", contribution: -0.09 },
    { feature: "industry specified", source: "industry", contribution: -0.07 },
  ].slice(0, Math.min(5, legitIndicators.length + 2));

  return {
    request_id: `demo-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    probability_fraud,
    binary_label: probability_fraud >= threshold ? 1 : 0,
    decision,
    threshold,
    gray_zone: {
      width: grayZoneWidth,
      lower: grayZoneLower,
      upper: grayZoneUpper,
      positive_label: "fraud",
      negative_label: "legit",
      review_label: "review",
    },
    meta: {
      model_type: "LogisticRegression",
      model_name: "demo-lr-tfidf-calibrated-v1.0",
    },
    explanation: {
      top_positive: topPositive,
      top_negative: topNegative,
      intercept: -0.42,
      summary: `Prediction based on ${topPositive.length} fraud indicators and ${topNegative.length} legitimate indicators.`,
    },
  };
}

export async function mockPredictBatch(
  instances: JobPostingInput[]
): Promise<PredictionBatchResponse> {
  await delay(300 + Math.random() * 200); // Simulate network delay

  return {
    predictions: instances.map(generateRealisticPrediction),
  };
}

export async function mockFetchMetadata(): Promise<MetadataResponse> {
  await delay(150);

  return {
    model_name: "demo-lr-tfidf-calibrated-v1.0",
    model_type: "LogisticRegression",
    feature_type: "TF-IDF",
    calibration_method: "isotonic",
    threshold: 0.5,
    gray_zone: {
      width: 0.2,
      lower: 0.4,
      upper: 0.6,
      positive_label: "fraud",
      negative_label: "legit",
      review_label: "review",
    },
    val_metrics: {
      f1: 0.892,
      precision: 0.885,
      recall: 0.899,
      roc_auc: 0.956,
      pr_auc: 0.943,
      brier: 0.087,
    },
    test_metrics: {
      f1: 0.878,
      precision: 0.871,
      recall: 0.885,
      roc_auc: 0.948,
      pr_auc: 0.936,
      brier: 0.092,
    },
    test_ece: 0.034,
  };
}

export async function mockFetchModelSummaries(limit = 20): Promise<ModelsResponse> {
  await delay(200);

  const models = [
    {
      model_name: "demo-lr-tfidf-calibrated-v1.0",
      model_type: "LogisticRegression",
      calibration_method: "isotonic",
      threshold: 0.5,
      timestamp: "2025-01-10T14:30:00Z",
      validation: {
        f1: 0.892,
        precision: 0.885,
        recall: 0.899,
        roc_auc: 0.956,
        pr_auc: 0.943,
      },
      test: {
        f1: 0.878,
        precision: 0.871,
        recall: 0.885,
        roc_auc: 0.948,
        pr_auc: 0.936,
      },
    },
    {
      model_name: "demo-xgb-tfidf-v1.0",
      model_type: "XGBoost",
      calibration_method: null,
      threshold: 0.48,
      timestamp: "2025-01-08T10:15:00Z",
      validation: {
        f1: 0.901,
        precision: 0.893,
        recall: 0.909,
        roc_auc: 0.963,
        pr_auc: 0.951,
      },
      test: {
        f1: 0.886,
        precision: 0.879,
        recall: 0.893,
        roc_auc: 0.955,
        pr_auc: 0.943,
      },
    },
    {
      model_name: "demo-rf-tfidf-v1.0",
      model_type: "RandomForest",
      calibration_method: null,
      threshold: 0.52,
      timestamp: "2025-01-05T16:45:00Z",
      validation: {
        f1: 0.867,
        precision: 0.859,
        recall: 0.875,
        roc_auc: 0.941,
        pr_auc: 0.928,
      },
      test: {
        f1: 0.853,
        precision: 0.845,
        recall: 0.861,
        roc_auc: 0.933,
        pr_auc: 0.92,
      },
    },
  ];

  return {
    items: models.slice(0, Math.min(limit, models.length)),
  };
}

export async function mockFetchTokenImportance(limit = 20): Promise<TokenImportanceResponse> {
  await delay(180);

  const positive = [
    { term: "urgent", weight: 2.34 },
    { term: "easy money", weight: 2.18 },
    { term: "work from home", weight: 1.95 },
    { term: "no experience", weight: 1.87 },
    { term: "immediate start", weight: 1.76 },
    { term: "guaranteed income", weight: 1.65 },
    { term: "limited spots", weight: 1.52 },
    { term: "act now", weight: 1.43 },
    { term: "free training", weight: 1.38 },
    { term: "earn $$$", weight: 1.29 },
    { term: "flexible hours", weight: 1.18 },
    { term: "be your own boss", weight: 1.12 },
    { term: "no interview", weight: 1.08 },
    { term: "instant approval", weight: 0.98 },
    { term: "work anywhere", weight: 0.91 },
  ];

  const negative = [
    { term: "401k", weight: -2.12 },
    { term: "health insurance", weight: -1.98 },
    { term: "pto", weight: -1.85 },
    { term: "competitive salary", weight: -1.72 },
    { term: "dental", weight: -1.68 },
    { term: "vision", weight: -1.54 },
    { term: "education reimbursement", weight: -1.43 },
    { term: "stock options", weight: -1.38 },
    { term: "professional development", weight: -1.29 },
    { term: "collaborative environment", weight: -1.18 },
    { term: "office location", weight: -1.12 },
    { term: "team building", weight: -1.05 },
    { term: "annual review", weight: -0.98 },
    { term: "career growth", weight: -0.92 },
    { term: "mentorship", weight: -0.87 },
  ];

  return {
    positive: positive.slice(0, Math.min(limit, positive.length)),
    negative: negative.slice(0, Math.min(limit, negative.length)),
  };
}

export async function mockFetchTokenFrequency(limit = 20): Promise<TokenFrequencyResponse> {
  await delay(180);

  const items = [
    { token: "urgent", positive_count: 1247, negative_count: 89, difference: 1158 },
    { token: "easy", positive_count: 1089, negative_count: 145, difference: 944 },
    { token: "home", positive_count: 1456, negative_count: 567, difference: 889 },
    { token: "guaranteed", positive_count: 892, negative_count: 45, difference: 847 },
    { token: "immediate", positive_count: 823, negative_count: 78, difference: 745 },
    { token: "experience required", positive_count: 234, negative_count: 912, difference: -678 },
    { token: "benefits", positive_count: 345, negative_count: 1001, difference: -656 },
    { token: "insurance", positive_count: 189, negative_count: 834, difference: -645 },
    { token: "salary", positive_count: 567, negative_count: 1198, difference: -631 },
    { token: "401k", positive_count: 123, negative_count: 745, difference: -622 },
  ];

  return {
    items: items.slice(0, Math.min(limit, items.length)),
  };
}

export async function mockFetchThresholdMetrics(limit = 50): Promise<ThresholdMetricsResponse> {
  await delay(200);

  const points = [];
  for (let i = 0; i <= 100; i += 2) {
    const threshold = i / 100;
    const recall = Math.max(0.1, 1 - threshold * 0.9);
    const precision = Math.min(0.95, 0.5 + threshold * 0.5);
    const f1 = (2 * precision * recall) / (precision + recall);

    points.push({
      threshold: parseFloat(threshold.toFixed(2)),
      precision: parseFloat(precision.toFixed(3)),
      recall: parseFloat(recall.toFixed(3)),
      f1: parseFloat(f1.toFixed(3)),
    });
  }

  return {
    points: points.slice(0, Math.min(limit, points.length)),
  };
}

export async function mockFetchLatencySummary(): Promise<LatencySummaryResponse> {
  await delay(150);

  return {
    items: [
      { batch_size: 1, latency_p50_ms: 12.3, latency_p95_ms: 18.7, throughput_rps: 81.3 },
      { batch_size: 10, latency_p50_ms: 45.2, latency_p95_ms: 67.8, throughput_rps: 221.2 },
      { batch_size: 50, latency_p50_ms: 178.5, latency_p95_ms: 245.3, throughput_rps: 280.1 },
      { batch_size: 100, latency_p50_ms: 342.7, latency_p95_ms: 478.9, throughput_rps: 291.8 },
    ],
  };
}

export async function mockFetchSliceMetrics(limit = 6): Promise<SliceMetricsResponse> {
  await delay(180);

  const items = [
    {
      slice: "Telecommuting",
      category: "telecommuting",
      count: 3245,
      f1: 0.823,
      precision: 0.815,
      recall: 0.831,
    },
    {
      slice: "Entry Level",
      category: "required_experience",
      count: 2876,
      f1: 0.891,
      precision: 0.883,
      recall: 0.899,
    },
    {
      slice: "Mid-Senior",
      category: "required_experience",
      count: 4532,
      f1: 0.856,
      precision: 0.848,
      recall: 0.864,
    },
    {
      slice: "IT/Tech",
      category: "industry",
      count: 5123,
      f1: 0.902,
      precision: 0.895,
      recall: 0.909,
    },
    {
      slice: "Healthcare",
      category: "industry",
      count: 3891,
      f1: 0.867,
      precision: 0.859,
      recall: 0.875,
    },
    {
      slice: "Finance",
      category: "industry",
      count: 2765,
      f1: 0.878,
      precision: 0.871,
      recall: 0.885,
    },
  ];

  return {
    items: items.slice(0, Math.min(limit, items.length)),
  };
}

export async function mockFetchReviewCases(
  limit = 25,
  policy = "gray-zone",
  offset = 0
): Promise<CasesResponse> {
  await delay(250);

  // Get user-submitted cases from the demo queue
  let userCases: any[] = [];
  if (typeof window !== "undefined") {
    try {
      const backendStatusModule = require("./backend-status");
      const demoQueue = backendStatusModule.useDemoReviewQueue.getState();
      userCases = demoQueue.getCases();
    } catch (e) {
      // Queue not available
    }
  }

  const sampleCases = [
    {
      request_id: `demo-case-${Date.now()}-1`,
      created_at: new Date(Date.now() - 3600000).toISOString(),
      probability: 0.52,
      predicted_label: "review",
      model_version: "demo-lr-tfidf-calibrated-v1.0",
      threshold: 0.5,
      text_hash: "hash1234567890abcdef",
      features_hash: "featurehash1234567890",
      payload: {
        title: "Remote Customer Service Representative - Flexible Hours",
        company_profile: "Growing tech startup seeking motivated individuals",
        description: "Handle customer inquiries via phone and email. Work from anywhere!",
        requirements: "Good communication skills, computer literacy",
        benefits: "Competitive pay, flexible schedule",
        location: "Remote",
        employment_type: "Full-time",
        required_experience: "Entry level",
        required_education: "High School or equivalent",
        industry: "Customer Service",
        function: "Customer Support",
      },
      explanation: {
        top_positive: [
          { feature: "work from home", source: "description", contribution: 0.19 },
          { feature: "flexible hours", source: "title", contribution: 0.15 },
        ],
        top_negative: [
          { feature: "tech startup", source: "company_profile", contribution: -0.12 },
          { feature: "benefits listed", source: "benefits", contribution: -0.08 },
        ],
        intercept: -0.42,
        summary: null,
      },
    },
    {
      request_id: `demo-case-${Date.now()}-2`,
      created_at: new Date(Date.now() - 7200000).toISOString(),
      probability: 0.48,
      predicted_label: "review",
      model_version: "demo-lr-tfidf-calibrated-v1.0",
      threshold: 0.5,
      text_hash: "hash0987654321fedcba",
      features_hash: "featurehash0987654321",
      payload: {
        title: "Marketing Coordinator - Immediate Start",
        company_profile: "Established marketing agency",
        description: "Coordinate marketing campaigns and social media. Fast-paced environment.",
        requirements: "Experience with social media platforms, creative mindset",
        benefits: "Health insurance, paid time off",
        location: "New York, NY",
        employment_type: "Full-time",
        required_experience: "Mid-Senior level",
        required_education: "Bachelor's degree",
        industry: "Marketing",
        function: "Marketing",
      },
      explanation: {
        top_positive: [{ feature: "immediate start", source: "title", contribution: 0.14 }],
        top_negative: [
          { feature: "health insurance", source: "benefits", contribution: -0.18 },
          { feature: "bachelor's degree", source: "required_education", contribution: -0.11 },
        ],
        intercept: -0.42,
        summary: null,
      },
    },
  ];

  // Combine user-submitted cases with sample cases (user cases first)
  const allCases = [...userCases, ...sampleCases];
  const total = allCases.length;
  const items = allCases.slice(offset, offset + limit);

  return {
    total_pending: total,
    items,
  };
}

export async function mockFetchHealth() {
  await delay(100);

  return {
    status: "demo",
    model_type: "LogisticRegression",
    threshold: 0.5,
  };
}

// Mock chat streaming
export async function mockStreamChat(
  message: string,
  onChunk: (chunk: string) => void,
  onComplete: () => void,
  onError: (error: Error) => void
): Promise<void> {
  try {
    // Generate contextual response based on message
    let response = "";

    const lowerMessage = message.toLowerCase();

    if (lowerMessage.includes("fraud") || lowerMessage.includes("scam")) {
      response = `Based on the job fraud detection analysis, I can help you understand the key indicators of fraudulent job postings:

**Common Fraud Indicators:**
1. **Vague job descriptions** - Legitimate employers provide detailed role information
2. **Unrealistic salary promises** - "Earn $10,000/week from home!" is a red flag
3. **No company information** - Scammers avoid providing verifiable company details
4. **Urgency tactics** - "Apply now! Limited spots!" creates pressure
5. **No interview process** - Legitimate companies vet candidates carefully

**Protection Tips:**
- Research the company independently
- Verify contact information
- Be wary of upfront fees
- Trust your instincts

Would you like me to analyze a specific job posting for you?`;
    } else if (lowerMessage.includes("model") || lowerMessage.includes("accuracy")) {
      response = `Our fraud detection model uses **Logistic Regression with TF-IDF features** and achieves strong performance:

**Model Performance (Test Set):**
- F1 Score: 87.8%
- Precision: 87.1% (few false alarms)
- Recall: 88.5% (catches most fraud)
- ROC-AUC: 94.8%

**Key Features:**
- **Calibrated probabilities** using isotonic regression
- **Gray-zone detection** for uncertain cases requiring human review
- **Explainable predictions** showing which words/features contributed to the decision

The model analyzes text patterns, company information, and job details to identify suspicious postings.`;
    } else if (lowerMessage.includes("how") || lowerMessage.includes("work")) {
      response = `The job fraud detection system works through several stages:

**1. Feature Extraction**
   - Converts job posting text into TF-IDF vectors
   - Captures word importance and frequency patterns

**2. Prediction**
   - Logistic regression model scores each posting
   - Outputs probability of fraud (0-100%)

**3. Decision Making**
   - High confidence fraud: probability > 60%
   - High confidence legitimate: probability < 40%
   - Gray zone (needs review): probability between 40-60%

**4. Explanation**
   - Shows top features contributing to fraud/legit classification
   - Helps reviewers understand the model's reasoning

**5. Human-in-the-Loop**
   - Uncertain cases go to human reviewers
   - Feedback improves the system over time

This demo mode uses realistic mock data to showcase these capabilities!`;
    } else if (lowerMessage.includes("demo") || lowerMessage.includes("backend")) {
      response = `You're currently using **Demo Mode** - the backend server is not connected, but I'm providing realistic mock responses to showcase the application's capabilities.

**What's Working in Demo Mode:**
- Job posting fraud analysis with realistic predictions
- Model performance metrics and leaderboard
- Feature importance visualization
- Gray-zone case reviews
- This AI chat assistant

**Differences from Live Mode:**
- Predictions use pattern matching instead of the actual ML model
- Review cases are pre-generated samples
- Feedback submissions are simulated (not persisted)

**To enable full functionality:**
1. Start the backend server: \`uvicorn app.main:app --reload\`
2. Set the API URL in environment variables
3. The frontend will automatically detect and connect

Feel free to explore all features - they work just like the real system!`;
    } else if (lowerMessage.includes("hello") || lowerMessage.includes("hi")) {
      response = `Hello! I'm the AI assistant for the Job Fraud Detection system. I can help you:

- **Understand fraud indicators** in job postings
- **Explain model predictions** and confidence scores
- **Provide insights** about common scam tactics
- **Answer questions** about the detection system

Currently running in **Demo Mode** with realistic mock data. What would you like to know?`;
    } else {
      response = `I understand you're asking about "${message}".

In this **Demo Mode**, I can provide information about:
- How the fraud detection model works
- Common indicators of fraudulent job postings
- Model performance and metrics
- The human review process for uncertain cases

The actual backend with the full AI model is not connected, but this demo showcases the system's capabilities with realistic responses.

What specific aspect would you like to learn more about?`;
    }

    // Stream the response word by word for realistic effect
    const words = response.split(" ");
    for (let i = 0; i < words.length; i++) {
      const chunk = (i === 0 ? "" : " ") + words[i];
      onChunk(chunk);
      await delay(30 + Math.random() * 40); // Realistic typing speed
    }

    onComplete();
  } catch (error) {
    onError(error as Error);
  }
}
