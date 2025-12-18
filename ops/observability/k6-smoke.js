import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  vus: 5,
  duration: '2m',
  thresholds: {
    http_req_duration: ['p(95)<800'],
    http_req_failed: ['rate<0.02'],
  },
};

const apiBase = __ENV.API_BASE || 'http://localhost:8000';

const sampleJob = {
  title: 'Software Engineer',
  description:
    'We are hiring a software engineer to build backend services. Work with distributed systems and APIs.',
  company_profile: 'Legit company focused on engineering excellence.',
  requirements: '3+ years experience, Python, FastAPI',
  benefits: 'Health, PTO, learning stipend',
  location: 'Remote',
  employment_type: 'Full-time',
  required_experience: 'Mid-Senior level',
  required_education: 'Bachelors',
  industry: 'Information Technology',
  function: 'Engineering',
  telecommuting: 1,
  has_company_logo: 1,
  has_questions: 0,
};

export default function () {
  const health = http.get(`${apiBase}/health`);
  check(health, {
    'health is 200': (r) => r.status === 200,
  });

  const predict = http.post(`${apiBase}/predict`, JSON.stringify(sampleJob), {
    headers: { 'Content-Type': 'application/json' },
  });

  check(predict, {
    'predict is 200': (r) => r.status === 200,
    'predict has score': (r) => !!(r.json() && r.json().score !== undefined),
  });

  sleep(1);
}
