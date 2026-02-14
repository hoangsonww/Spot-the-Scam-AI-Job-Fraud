variable "project_id" {
  type        = string
  description = "GCP project ID."
}

variable "region" {
  type        = string
  description = "Primary GCP region."
  default     = "us-central1"
}

variable "zone" {
  type        = string
  description = "Primary GCP zone."
  default     = "us-central1-a"
}

variable "project_name" {
  type        = string
  description = "Project prefix for resource names."
  default     = "spot-scam"
}

variable "environment" {
  type        = string
  description = "Environment identifier."
  default     = "prod"
}

variable "gke_version" {
  type        = string
  description = "GKE master/node version."
  default     = "1.30.4-gke.1451000"
}

variable "node_count" {
  type        = number
  description = "Initial worker node count."
  default     = 3
}

variable "node_machine_type" {
  type        = string
  description = "Worker node machine type."
  default     = "e2-standard-4"
}

variable "vpc_cidr" {
  type        = string
  description = "Primary VPC CIDR range."
  default     = "10.60.0.0/16"
}

variable "gke_subnet_cidr" {
  type        = string
  description = "GKE subnet CIDR."
  default     = "10.60.1.0/24"
}
