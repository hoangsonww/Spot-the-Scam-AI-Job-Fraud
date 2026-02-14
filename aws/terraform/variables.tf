variable "region" {
  type        = string
  description = "AWS region for deployment."
  default     = "us-east-1"
}

variable "project_name" {
  type        = string
  description = "Project prefix used for naming resources."
  default     = "spot-scam"
}

variable "environment" {
  type        = string
  description = "Environment identifier."
  default     = "prod"
}

variable "cluster_version" {
  type        = string
  description = "EKS control plane version."
  default     = "1.30"
}

variable "vpc_cidr" {
  type        = string
  description = "VPC CIDR range."
  default     = "10.40.0.0/16"
}

variable "private_subnets" {
  type        = list(string)
  description = "Private subnet CIDR blocks."
  default     = ["10.40.1.0/24", "10.40.2.0/24", "10.40.3.0/24"]
}

variable "public_subnets" {
  type        = list(string)
  description = "Public subnet CIDR blocks."
  default     = ["10.40.101.0/24", "10.40.102.0/24", "10.40.103.0/24"]
}

variable "node_instance_types" {
  type        = list(string)
  description = "EKS node group instance types."
  default     = ["m6i.large"]
}

variable "node_desired_size" {
  type        = number
  description = "Desired worker node count."
  default     = 3
}

variable "node_min_size" {
  type        = number
  description = "Minimum worker node count."
  default     = 3
}

variable "node_max_size" {
  type        = number
  description = "Maximum worker node count."
  default     = 8
}
