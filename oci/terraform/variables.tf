variable "tenancy_ocid" {
  type        = string
  description = "OCI tenancy OCID."
}

variable "user_ocid" {
  type        = string
  description = "OCI user OCID for Terraform authentication."
}

variable "fingerprint" {
  type        = string
  description = "API signing key fingerprint."
}

variable "private_key_path" {
  type        = string
  description = "Path to OCI API private key."
}

variable "region" {
  type        = string
  description = "OCI region identifier."
  default     = "us-ashburn-1"
}

variable "compartment_ocid" {
  type        = string
  description = "Compartment OCID used for deployment resources."
}

variable "project_name" {
  type        = string
  description = "Project prefix for OCI resources."
  default     = "spot-scam"
}

variable "environment" {
  type        = string
  description = "Environment identifier."
  default     = "prod"
}

variable "vcn_cidr" {
  type        = string
  description = "VCN CIDR range."
  default     = "10.70.0.0/16"
}

variable "oke_subnet_cidr" {
  type        = string
  description = "Subnet CIDR used by OKE workers."
  default     = "10.70.10.0/24"
}

variable "lb_subnet_cidr" {
  type        = string
  description = "Subnet CIDR used for load balancers."
  default     = "10.70.20.0/24"
}

variable "kubernetes_version" {
  type        = string
  description = "OKE Kubernetes version."
  default     = "v1.30.1"
}

variable "node_shape" {
  type        = string
  description = "Compute shape for OKE worker nodes."
  default     = "VM.Standard.E4.Flex"
}

variable "node_ocpus" {
  type        = number
  description = "OCPUs per worker node for flex shape."
  default     = 4
}

variable "node_memory_gb" {
  type        = number
  description = "Memory per worker node in GB for flex shape."
  default     = 32
}

variable "node_count" {
  type        = number
  description = "Worker node count."
  default     = 3
}
