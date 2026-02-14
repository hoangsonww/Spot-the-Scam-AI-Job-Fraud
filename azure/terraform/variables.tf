variable "location" {
  type        = string
  description = "Azure region."
  default     = "eastus"
}

variable "project_name" {
  type        = string
  description = "Project prefix used in resource names."
  default     = "spotscam"
}

variable "environment" {
  type        = string
  description = "Environment identifier."
  default     = "prod"
}

variable "kubernetes_version" {
  type        = string
  description = "AKS Kubernetes version."
  default     = "1.30.3"
}

variable "node_count" {
  type        = number
  description = "Initial AKS node count."
  default     = 3
}

variable "node_vm_size" {
  type        = string
  description = "AKS VM size."
  default     = "Standard_D4s_v5"
}

variable "vnet_cidr" {
  type        = string
  description = "CIDR range for VNet."
  default     = "10.50.0.0/16"
}

variable "aks_subnet_cidr" {
  type        = string
  description = "CIDR range for AKS nodes."
  default     = "10.50.1.0/24"
}
