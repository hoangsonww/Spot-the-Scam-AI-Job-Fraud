terraform {
  required_version = ">= 1.8.0"

  required_providers {
    oci = {
      source  = "oracle/oci"
      version = "~> 6.20"
    }
  }
}
