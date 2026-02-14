provider "oci" {
  tenancy_ocid     = var.tenancy_ocid
  user_ocid        = var.user_ocid
  fingerprint      = var.fingerprint
  private_key_path = var.private_key_path
  region           = var.region
}

locals {
  name = "${var.project_name}-${var.environment}"
}

data "oci_identity_availability_domains" "ads" {
  compartment_id = var.tenancy_ocid
}

resource "oci_core_vcn" "main" {
  cidr_block     = var.vcn_cidr
  compartment_id = var.compartment_ocid
  display_name   = "${local.name}-vcn"
  dns_label      = "spotscam"
}

resource "oci_core_internet_gateway" "main" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.main.id
  display_name   = "${local.name}-igw"
  enabled        = true
}

resource "oci_core_route_table" "main" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.main.id
  display_name   = "${local.name}-rt"

  route_rules {
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
    network_entity_id = oci_core_internet_gateway.main.id
  }
}

resource "oci_core_security_list" "main" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.main.id
  display_name   = "${local.name}-sl"

  ingress_security_rules {
    protocol = "6"
    source   = "0.0.0.0/0"
    tcp_options {
      min = 80
      max = 80
    }
  }

  ingress_security_rules {
    protocol = "6"
    source   = "0.0.0.0/0"
    tcp_options {
      min = 443
      max = 443
    }
  }

  egress_security_rules {
    protocol    = "all"
    destination = "0.0.0.0/0"
  }
}

resource "oci_core_subnet" "oke" {
  cidr_block        = var.oke_subnet_cidr
  compartment_id    = var.compartment_ocid
  vcn_id            = oci_core_vcn.main.id
  display_name      = "${local.name}-oke-subnet"
  route_table_id    = oci_core_route_table.main.id
  security_list_ids = [oci_core_security_list.main.id]
  dns_label         = "okesub"
}

resource "oci_core_subnet" "lb" {
  cidr_block        = var.lb_subnet_cidr
  compartment_id    = var.compartment_ocid
  vcn_id            = oci_core_vcn.main.id
  display_name      = "${local.name}-lb-subnet"
  route_table_id    = oci_core_route_table.main.id
  security_list_ids = [oci_core_security_list.main.id]
  dns_label         = "lbsub"
}

resource "oci_containerengine_cluster" "main" {
  compartment_id     = var.compartment_ocid
  kubernetes_version = var.kubernetes_version
  name               = "${local.name}-oke"
  vcn_id             = oci_core_vcn.main.id

  endpoint_config {
    is_public_ip_enabled = true
    subnet_id            = oci_core_subnet.lb.id
  }

  options {
    service_lb_subnet_ids = [oci_core_subnet.lb.id]
  }
}

resource "oci_containerengine_node_pool" "main" {
  cluster_id         = oci_containerengine_cluster.main.id
  compartment_id     = var.compartment_ocid
  kubernetes_version = var.kubernetes_version
  name               = "${local.name}-nodepool"
  node_shape         = var.node_shape

  node_config_details {
    size = var.node_count

    node_shape_config {
      memory_in_gbs = var.node_memory_gb
      ocpus         = var.node_ocpus
    }

    placement_configs {
      availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
      subnet_id           = oci_core_subnet.oke.id
    }
  }
}

resource "oci_artifacts_container_repository" "api" {
  compartment_id = var.compartment_ocid
  display_name   = "${local.name}/spot-scam-api"
  is_immutable   = false
}

resource "oci_artifacts_container_repository" "frontend" {
  compartment_id = var.compartment_ocid
  display_name   = "${local.name}/spot-scam-frontend"
  is_immutable   = false
}

resource "oci_artifacts_container_repository" "model" {
  compartment_id = var.compartment_ocid
  display_name   = "${local.name}/spot-scam-model"
  is_immutable   = false
}

resource "oci_objectstorage_bucket" "artifacts_backup" {
  compartment_id = var.compartment_ocid
  namespace      = data.oci_objectstorage_namespace.ns.namespace
  name           = "${local.name}-artifacts-backup"
  access_type    = "NoPublicAccess"
  storage_tier   = "Standard"
}

data "oci_objectstorage_namespace" "ns" {
  compartment_id = var.compartment_ocid
}

resource "oci_file_storage_file_system" "shared" {
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  compartment_id      = var.compartment_ocid
  display_name        = "${local.name}-fss"
}

resource "oci_file_storage_mount_target" "shared" {
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  compartment_id      = var.compartment_ocid
  subnet_id           = oci_core_subnet.oke.id
  display_name        = "${local.name}-mt"
}

resource "oci_file_storage_export" "shared" {
  export_set_id  = oci_file_storage_mount_target.shared.export_set_id
  file_system_id = oci_file_storage_file_system.shared.id
  path           = "/spot-scam"
}
