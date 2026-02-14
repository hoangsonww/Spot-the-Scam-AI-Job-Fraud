provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

locals {
  name = "${var.project_name}-${var.environment}"
}

resource "google_project_service" "container" {
  service            = "container.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifact_registry" {
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "filestore" {
  service            = "file.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "compute" {
  service            = "compute.googleapis.com"
  disable_on_destroy = false
}

resource "google_compute_network" "vpc" {
  name                    = "${local.name}-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "gke" {
  name          = "${local.name}-gke-subnet"
  region        = var.region
  network       = google_compute_network.vpc.id
  ip_cidr_range = var.gke_subnet_cidr

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.61.0.0/16"
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.62.0.0/20"
  }
}

resource "google_container_cluster" "main" {
  name     = "${local.name}-gke"
  location = var.region

  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.gke.name

  min_master_version = var.gke_version

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  deletion_protection = false

  depends_on = [
    google_project_service.container,
    google_project_service.compute
  ]
}

resource "google_container_node_pool" "main" {
  name       = "primary-node-pool"
  location   = var.region
  cluster    = google_container_cluster.main.name
  node_count = var.node_count

  node_config {
    machine_type = var.node_machine_type
    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    labels = {
      workload = "spot-scam"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

resource "google_artifact_registry_repository" "docker" {
  location      = var.region
  repository_id = "spot-scam"
  description   = "Spot the Scam container images"
  format        = "DOCKER"

  depends_on = [google_project_service.artifact_registry]
}

resource "google_filestore_instance" "shared" {
  name     = "${local.name}-filestore"
  location = var.zone
  tier     = "BASIC_HDD"

  file_shares {
    capacity_gb = 1024
    name        = "spot_scam"
  }

  networks {
    network = google_compute_network.vpc.name
    modes   = ["MODE_IPV4"]
  }

  depends_on = [google_project_service.filestore]
}

data "google_client_config" "default" {}

provider "kubernetes" {
  host                   = "https://${google_container_cluster.main.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.main.master_auth[0].cluster_ca_certificate)
}

provider "helm" {
  kubernetes {
    host                   = "https://${google_container_cluster.main.endpoint}"
    token                  = data.google_client_config.default.access_token
    cluster_ca_certificate = base64decode(google_container_cluster.main.master_auth[0].cluster_ca_certificate)
  }
}

resource "helm_release" "ingress_nginx" {
  name             = "ingress-nginx"
  repository       = "https://kubernetes.github.io/ingress-nginx"
  chart            = "ingress-nginx"
  namespace        = "ingress-nginx"
  create_namespace = true
}

resource "helm_release" "argo_rollouts" {
  name             = "argo-rollouts"
  repository       = "https://argoproj.github.io/argo-helm"
  chart            = "argo-rollouts"
  namespace        = "argo-rollouts"
  create_namespace = true
}

resource "kubernetes_storage_class" "filestore_rwx" {
  metadata {
    name = "filestore-rwx"
  }

  storage_provisioner = "filestore.csi.storage.gke.io"
  reclaim_policy      = "Retain"
  allow_volume_expansion = true
  volume_binding_mode = "Immediate"

  parameters = {
    tier                  = "BASIC_HDD"
    network               = google_compute_network.vpc.name
    connect-mode          = "PRIVATE_SERVICE_ACCESS"
    reserved-ipv4-cidr    = "10.63.0.0/29"
  }
}
