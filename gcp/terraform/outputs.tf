output "cluster_name" {
  value       = google_container_cluster.main.name
  description = "GKE cluster name."
}

output "cluster_location" {
  value       = google_container_cluster.main.location
  description = "GKE cluster location."
}

output "configure_kubectl" {
  value       = "gcloud container clusters get-credentials ${google_container_cluster.main.name} --region ${google_container_cluster.main.location} --project ${var.project_id}"
  description = "Command to configure local kubectl context."
}

output "artifact_registry_repo" {
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker.repository_id}"
  description = "Artifact Registry repository for container images."
}

output "filestore_instance" {
  value       = google_filestore_instance.shared.name
  description = "Filestore instance used for RWX volume provisioning."
}

output "storage_class_name" {
  value       = kubernetes_storage_class.filestore_rwx.metadata[0].name
  description = "RWX storage class for Spot the Scam PVCs."
}
