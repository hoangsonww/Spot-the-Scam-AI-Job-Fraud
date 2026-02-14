output "cluster_name" {
  value       = oci_containerengine_cluster.main.name
  description = "OKE cluster name."
}

output "cluster_id" {
  value       = oci_containerengine_cluster.main.id
  description = "OKE cluster OCID."
}

output "configure_kubectl" {
  value       = "oci ce cluster create-kubeconfig --cluster-id ${oci_containerengine_cluster.main.id} --file $HOME/.kube/config --region ${var.region} --token-version 2.0.0 --kube-endpoint PUBLIC_ENDPOINT"
  description = "Command to configure local kubectl context."
}

output "api_repository" {
  value       = oci_artifacts_container_repository.api.display_name
  description = "OCI registry repository for API image."
}

output "frontend_repository" {
  value       = oci_artifacts_container_repository.frontend.display_name
  description = "OCI registry repository for frontend image."
}

output "model_repository" {
  value       = oci_artifacts_container_repository.model.display_name
  description = "OCI registry repository for model image."
}

output "fss_filesystem_id" {
  value       = oci_file_storage_file_system.shared.id
  description = "OCI File Storage file system OCID used for RWX volumes."
}
