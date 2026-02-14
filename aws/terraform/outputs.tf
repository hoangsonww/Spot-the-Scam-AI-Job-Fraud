output "cluster_name" {
  value       = module.eks.cluster_name
  description = "EKS cluster name."
}

output "cluster_region" {
  value       = var.region
  description = "AWS region used by the deployment."
}

output "configure_kubectl" {
  value       = "aws eks update-kubeconfig --name ${module.eks.cluster_name} --region ${var.region}"
  description = "Command to configure local kubectl context."
}

output "api_repository_url" {
  value       = aws_ecr_repository.api.repository_url
  description = "ECR URL for API image."
}

output "frontend_repository_url" {
  value       = aws_ecr_repository.frontend.repository_url
  description = "ECR URL for frontend image."
}

output "model_repository_url" {
  value       = aws_ecr_repository.model.repository_url
  description = "ECR URL for model image."
}

output "storage_class_name" {
  value       = kubernetes_storage_class.efs_sc.metadata[0].name
  description = "RWX storage class for Spot the Scam PVCs."
}

output "efs_filesystem_id" {
  value       = aws_efs_file_system.shared.id
  description = "EFS file system backing shared volumes."
}
