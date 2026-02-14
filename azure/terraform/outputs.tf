output "resource_group" {
  value       = azurerm_resource_group.main.name
  description = "Resource group containing Spot the Scam infrastructure."
}

output "cluster_name" {
  value       = azurerm_kubernetes_cluster.main.name
  description = "AKS cluster name."
}

output "configure_kubectl" {
  value       = "az aks get-credentials --resource-group ${azurerm_resource_group.main.name} --name ${azurerm_kubernetes_cluster.main.name} --overwrite-existing"
  description = "Command to configure local kubectl context."
}

output "acr_login_server" {
  value       = azurerm_container_registry.main.login_server
  description = "ACR login server for image pushes."
}

output "storage_class_name" {
  value       = kubernetes_storage_class.azurefile_rwx.metadata[0].name
  description = "RWX storage class used by Spot the Scam PVCs."
}
