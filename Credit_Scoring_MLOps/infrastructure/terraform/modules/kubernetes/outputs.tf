output "cluster_id" {
  value = yandex_kubernetes_cluster.main.id
}

output "public_endpoint" {
  value = yandex_kubernetes_cluster.main.master[0].external_v4_address
}
