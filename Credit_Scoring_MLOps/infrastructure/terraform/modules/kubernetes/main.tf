resource "yandex_kubernetes_cluster" "this" {
  name       = "${var.env_name}-k8s"
  network_id = var.network_id
  master {
    version   = "1.29"
    public_ip = true

    maintenance_policy {
      auto_upgrade = true
    }
  }
  service_account_id      = var.service_account_id
  node_service_account_id = var.service_account_id
}

resource "yandex_kubernetes_node_group" "cpu_nodes" {
  cluster_id = yandex_kubernetes_cluster.this.id
  name       = "${var.env_name}-cpu-nodes"
  version    = "1.29"

  scale_policy {
    auto_scale {
      min = 1
      max = 3
    }
  }

  allocation_policy {
    location {
      zone = "ru-central1-a"
    }
  }

  instance_template {
    platform_id = "standard-v3"
    resources {
      cores  = 2
      memory = 4
    }
    boot_disk {
      type = "network-ssd"
      size = 64
    }
    network_interface {
      subnet_id = var.subnet_id
    }
  }
}

output "cluster_id" {
  value = yandex_kubernetes_cluster.this.id
}
