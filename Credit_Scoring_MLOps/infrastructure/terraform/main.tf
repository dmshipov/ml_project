terraform {
  required_version = ">= 1.6.0"

  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = ">= 0.130.0"
    }
  }

  backend "s3" {
    endpoint   = "storage.yandexcloud.net"
    bucket     = "credit-tf-state"
    key        = "terraform.tfstate"
    region     = "ru-central1"
    access_key = "CHANGE_ME"
    secret_key = "CHANGE_ME"
    skip_region_validation      = true
    skip_credentials_validation = true
  }
}

provider "yandex" {
  token     = var.yc_token
  cloud_id  = var.yc_cloud_id
  folder_id = var.yc_folder_id
  zone      = var.yc_zone
}

module "network" {
  source   = "./modules/network"
  env_name = var.env_name
}

module "kubernetes" {
  source             = "./modules/kubernetes"
  env_name           = var.env_name
  network_id         = module.network.network_id
  subnet_id          = module.network.subnet_id
  service_account_id = var.k8s_service_account_id
}

module "storage" {
  source   = "./modules/storage"
  env_name = var.env_name
}
