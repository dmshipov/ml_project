variable "env_name" {
  type        = string
  description = "Имя окружения (staging/prod)"
}

variable "yc_token" {
  type        = string
  description = "OAuth токен Яндекс.Облака"
}

variable "yc_cloud_id" {
  type        = string
  description = "Cloud ID"
}

variable "yc_folder_id" {
  type        = string
  description = "Folder ID"
}

variable "yc_zone" {
  type        = string
  description = "Зона доступности"
  default     = "ru-central1-a"
}

variable "k8s_service_account_id" {
  type        = string
  description = "ID сервисного аккаунта для Kubernetes"
}
