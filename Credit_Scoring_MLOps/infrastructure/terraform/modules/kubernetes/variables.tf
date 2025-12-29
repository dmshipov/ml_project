variable "env_name" {
  type        = string
  description = "Имя окружения"
}

variable "network_id" {
  type        = string
  description = "ID сети"
}

variable "subnet_id" {
  type        = string
  description = "ID подсети"
}

variable "service_account_id" {
  type        = string
  description = "ID сервисного аккаунта"
}
