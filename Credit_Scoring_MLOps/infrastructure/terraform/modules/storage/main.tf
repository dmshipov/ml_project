resource "yandex_storage_bucket" "models" {
  bucket        = "${var.env_name}-credit-models"
  acl           = "private"
  force_destroy = true
}

output "bucket_name" {
  value = yandex_storage_bucket.models.bucket
}
