provider "aws" {
  region = var.region
}

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  name = "${var.project_name}-${var.environment}"
  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.8"

  name = "${local.name}-vpc"
  cidr = var.vpc_cidr

  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = var.private_subnets
  public_subnets  = var.public_subnets

  enable_nat_gateway = true
  single_nat_gateway = false

  tags = local.tags
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.24"

  cluster_name    = "${local.name}-eks"
  cluster_version = var.cluster_version

  cluster_endpoint_public_access = true

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    default = {
      min_size       = var.node_min_size
      max_size       = var.node_max_size
      desired_size   = var.node_desired_size
      instance_types = var.node_instance_types
      capacity_type  = "ON_DEMAND"
    }
  }

  enable_cluster_creator_admin_permissions = true

  tags = local.tags
}

resource "aws_ecr_repository" "api" {
  name                 = "${local.name}/spot-scam-api"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = local.tags
}

resource "aws_ecr_repository" "frontend" {
  name                 = "${local.name}/spot-scam-frontend"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = local.tags
}

resource "aws_ecr_repository" "model" {
  name                 = "${local.name}/spot-scam-model"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = local.tags
}

resource "aws_s3_bucket" "artifacts_backup" {
  bucket        = "${replace(local.name, "_", "-")}-artifacts-backup"
  force_destroy = false
  tags          = local.tags
}

resource "aws_security_group" "efs" {
  name        = "${local.name}-efs-sg"
  description = "Security group for EFS"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 2049
    to_port     = 2049
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.tags
}

resource "aws_efs_file_system" "shared" {
  creation_token = "${local.name}-efs"
  encrypted      = true
  tags           = local.tags
}

resource "aws_efs_mount_target" "shared" {
  for_each = toset(module.vpc.private_subnets)

  file_system_id  = aws_efs_file_system.shared.id
  subnet_id       = each.value
  security_groups = [aws_security_group.efs.id]
}

resource "aws_efs_access_point" "shared" {
  file_system_id = aws_efs_file_system.shared.id

  root_directory {
    path = "/spot-scam"

    creation_info {
      owner_gid   = 1000
      owner_uid   = 1000
      permissions = "0770"
    }
  }

  tags = local.tags
}

data "aws_eks_cluster_auth" "this" {
  name = module.eks.cluster_name
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  token                  = data.aws_eks_cluster_auth.this.token
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    token                  = data.aws_eks_cluster_auth.this.token
  }
}

resource "helm_release" "ingress_nginx" {
  name             = "ingress-nginx"
  repository       = "https://kubernetes.github.io/ingress-nginx"
  chart            = "ingress-nginx"
  namespace        = "ingress-nginx"
  create_namespace = true
}

resource "helm_release" "argo_rollouts" {
  name             = "argo-rollouts"
  repository       = "https://argoproj.github.io/argo-helm"
  chart            = "argo-rollouts"
  namespace        = "argo-rollouts"
  create_namespace = true
}

resource "helm_release" "efs_csi_driver" {
  name             = "aws-efs-csi-driver"
  repository       = "https://kubernetes-sigs.github.io/aws-efs-csi-driver/"
  chart            = "aws-efs-csi-driver"
  namespace        = "kube-system"
  create_namespace = false
}

resource "kubernetes_storage_class" "efs_sc" {
  metadata {
    name = "efs-sc"
  }

  storage_provisioner = "efs.csi.aws.com"
  reclaim_policy      = "Retain"
  volume_binding_mode = "Immediate"
  allow_volume_expansion = true

  parameters = {
    provisioningMode = "efs-ap"
    fileSystemId     = aws_efs_file_system.shared.id
    directoryPerms   = "770"
    gidRangeStart    = "1000"
    gidRangeEnd      = "2000"
  }

  depends_on = [helm_release.efs_csi_driver]
}
