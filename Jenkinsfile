pipeline {
  agent any

  options {
    ansiColor('xterm')
    timestamps()
    disableConcurrentBuilds()
    timeout(time: 120, unit: 'MINUTES')
    buildDiscarder(logRotator(numToKeepStr: '50', artifactNumToKeepStr: '30'))
  }

  parameters {
    choice(name: 'ACTION', choices: ['deploy', 'promote', 'abort', 'undo', 'argocd_sync'], description: 'Pipeline action to execute.')
    choice(name: 'PROVIDER', choices: ['aws', 'azure', 'gcp', 'oci'], description: 'Cloud provider pack to use.')
    choice(name: 'STRATEGY', choices: ['canary', 'bluegreen'], description: 'Deployment strategy for deploy action.')
    string(name: 'NAMESPACE', defaultValue: 'spot-scam', description: 'Kubernetes namespace.')
    string(name: 'IMAGE_TAG', defaultValue: '', description: 'Image tag override. Leave empty for release-<build>-<sha>.')
    string(name: 'API_IMAGE_REPO', defaultValue: '', description: 'API image repository, for example <registry>/spot-scam-api.')
    string(name: 'FRONTEND_IMAGE_REPO', defaultValue: '', description: 'Frontend image repository.')
    string(name: 'MODEL_IMAGE_REPO', defaultValue: '', description: 'Model image repository (optional).')
    string(name: 'GCP_PROJECT_ID', defaultValue: '', description: 'Required when PROVIDER=gcp.')
    string(name: 'KUBECONFIG_COMMAND', defaultValue: '', description: 'Optional command to configure kubectl context (used for manual actions and deploys without terraform apply).')
    string(name: 'ARGOCD_APP_NAME', defaultValue: '', description: 'Required when ACTION=argocd_sync.')
    string(name: 'ARGOCD_SERVER', defaultValue: '', description: 'Optional Argo CD API server for argocd CLI.')
    string(name: 'ARGOCD_TIMEOUT_SEC', defaultValue: '600', description: 'Timeout seconds for argocd sync/wait.')
    booleanParam(name: 'RUN_QUALITY_GATES', defaultValue: true, description: 'Run backend/frontend quality gates for deploy action.')
    booleanParam(name: 'RUN_TERRAFORM_APPLY', defaultValue: true, description: 'Execute terraform apply for deploy action.')
    booleanParam(name: 'RUN_IMAGE_BUILD_PUSH', defaultValue: true, description: 'Build and push images for deploy action.')
    booleanParam(name: 'PUSH_MODEL_IMAGE', defaultValue: false, description: 'Build and push model image from docker/model context when artifacts are present.')
    booleanParam(name: 'AUTO_PROMOTE_AFTER_DEPLOY', defaultValue: false, description: 'Automatically promote rollout after deploy.')
    booleanParam(name: 'AUTO_APPROVE_TERRAFORM', defaultValue: false, description: 'If false, pipeline pauses for manual terraform apply approval.')
  }

  environment {
    TF_IN_AUTOMATION = 'true'
    TF_INPUT = '0'
    PIP_NO_CACHE_DIR = '1'
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
        script {
          env.GIT_SHA = sh(returnStdout: true, script: 'git rev-parse --short=12 HEAD').trim()
          env.EFFECTIVE_TAG = params.IMAGE_TAG?.trim() ? params.IMAGE_TAG.trim() : "release-${env.BUILD_NUMBER}-${env.GIT_SHA}"
          env.TERRAFORM_DIR = "${params.PROVIDER}/terraform"
          env.OVERLAY_DIR = "${params.PROVIDER}/k8s/${params.STRATEGY}"
        }
      }
    }

    stage('Validate Inputs') {
      steps {
        script {
          if (params.NAMESPACE != 'spot-scam') {
            error('Current kustomize overlays are namespace-pinned to spot-scam. Use NAMESPACE=spot-scam or refactor overlay namespaces.')
          }
          if (params.ACTION == 'deploy') {
            if (!params.API_IMAGE_REPO?.trim()) {
              error('API_IMAGE_REPO is required for deploy action.')
            }
            if (!params.FRONTEND_IMAGE_REPO?.trim()) {
              error('FRONTEND_IMAGE_REPO is required for deploy action.')
            }
            if (params.PROVIDER == 'gcp' && !params.GCP_PROJECT_ID?.trim()) {
              error('GCP_PROJECT_ID is required when PROVIDER=gcp.')
            }
            if (!params.RUN_IMAGE_BUILD_PUSH && !params.IMAGE_TAG?.trim()) {
              error('IMAGE_TAG must be explicitly provided when RUN_IMAGE_BUILD_PUSH=false to avoid deploying a non-existent image tag.')
            }
          } else if (params.ACTION == 'argocd_sync') {
            if (!params.ARGOCD_APP_NAME?.trim()) {
              error('ARGOCD_APP_NAME is required when ACTION=argocd_sync.')
            }
          }
        }

        sh '''#!/usr/bin/env bash
          set -euo pipefail
          test -f "${OVERLAY_DIR}/kustomization.yaml"
          test -x scripts/deploy_multi_cloud.sh
          test -x ops/ci/validate_deployment_assets.sh
          test -x ops/ci/preflight_deploy_checks.sh
        '''

        script {
          if (params.ACTION == 'deploy') {
            sh '''#!/usr/bin/env bash
              set -euo pipefail
              command -v kubectl >/dev/null
              if [[ "${RUN_IMAGE_BUILD_PUSH}" == "true" ]]; then
                command -v docker >/dev/null
              fi
              if [[ "${RUN_TERRAFORM_APPLY}" == "true" ]]; then
                test -f "${TERRAFORM_DIR}/main.tf"
                command -v terraform >/dev/null
              fi
            '''
          } else if (params.ACTION == 'promote' || params.ACTION == 'abort' || params.ACTION == 'undo') {
            sh '''#!/usr/bin/env bash
              set -euo pipefail
              command -v kubectl >/dev/null
              command -v argo-rollouts >/dev/null
            '''
          } else if (params.ACTION == 'argocd_sync') {
            sh '''#!/usr/bin/env bash
              set -euo pipefail
              command -v argocd >/dev/null
              test -x ops/ci/argocd_sync_wait.sh
            '''
          }
        }
      }
    }

    stage('Quality Gates') {
      when {
        expression { params.ACTION == 'deploy' && params.RUN_QUALITY_GATES }
      }
      steps {
        sh '''#!/usr/bin/env bash
          set -euo pipefail
          python3 -m venv .venv
          source .venv/bin/activate
          python -m pip install --upgrade pip setuptools wheel
          pip install -e '.[dev]'
          make format-check
          make lint
          make type-check
          make test
        '''

        sh '''#!/usr/bin/env bash
          set -euo pipefail
          cd frontend
          npm ci
          npm run lint
          npm run type-check
          npm run build
        '''
      }
    }

    stage('Validate Deployment Assets') {
      when {
        expression { params.ACTION == 'deploy' }
      }
      steps {
        sh '''#!/usr/bin/env bash
          set -euo pipefail
          ./ops/ci/validate_deployment_assets.sh
        '''
      }
    }

    stage('Build Images') {
      when {
        expression { params.ACTION == 'deploy' && params.RUN_IMAGE_BUILD_PUSH }
      }
      steps {
        sh '''#!/usr/bin/env bash
          set -euo pipefail
          docker build -t "${API_IMAGE_REPO}:${EFFECTIVE_TAG}" .
          docker build -t "${FRONTEND_IMAGE_REPO}:${EFFECTIVE_TAG}" -f frontend/Dockerfile frontend

          if [[ "${PUSH_MODEL_IMAGE}" == "true" ]]; then
            if [[ -n "${MODEL_IMAGE_REPO}" && -d artifacts ]]; then
              rm -rf docker/model/artifacts
              cp -R artifacts docker/model/artifacts
              docker build -t "${MODEL_IMAGE_REPO}:${EFFECTIVE_TAG}" -f docker/model/Dockerfile docker/model
            else
              echo "Skipping model image build: MODEL_IMAGE_REPO not set or artifacts/ missing."
            fi
          fi
        '''
      }
    }

    stage('Cloud and Registry Auth') {
      when {
        expression { params.ACTION == 'deploy' && (params.RUN_IMAGE_BUILD_PUSH || params.RUN_TERRAFORM_APPLY) }
      }
      steps {
        script {
          if (params.PROVIDER == 'aws') {
            withCredentials([
              usernamePassword(credentialsId: 'spot-scam-aws-credentials', usernameVariable: 'AWS_ACCESS_KEY_ID', passwordVariable: 'AWS_SECRET_ACCESS_KEY')
            ]) {
              sh '''#!/usr/bin/env bash
                set -euo pipefail
                command -v aws >/dev/null
                aws sts get-caller-identity >/dev/null
                if [[ "${RUN_IMAGE_BUILD_PUSH}" == "true" ]]; then
                  registry_host="$(echo "${API_IMAGE_REPO}" | cut -d/ -f1)"
                  aws ecr get-login-password --region "${AWS_REGION:-us-east-1}" | docker login --username AWS --password-stdin "${registry_host}"
                fi
              '''
            }
          } else if (params.PROVIDER == 'azure') {
            withCredentials([
              usernamePassword(credentialsId: 'spot-scam-azure-sp', usernameVariable: 'AZURE_CLIENT_ID', passwordVariable: 'AZURE_CLIENT_SECRET'),
              string(credentialsId: 'spot-scam-azure-tenant', variable: 'AZURE_TENANT_ID'),
              string(credentialsId: 'spot-scam-azure-subscription', variable: 'AZURE_SUBSCRIPTION_ID')
            ]) {
              sh '''#!/usr/bin/env bash
                set -euo pipefail
                command -v az >/dev/null
                az login --service-principal -u "${AZURE_CLIENT_ID}" -p "${AZURE_CLIENT_SECRET}" --tenant "${AZURE_TENANT_ID}" >/dev/null
                az account set --subscription "${AZURE_SUBSCRIPTION_ID}"
                if [[ "${RUN_IMAGE_BUILD_PUSH}" == "true" ]]; then
                  acr_name="$(echo "${API_IMAGE_REPO}" | cut -d/ -f1 | cut -d. -f1)"
                  az acr login --name "${acr_name}"
                fi
              '''
            }
          } else if (params.PROVIDER == 'gcp') {
            withCredentials([
              file(credentialsId: 'spot-scam-gcp-sa', variable: 'GCP_SA_FILE')
            ]) {
              sh '''#!/usr/bin/env bash
                set -euo pipefail
                command -v gcloud >/dev/null
                gcloud auth activate-service-account --key-file="${GCP_SA_FILE}"
                gcloud config set project "${GCP_PROJECT_ID}" >/dev/null
                if [[ "${RUN_IMAGE_BUILD_PUSH}" == "true" ]]; then
                  registry_host="$(echo "${API_IMAGE_REPO}" | cut -d/ -f1)"
                  gcloud auth configure-docker "${registry_host}" --quiet
                fi
              '''
            }
          } else if (params.PROVIDER == 'oci') {
            if (params.RUN_IMAGE_BUILD_PUSH) {
              withCredentials([
                file(credentialsId: 'spot-scam-oci-config', variable: 'OCI_CONFIG_FILE'),
                usernamePassword(credentialsId: 'spot-scam-ocir', usernameVariable: 'OCIR_USERNAME', passwordVariable: 'OCIR_PASSWORD')
              ]) {
                sh '''#!/usr/bin/env bash
                  set -euo pipefail
                  command -v oci >/dev/null
                  export OCI_CLI_CONFIG_FILE="${OCI_CONFIG_FILE}"
                  oci os ns get >/dev/null
                  registry_host="$(echo "${API_IMAGE_REPO}" | cut -d/ -f1)"
                  echo "${OCIR_PASSWORD}" | docker login "${registry_host}" -u "${OCIR_USERNAME}" --password-stdin
                '''
              }
            } else {
              withCredentials([
                file(credentialsId: 'spot-scam-oci-config', variable: 'OCI_CONFIG_FILE')
              ]) {
                sh '''#!/usr/bin/env bash
                  set -euo pipefail
                  command -v oci >/dev/null
                  export OCI_CLI_CONFIG_FILE="${OCI_CONFIG_FILE}"
                  oci os ns get >/dev/null
                '''
              }
            }
          }
        }
      }
    }

    stage('Push Images') {
      when {
        expression { params.ACTION == 'deploy' && params.RUN_IMAGE_BUILD_PUSH }
      }
      steps {
        sh '''#!/usr/bin/env bash
          set -euo pipefail
          docker push "${API_IMAGE_REPO}:${EFFECTIVE_TAG}"
          docker push "${FRONTEND_IMAGE_REPO}:${EFFECTIVE_TAG}"

          if [[ "${PUSH_MODEL_IMAGE}" == "true" && -n "${MODEL_IMAGE_REPO}" ]]; then
            docker push "${MODEL_IMAGE_REPO}:${EFFECTIVE_TAG}" || true
          fi
        '''
      }
    }

    stage('Terraform Plan/Apply') {
      when {
        expression { params.ACTION == 'deploy' && params.RUN_TERRAFORM_APPLY }
      }
      steps {
        sh '''#!/usr/bin/env bash
          set -euo pipefail
          terraform -chdir="${TERRAFORM_DIR}" init -upgrade
          terraform -chdir="${TERRAFORM_DIR}" validate
          terraform -chdir="${TERRAFORM_DIR}" plan -out tfplan
        '''

        script {
          if (!params.AUTO_APPROVE_TERRAFORM) {
            input message: "Apply Terraform for ${params.PROVIDER} in ${env.TERRAFORM_DIR}?", ok: 'Apply'
          }
        }

        sh '''#!/usr/bin/env bash
          set -euo pipefail
          terraform -chdir="${TERRAFORM_DIR}" apply -auto-approve tfplan
        '''
      }
    }

    stage('Configure kubectl Context') {
      when {
        expression { params.ACTION == 'deploy' || params.ACTION == 'promote' || params.ACTION == 'abort' || params.ACTION == 'undo' }
      }
      steps {
        script {
          if (params.ACTION == 'deploy') {
            if (params.KUBECONFIG_COMMAND?.trim()) {
              sh """#!/usr/bin/env bash
                set -euo pipefail
                ${params.KUBECONFIG_COMMAND}
                kubectl get nodes
              """
            } else if (params.RUN_TERRAFORM_APPLY) {
              sh '''#!/usr/bin/env bash
                set -euo pipefail
                terraform -chdir="${TERRAFORM_DIR}" init -input=false -upgrade
                cfg_cmd="$(terraform -chdir="${TERRAFORM_DIR}" output -raw configure_kubectl)"
                if [[ -z "${cfg_cmd}" ]]; then
                  echo "configure_kubectl output is empty"
                  exit 1
                fi
                eval "${cfg_cmd}"
                kubectl get nodes
              '''
            } else {
              sh '''#!/usr/bin/env bash
                set -euo pipefail
                kubectl get ns "${NAMESPACE}" >/dev/null
                echo "Using existing kubectl context for deploy without terraform apply."
              '''
            }
          } else if (params.KUBECONFIG_COMMAND?.trim()) {
            sh """#!/usr/bin/env bash
              set -euo pipefail
              ${params.KUBECONFIG_COMMAND}
              kubectl get ns "${params.NAMESPACE}" >/dev/null
            """
          } else {
            sh '''#!/usr/bin/env bash
              set -euo pipefail
              kubectl get ns >/dev/null
              echo "Using existing kubectl context for manual rollout action."
            '''
          }
        }
      }
    }

    stage('Validate Overlay Rendering') {
      when {
        expression { params.ACTION == 'deploy' }
      }
      steps {
        sh '''#!/usr/bin/env bash
          set -euo pipefail
          kubectl kustomize "${OVERLAY_DIR}" >/tmp/spot_scam_rendered.yaml
          test -s /tmp/spot_scam_rendered.yaml
        '''
      }
    }

    stage('Preflight Deployment Checks') {
      when {
        expression { params.ACTION == 'deploy' }
      }
      steps {
        sh '''#!/usr/bin/env bash
          set -euo pipefail
          ./ops/ci/preflight_deploy_checks.sh \
            --provider "${PROVIDER}" \
            --strategy "${STRATEGY}" \
            --namespace "${NAMESPACE}"
        '''
      }
    }

    stage('Deploy Rollout') {
      when {
        expression { params.ACTION == 'deploy' }
      }
      steps {
        sh '''#!/usr/bin/env bash
          set -euo pipefail
          ./scripts/deploy_multi_cloud.sh --provider "${PROVIDER}" --strategy "${STRATEGY}" --namespace "${NAMESPACE}"

          kubectl -n "${NAMESPACE}" patch rollout spot-scam-api \
            --type='merge' \
            -p "{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"api\",\"image\":\"${API_IMAGE_REPO}:${EFFECTIVE_TAG}\"}]}}}}"

          kubectl -n "${NAMESPACE}" get rollout spot-scam-api -o wide
          argo-rollouts get rollout spot-scam-api -n "${NAMESPACE}" || true
        '''
      }
    }

    stage('Post Deploy Action') {
      when {
        expression { params.ACTION == 'deploy' && params.AUTO_PROMOTE_AFTER_DEPLOY }
      }
      steps {
        sh '''#!/usr/bin/env bash
          set -euo pipefail
          argo-rollouts promote spot-scam-api -n "${NAMESPACE}"
        '''
      }
    }

    stage('Manual Rollout Control') {
      when {
        expression { params.ACTION == 'promote' || params.ACTION == 'abort' || params.ACTION == 'undo' }
      }
      steps {
        sh '''#!/usr/bin/env bash
          set -euo pipefail
          case "${ACTION}" in
            promote)
              argo-rollouts promote spot-scam-api -n "${NAMESPACE}"
              ;;
            abort)
              argo-rollouts abort spot-scam-api -n "${NAMESPACE}"
              ;;
            undo)
              argo-rollouts undo spot-scam-api -n "${NAMESPACE}"
              ;;
            *)
              echo "Unsupported action: ${ACTION}" >&2
              exit 1
              ;;
          esac
        '''
      }
    }

    stage('Argo CD Sync') {
      when {
        expression { params.ACTION == 'argocd_sync' }
      }
      steps {
        script {
          if (params.ARGOCD_SERVER?.trim()) {
            sh '''#!/usr/bin/env bash
              set -euo pipefail
              ./ops/ci/argocd_sync_wait.sh \
                --app "${ARGOCD_APP_NAME}" \
                --timeout-sec "${ARGOCD_TIMEOUT_SEC}" \
                --server "${ARGOCD_SERVER}"
            '''
          } else {
            sh '''#!/usr/bin/env bash
              set -euo pipefail
              ./ops/ci/argocd_sync_wait.sh \
                --app "${ARGOCD_APP_NAME}" \
                --timeout-sec "${ARGOCD_TIMEOUT_SEC}"
            '''
          }
        }
      }
    }
  }

  post {
    always {
      echo "Build URL: ${env.BUILD_URL}"
      echo "Provider: ${params.PROVIDER}"
      echo "Action: ${params.ACTION}"
      echo "Strategy: ${params.STRATEGY}"
      echo "Namespace: ${params.NAMESPACE}"
      echo "Tag: ${env.EFFECTIVE_TAG ?: 'n/a'}"
      echo "ArgoCD App: ${params.ARGOCD_APP_NAME ?: 'n/a'}"
    }
    success {
      echo 'Pipeline completed successfully.'
    }
    failure {
      echo 'Pipeline failed. Review stage logs and rollout status before retrying.'
    }
  }
}
