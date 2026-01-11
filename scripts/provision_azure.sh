#!/usr/bin/env bash
# Provision Azure resources for the federated training demo
# Creates: Resource Group, ACR, Log Analytics workspace, Key Vault, AKS (with ACR integration)
# Usage: ./scripts/provision_azure.sh <RESOURCE_GROUP> <LOCATION> <ACR_NAME> <AKS_NAME>

set -euo pipefail

RG=${1:-rg-federated-ml}
LOCATION=${2:-eastus}
ACR_NAME=${3:-myacr$RANDOM}
AKS_NAME=${4:-aks-federated}
NODE_COUNT=${5:-3}
NODE_VM_SIZE=${6:-Standard_D2s_v3}
K8S_VERSION=${7:-}
LOG_ANALYTICS_NAME="la-${RG}"
KEYVAULT_NAME=${8:-kv-${RG}}

echo "Resource Group: $RG"
echo "Location: $LOCATION"
echo "ACR Name: $ACR_NAME"
echo "AKS Name: $AKS_NAME"

test_cmd() {
  command -v az >/dev/null 2>&1 || { echo "Azure CLI (az) not found. Install and login: az login"; exit 1; }
}

test_cmd

# 1) Create resource group
echo "Creating resource group..."
az group create --name "$RG" --location "$LOCATION"

# 2) Create ACR (Azure Container Registry)
echo "Creating ACR ($ACR_NAME)..."
az acr create --resource-group "$RG" --name "$ACR_NAME" --sku Standard --admin-enabled false

# 3) Create Log Analytics workspace
echo "Creating Log Analytics workspace ($LOG_ANALYTICS_NAME)..."
az monitor log-analytics workspace create --resource-group "$RG" --workspace-name "$LOG_ANALYTICS_NAME"
LA_ID=$(az monitor log-analytics workspace show --resource-group "$RG" --workspace-name "$LOG_ANALYTICS_NAME" --query id -o tsv)

# 4) Create Key Vault
echo "Creating Key Vault ($KEYVAULT_NAME)..."
az keyvault create --name "$KEYVAULT_NAME" --resource-group "$RG" --location "$LOCATION" --sku standard

# 5) Create AKS with managed identity and attach ACR
# If you want a specific k8s version, set K8S_VERSION env var before running
AKS_CREATE_CMD=(az aks create --resource-group "$RG" --name "$AKS_NAME" --node-count "$NODE_COUNT" --node-vm-size "$NODE_VM_SIZE" --enable-managed-identity --generate-ssh-keys --attach-acr "$ACR_NAME" --workspace-resource-id "$LA_ID")
if [[ -n "$K8S_VERSION" ]]; then
  AKS_CREATE_CMD+=(--kubernetes-version "$K8S_VERSION")
fi

echo "Creating AKS cluster (this can take 10-20 minutes)..."
"${AKS_CREATE_CMD[@]}"

# 6) (Optional) Add GPU node pool example
cat <<'EOF'
# To add a GPU node pool (example: NC-series), run:
# az aks nodepool add --resource-group $RG --cluster-name $AKS_NAME --name gpu --node-count 1 --node-vm-size Standard_NC6
EOF

# 7) Get kubeconfig
echo "Fetching kubeconfig for cluster..."
az aks get-credentials --resource-group "$RG" --name "$AKS_NAME" --overwrite-existing

# 8) Output ACR login server
ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --resource-group "$RG" --query loginServer -o tsv)
echo "ACR login server: $ACR_LOGIN_SERVER"

echo "Provisioning complete. Next steps:"
echo " - Build and push images to $ACR_LOGIN_SERVER"
echo " - Update k8s manifests in k8s/ to use $ACR_LOGIN_SERVER and apply them"
echo " - Store TLS certs and other secrets in Key Vault: $KEYVAULT_NAME"

echo "Example: build and push (replace <image> names):"
cat <<EOF
az acr login --name $ACR_NAME
docker build -f Dockerfile.server -t $ACR_LOGIN_SERVER/fed-skin-server:latest .
docker push $ACR_LOGIN_SERVER/fed-skin-server:latest

docker build -f Dockerfile.client -t $ACR_LOGIN_SERVER/fed-skin-client:latest .
docker push $ACR_LOGIN_SERVER/fed-skin-client:latest
EOF

# 9) Optional: create service principal for CI/CD to push images and deploy
cat <<'EOF'
# To create a Service Principal for CI/CD (store the output safely):
# az ad sp create-for-rbac --name "http://sp-$ACR_NAME-ci" --role contributor --scopes /subscriptions/<SUBSCRIPTION_ID>/resourceGroups/$RG
# For pushing to ACR, it's preferable to assign 'AcrPush' scoped to the ACR resource.
EOF

exit 0
