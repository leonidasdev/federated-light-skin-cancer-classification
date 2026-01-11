Kubernetes manifests for the federated training demo

Usage (basic):

1. Create namespace and config:

   kubectl apply -f namespace.yaml
   kubectl apply -f configmap.yaml -n federated-ml
   kubectl apply -f secret-template.yaml -n federated-ml   # edit placeholders first

2. Deploy server (replace <ACR_NAME> in manifests with your registry):

   kubectl apply -f server-deployment.yaml -n federated-ml
   kubectl apply -f server-service.yaml -n federated-ml

Build & push (example):

   az acr login --name <ACR_NAME>
   docker build -f server/Dockerfile -t <ACR_NAME>.azurecr.io/fed-skin-server:latest .
   docker push <ACR_NAME>.azurecr.io/fed-skin-server:latest

3. Deploy clients (scale `replicas` in `client-deployment.yaml` as needed):

   kubectl apply -f client-deployment.yaml -n federated-ml

Build & push client (example):

   az acr login --name <ACR_NAME>
   docker build -f client/Dockerfile -t <ACR_NAME>.azurecr.io/fed-skin-client:latest .
   docker push <ACR_NAME>.azurecr.io/fed-skin-client:latest

Notes:
- For private ACR, create image pull secret or use AAD integration.
- Replace placeholders and set secrets before applying manifests.
- Consider using `kubectl set image` to update images during CI/CD.
