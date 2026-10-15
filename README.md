# Smart Plant Disease Identifier: CI/CD Pipeline (CCA 2)

Streamlit web app that identifies plant diseases from a leaf photo
(MobileNetV2 model, 38 classes). It is built, tested, packaged and deployed
automatically with GitHub, Jenkins, Docker, Docker Hub and Kubernetes (Minikube) on AWS EC2.

## Project
Existing academic project: Smart Plant Disease Identification System
(Streamlit + TensorFlow/Keras). Port 8501.

## Repository contents
- `app.py`, `advice.py`, `gradcam.py`, `class_names.json`, `plant_disease_mobilenet.keras`: application
- `tests/test_basic.py`: tests used in the Build & Test stage
- `Dockerfile`, `requirements.txt`: container image (python:3.11-slim)
- `Jenkinsfile`: declarative pipeline
- `k8s/deployment.yaml`, `k8s/service.yaml`: Kubernetes Deployment and NodePort Service

## Pipeline stages (Jenkins job: plant-disease-pipeline)
1. **Checkout**: Jenkins pulls the latest code from GitHub (`main`).
2. **Build & Test**: creates a Python virtual environment and runs pytest
   (syntax check, 38 classes in class_names.json, model file present, advice text for every condition).
3. **Docker Build**: builds the image, tagged with the build number and `latest`.
4. **Push Image**: logs in with the `dockerhub-creds` credential stored in Jenkins and pushes
   `keshvi123/plant-disease-app` to Docker Hub.
5. **Deploy to Minikube**: applies the Deployment (image tag filled in with `sed`) and the Service,
   then waits for `kubectl rollout status`.
6. **Verify**: lists pods, deployment and service, and checks the app health endpoint.

## Jenkins on AWS
- EC2 instance (Amazon Linux 2023, m7i-flex.large, ap-south-1, 30 GiB disk).
- Security group ports: 22 (SSH), 8080 (Jenkins), 8501 (app).
- Installed: Docker, Git, Python 3, Java 21 (Corretto), Jenkins, kubectl, Minikube (Docker driver, run as the `jenkins` user).
- Jenkins plugins: Pipeline, Git, Credentials Binding. Docker Hub login stored as the global credential `dockerhub-creds`.
- Job: Pipeline script from SCM, repo above, branch `main`, script path `Jenkinsfile`, Poll SCM `H/2 * * * *`.

## Image push
Jenkins logs in to Docker Hub with the stored token and pushes `keshvi123/plant-disease-app:<build number>`
and `:latest`.

## Deployment and access
Kubernetes pulls the image from Docker Hub and runs it in Minikube (NodePort 30007).
To open the app from a browser:
`kubectl port-forward --address 0.0.0.0 svc/plant-service 8501:80`, then visit `http://<server-ip>:8501`.

## Code change for the second run
The subtitle text in `app.py` was changed (added "(version 2)"), pushed to GitHub, and Jenkins
built and deployed it as a new image tag. The deployment image changed from `:6` to `:7`.
