pipeline {
  agent any
  environment {
    IMAGE = keshvi123/plant-disease-app"
    TAG   = "${BUILD_NUMBER}"
  }
  stages {
    stage('Checkout') {
      steps { checkout scm }
    }
    stage('Build & Test') {
      steps {
        sh '''
          python3 -m venv .venv
          . .venv/bin/activate
          pip install --quiet pytest
          python -m pytest -q tests
        '''
      }
    }
    stage('Docker Build') {
      steps { sh 'docker build -t $IMAGE:$TAG -t $IMAGE:latest .' }
    }
    stage('Push Image') {
      steps {
        withCredentials([usernamePassword(credentialsId: 'dockerhub-creds',
                         usernameVariable: 'DH_USER', passwordVariable: 'DH_PASS')]) {
          sh '''
            echo "$DH_PASS" | docker login -u "$DH_USER" --password-stdin
            docker push $IMAGE:$TAG
            docker push $IMAGE:latest
          '''
        }
      }
    }
    stage('Deploy to Minikube') {
      steps {
        sh '''
          sed "s|IMAGE_PLACEHOLDER|$IMAGE:$TAG|" k8s/deployment.yaml | kubectl apply -f -
          kubectl apply -f k8s/service.yaml
          kubectl rollout status deployment/plant-app --timeout=600s
        '''
      }
    }
    stage('Verify') {
      steps {
        sh '''
          kubectl get pods
          kubectl get deployment plant-app
          kubectl get svc plant-service
          curl -s http://$(minikube ip):30007/_stcore/health
        '''
      }
    }
  }
}
