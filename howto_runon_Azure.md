# Your Docker on Azure ML Deployment Guide
This guide provides step-by-step instructions for deploying Docker file ( I use the EDGS (Eliminating Densification for Gaussian Splatting for example) application to Azure Machine Learning using sensynml and sensyn-gym.

## Prerequisites

- Docker installed locally
- Azure CLI installed and configured
- Access to Azure Container Registry (ACR)
- Access to Azure ML workspace
- GPU-enabled compute cluster with managed identity in Azure ML

## Step 1: Build and Prepare Docker Image

### 1.1 Build the Docker Image

```bash
# Navigate to your project directory
cd /path/to/your/project

# Build the Docker image. In this example, I will use edgs-app.
docker build -t edgs-app:latest .
```

### 1.2 Verify the Image

```bash
# Check if the image was built successfully
docker images | grep edgs-app
```

Expected output:
```
edgs-app    latest    <IMAGE_ID>    <TIME>    34GB
```

## Step 2: Upload Docker Image to Azure Container Registry
ref: [sensynml](https://github.com/sensyn-robotics/sensynml)
### 2.1 Tag the Image for Azure Registry

```bash
# Tag the image with your Azure Container Registry URL
# Replace 'sensynmldev' with your ACR name
docker tag edgs-app:latest sensynmldev.azurecr.io/edgs-app:latest
```

### 2.2 Login to Azure Container Registry

```bash
# Login to ACR (this will likely require your vcuberobotics.onmicrosoft.com email address)
az login
az acr login --name sensynmldev
```

### 2.3 Push the Image to ACR

```bash
# Push the tagged image to ACR
docker push sensynmldev.azurecr.io/edgs-app:latest
```


## Step 3: Prepare Data for Azure ML

### 3.1 Data Upload Options

You have two options for data management:

#### Option A: Small Datasets (< 500MB)
- Include data directly with the project upload
- Place data in the `data/` directory

#### Option B: Large Datasets (> 500MB)
- Upload as Azure ML datasets
- Use Azure Storage Explorer or Azure ML Studio

### 3.2 Creating an Azure ML Dataset

#### Using Azure Storage Explorer:

1. Open Azure Storage Explorer
2. Navigate to your ML workspace storage account
3. Create a new container or use existing one (e.g. powergridcheckdev/blob Containers/dataset/)
4. Upload your data folder (e.g., `otowa360/images/`)
5. Note the path for dataset creation

#### Using Azure ML Studio:

1. Go to Azure ML Studio
2. Navigate to "Data" section
3. Click "Create dataset" → "From datastore"
4. Select your uploaded data location
5. Name your dataset (e.g., "ADA360")
6. Register the dataset

## Step 4: Configure sensyn-gym YAML

### 4.1 Create the gym Configuration File

Create a file.
In EDGS example, it's named `edgs-gym.yaml` with the following structure:

```yaml
experiment_name: "edgs"
workspace: "power-grid-check-dev-ml"  # Your Azure ML workspace name
environment: 
  name: "sensynmldev.azurecr.io/edgs-app"  # Your ACR image path
  version: "latest"
compute: "NC24ads-A100"  # Your GPU compute cluster name
command: "python script/fit_model_to_scene_full.py --config train_02_high_quality --output_path outputs/experiment_name"
data:  # optional - use for Azure ML datasets
  video_path: "ADA360"  # Your Azure ML dataset name
metadata:  # optional
  task: "gaussian_splatting"
  description: "EDGS training on Azure ML"
```

### 4.2 Configuration Parameters Explained

- **experiment_name**: Name for tracking in Azure ML
- **workspace**: Your Azure ML workspace name
- **environment**: 
  - `name`: Full ACR path to your Docker image
  - `version`: Image tag (usually "latest")
- **compute**: Name of your GPU compute cluster
  - Must have managed identity for ACR access
  - Recommended: NC24ads-A100 or similar GPU instances
- **command**: The training command to execute
- **data**: Maps to Azure ML datasets
  - Key names (e.g., `video_path`) become environment variables
  - Values are Azure ML dataset names

## Step 5: GPU Compute Cluster Requirements

### 5.1 Compute Cluster Configuration

Your GPU compute cluster must have:
- **Managed Identity**: "Power Grid Check Dev Compute Identity" (or equivalent)
- **GPU SKU**: Based on memory requirements:
  - 16GB+ GPU: NC24ads-A100, NC6s_v3
  - 8-12GB GPU: NC12s_v3, NC6s_v2
  - 4-8GB GPU: NC6, NV6

### 5.2 Verify Compute Access

```bash
# Check if compute has ACR access
az ml compute show --name <compute-name> --workspace-name <workspace-name>
```

## Step 6: Submit Job to Azure ML

### 6.a Using sensyn-gym CLI
ref: [sensyn-gym](https://github.com/sensyn-robotics/sensyn-gym)
```bash
# Submit the job using sensyn-gym
cd sensyn-gym 
poetry run gym <yourproject> -c <yourgym.yaml>
```
In EDGS case,
```bash
poetry run gym ../EDGS/ -c edgs-gym.yaml
```

### 6.b Using Azure ML CLI

```bash
# Alternative: Direct Azure ML submission
az ml job create --file edgs-gym.yaml --workspace-name <workspace-name>
```

## Step 7: Monitor and Retrieve Results

### 7.1 Monitor Job Progress

- Azure ML Studio: Navigate to "Experiments" → Your experiment name
- CLI: `az ml job show --name <job-name> --workspace-name <workspace-name>`

### 7.2 Download Results

```bash
# Download outputs after completion
az ml job download --name <job-name> --workspace-name <workspace-name> --output-path ./results
```

## Troubleshooting

### Common Issues and Solutions

1. **Docker push fails**
   - Ensure you're logged into ACR: `docker login <acr-name>.azurecr.io`
   - Check ACR permissions

2. **Compute cannot access Docker image**
   - Verify compute has managed identity attached
   - Check ACR access policies

3. **Out of Memory errors**
   - Use lower quality configuration
   - Reduce batch size in config
   - Use appropriate COLMAP settings

4. **Data not found**
   - Ensure dataset is registered in Azure ML
   - Check dataset name matches gym.yaml
   - Verify data path structure

5. **Job fails immediately**
   - Check compute cluster is running
   - Verify Docker image exists in ACR
   - Review command syntax in gym.yaml


## Best Practices

1. **Image Management**
   - Use specific version tags instead of "latest" for production
   - Keep image size optimized by cleaning unnecessary files
   - Multi-stage builds can reduce final image size

2. **Data Management**
   - Use Azure ML datasets for reproducibility
   - Version your datasets
   - Keep raw data separate from processed data

3. **Experiment Tracking**
   - Use meaningful experiment names
   - Log metrics and parameters
   - Save intermediate checkpoints

4. **Cost Optimization**
   - Use spot instances for non-critical training
   - Set appropriate job timeout limits
   - Clean up unused resources