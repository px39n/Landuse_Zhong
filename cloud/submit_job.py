"""
Submit cloud job to Google Cloud Platform
Supports Cloud Run, Vertex AI, or Compute Engine
"""

import argparse
import yaml
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def load_config(config_path: str) -> Dict:
    """Load pipeline configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def submit_cloud_run_job(
    stage: str,
    config: Dict,
    machine_type: str = "n1-standard-4",
    region: str = "us-central1"
) -> str:
    """
    Submit job to Cloud Run
    
    Args:
        stage: Pipeline stage to run
        config: Configuration dictionary
        machine_type: GCP machine type
        region: GCP region
    
    Returns:
        Job ID
    """
    gcs_config = config["gcs"]
    project = gcs_config.get("project")
    image = config["cloud_job"]["docker"]["image"]
    
    job_name = f"landuse-{stage}-{int(__import__('time').time())}"
    
    cmd = [
        "gcloud", "run", "jobs", "create", job_name,
        "--image", image,
        "--region", region,
        "--project", project,
        "--execute-now",
        "--args", f"pipelines/global/{stage}.py",
        "--args", "--config",
        "--args", "configs/cloud.yaml"
    ]
    
    print(f"Submitting Cloud Run job: {job_name}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    
    print(f"Job submitted: {job_name}")
    print(result.stdout)
    
    return job_name


def submit_vertex_ai_job(
    stage: str,
    config: Dict,
    machine_type: str = "n1-standard-8",
    gpu: bool = False,
    gpu_type: str = "nvidia-tesla-t4",
    gpu_count: int = 1,
    region: str = "us-central1"
) -> str:
    """
    Submit training job to Vertex AI
    
    Args:
        stage: Pipeline stage to run
        config: Configuration dictionary
        machine_type: GCP machine type
        gpu: Whether to use GPU
        gpu_type: GPU type
        gpu_count: Number of GPUs
        region: GCP region
    
    Returns:
        Job ID
    """
    gcs_config = config["gcs"]
    project = gcs_config.get("project")
    bucket = gcs_config["bucket"]
    prefix = gcs_config["prefix"]
    image = config["cloud_job"]["docker"]["image"]
    
    job_name = f"landuse-{stage}-{int(__import__('time').time())}"
    
    # Build Vertex AI job spec
    job_spec = {
        "displayName": job_name,
        "jobSpec": {
            "workerPoolSpecs": [
                {
                    "machineSpec": {
                        "machineType": machine_type
                    },
                    "replicaCount": 1,
                    "containerSpec": {
                        "imageUri": image,
                        "args": [
                            f"pipelines/global/{stage}.py",
                            "--config",
                            "configs/cloud.yaml"
                        ]
                    }
                }
            ]
        }
    }
    
    # Add GPU if requested
    if gpu:
        job_spec["jobSpec"]["workerPoolSpecs"][0]["machineSpec"]["acceleratorType"] = gpu_type
        job_spec["jobSpec"]["workerPoolSpecs"][0]["machineSpec"]["acceleratorCount"] = gpu_count
    
    # Save job spec to temp file
    job_spec_path = f"/tmp/{job_name}_spec.json"
    with open(job_spec_path, 'w') as f:
        json.dump(job_spec, f, indent=2)
    
    cmd = [
        "gcloud", "ai", "custom-jobs", "create",
        "--region", region,
        "--project", project,
        "--display-name", job_name,
        "--config", job_spec_path
    ]
    
    print(f"Submitting Vertex AI job: {job_name}")
    print(f"Machine: {machine_type}, GPU: {gpu_type if gpu else 'None'}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    
    print(f"Job submitted: {job_name}")
    print(result.stdout)
    
    return job_name


def submit_batch_jobs(
    stage: str,
    config: Dict,
    tiles: List[str],
    max_workers: int = 10,
    **kwargs
) -> List[str]:
    """
    Submit batch jobs for multiple tiles
    
    Args:
        stage: Pipeline stage to run
        config: Configuration dictionary
        tiles: List of tile IDs
        max_workers: Maximum parallel workers
        **kwargs: Additional arguments for job submission
    
    Returns:
        List of job IDs
    """
    job_ids = []
    
    print(f"Submitting {len(tiles)} jobs for stage {stage}")
    print(f"Max parallel workers: {max_workers}")
    
    for i, tile in enumerate(tiles):
        print(f"\n[{i+1}/{len(tiles)}] Submitting job for tile: {tile}")
        
        # Modify config to include tile ID
        config_copy = config.copy()
        config_copy["tiling"] = config_copy.get("tiling", {})
        config_copy["tiling"]["current_tile"] = tile
        
        # Submit job
        job_id = submit_vertex_ai_job(stage, config_copy, **kwargs)
        job_ids.append(job_id)
        
        # Throttle to avoid rate limits
        if (i + 1) % max_workers == 0 and i < len(tiles) - 1:
            print(f"\nWaiting for batch to complete...")
            import time
            time.sleep(60)  # Wait 1 minute between batches
    
    print(f"\nSubmitted {len(job_ids)} jobs")
    return job_ids


def main():
    parser = argparse.ArgumentParser(description="Submit cloud job")
    parser.add_argument("--stage", required=True, help="Pipeline stage to run")
    parser.add_argument("--config", default="configs/cloud.yaml", help="Config file")
    parser.add_argument("--backend", default="vertex-ai", choices=["cloud-run", "vertex-ai"],
                       help="Cloud backend to use")
    parser.add_argument("--machine-type", default="n1-standard-8", help="Machine type")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--gpu-type", default="nvidia-tesla-t4", help="GPU type")
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    parser.add_argument("--tiles", help="Comma-separated tile IDs for batch processing")
    parser.add_argument("--max-workers", type=int, default=10, help="Max parallel workers")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Check if batch mode
    if args.tiles:
        tiles = args.tiles.split(',')
        job_ids = submit_batch_jobs(
            args.stage,
            config,
            tiles,
            max_workers=args.max_workers,
            machine_type=args.machine_type,
            gpu=args.gpu,
            gpu_type=args.gpu_type,
            gpu_count=args.gpu_count,
            region=args.region
        )
        print(f"\nBatch submission complete. Job IDs:")
        for job_id in job_ids:
            print(f"  - {job_id}")
    else:
        # Single job
        if args.backend == "cloud-run":
            job_id = submit_cloud_run_job(
                args.stage,
                config,
                machine_type=args.machine_type,
                region=args.region
            )
        else:  # vertex-ai
            job_id = submit_vertex_ai_job(
                args.stage,
                config,
                machine_type=args.machine_type,
                gpu=args.gpu,
                gpu_type=args.gpu_type,
                gpu_count=args.gpu_count,
                region=args.region
            )
        
        print(f"\nJob ID: {job_id}")
        print(f"\nMonitor job:")
        print(f"  gcloud ai custom-jobs describe {job_id} --region {args.region}")


if __name__ == "__main__":
    main()
