# SandBoxFusion Deployment Guide

This guide provides step-by-step instructions to deploy the SandBoxFusion service using Docker containers across multiple nodes.

Currently we deploy SandBoxFusion with Docker Swarm, which will deploy the service on all nodes. Please follow the steps below to set up and launch the service.

## Prerequisites

- Docker installed on all nodes.
- SSH access to all nodes.
- `NODE_IP_LIST` in environment variables: A list of node IPs where the service will be deployed.
- `IMAGE_NAME` in environment variables: The name of the Docker image to be used for deployment.

## Deployment Steps

1. **Build the Docker Image**

    Ensure you have built the Docker image for the SandBoxFusion service. You can do this by following the guide in [SandBoxFusion](https://github.com/bytedance/SandboxFusion) to build the code sandbox server image.

2. **Prepare the Launch Script**

    Use the provided `launch_service.sh` script to automate the deployment process. Make sure to set the `NODE_IP_LIST` and `IMAGE_NAME` environment variables before running the script.

3. **Launch the Service**

    Run the `launch_service.sh` script on the master node:

    ```bash
    bash launch_service.sh
    ```

    This script will:
    - Check and start docker service on all nodes.
    - Initialize docker swarm on the master node.
    - Join worker nodes to the swarm.
    - Deploy the SandBoxFusion service using the specified Docker image.

4. **Validate the Deployment**

    After deployment, you can execute the following command in the shell to request the sandbox to run a Python code snippet in each node:

    ```bash
    curl 'http://localhost:8080/run_code' \
        -H 'Content-Type: application/json' \
        --data-raw '{"code": "print(\"Hello, world!\")", "language": "python"}'
    ```

    Sample output:

    ```json
    {
        "status": "Success",
        "message": "",
        "compile_result": null,
        "run_result": {
            "status": "Finished",
            "execution_time": 0.016735315322875977,
            "return_code": 0,
            "stdout": "Hello, world!\n",
            "stderr": ""
        },
        "executor_pod_name": null,
        "files": {}
    }
    ```

5. **(Optional) Clean the Service**:

    If you need to remove the deployed service, you can run the following command on the master node, which will use pssh to stop and remove the service from all nodes:

    ```bash
    bash clear_service.sh
    ```

## Notes

As a helper tool, you can use `scripts/docker/docker_manager.sh` in PSRL repository to manager docker across multiple nodes. It provides functionalities such as starting/stopping/restarting docker service, checking docker status and viewing docker logs.
