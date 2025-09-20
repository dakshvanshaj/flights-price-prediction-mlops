# MLflow Integration and Deployment

This document details the integration of MLflow for experiment tracking and model management, covering both client-side connection and server-side deployment on AWS.

## 1. Connecting a Client Application to MLflow

This section explains how a client application (e.g., a prediction API running in a Docker container) securely connects to a remote MLflow server to fetch models and artifacts.

### 1.1. Connection Mechanism: A Two-Step Process

Client connection occurs in two distinct stages, controlled by specific environment variables.

1.  **Initial Connection to the Tracking Server:**
    *   The client application requires the address of the MLflow tracking server running on the EC2 instance.
    *   This is controlled by the `MLFLOW_TRACKING_URI` environment variable.
    *   The MLflow client library automatically uses this variable. Upon application startup and model loading, MLflow initiates a network request to this URI to communicate with the server.

2.  **Fetching Artifacts from the S3 Bucket:**
    *   Once connected, the tracking server provides the client application with the location of the model files, typically an S3 bucket path.
    *   The client application then requires permissions to access S3, which are provided via AWS credentials.
    *   These are controlled by the `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_DEFAULT_REGION` environment variables.

### 1.2. Automatic Credential Detection

Installation of the AWS CLI or execution of `aws configure` inside the Docker container is not required.

The AWS SDK for Python (`boto3`), utilized by MLflow, automatically searches for credentials in a standard sequence, prioritizing environment variables.

When `boto3` detects `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in the container's environment, it automatically uses them for authentication with AWS services like S3.

### 1.3. Production Best Practice: Using an Environment File

Providing these variables to a Docker container via an environment file (e.g., `prediction_app.env`) is the most secure and standard method. This file must **never be committed to version control**.

**1. Create the `.env` file:**
```env
# ----------------------------------
# MLflow Production Configuration
# ----------------------------------
# The public IP or domain of your EC2 instance running the MLflow server.
MLFLOW_TRACKING_URI="http://<YOUR_EC2_INSTANCE_IP>:5000"

# ----------------------------------
# AWS Credentials for MLflow Artifacts
# ----------------------------------
# Credentials for an IAM user with read-only access to your S3 artifact bucket.
AWS_ACCESS_KEY_ID="<YOUR_AWS_ACCESS_KEY_ID>"
AWS_SECRET_ACCESS_KEY="<YOUR_AWS_SECRET_ACCESS_KEY>"
AWS_DEFAULT_REGION="<YOUR_S3_BUCKET_REGION>"
```

**2. Run the container with the `--env-file` flag:**
This injects the variables into the container at runtime.
```bash
docker run --env-file ./src/prediction_server/prediction_app.env -p 8000:8000 your-image-name
```

## 2. Deploying an MLflow Tracking Server on AWS

This section provides a step-by-step guide to setting up a robust and scalable MLflow Tracking Server using AWS services: EC2 (for the server), S3 (for artifact storage), and RDS (for the backend database).

### 2.1. Step 1: Create an S3 Bucket for Artifacts

S3 is the initial component due to its simplicity and standalone nature, serving as the central artifact store.

1.  **Navigate to the S3 Service**
    *   Log in to your AWS account.
    *   In the console search bar, type **`S3`** and select the service.
2.  **Create the Bucket**
    *   On the S3 dashboard, click **"Create bucket"**.
    *   Configure the following:
        *   **Bucket name**: Must be **globally unique**. A unique prefix is recommended (e.g., `yourname-mlflow-artifacts-2025`).
        *   **AWS Region**: Select a region close to your location to minimize latency (e.g., **Asia Pacific (Mumbai) `ap-south-1`**). **Important**: All other services (EC2, RDS) should be launched in this *same region*.
        *   **Block Public Access settings**: Keep **"Block all public access"** checked for security. The MLflow server will access the bucket using secure credentials.
        *   **Bucket Versioning**: **Enable** this to maintain object history and protect against accidental data loss.
        *   **Tags (Optional but recommended)**: Add a tag (e.g., `Key: Project`, `Value: mlflow-server`) for cost tracking.
        *   **Default encryption**: Retain the default (`Amazon S3-managed keys (SSE-S3)`).
3.  **Finalize Creation**
    *   Review settings and estimated costs.
    *   Click **"Create bucket"**.

This completes the setup of a secure, scalable, and durable location for ML experiment artifacts.

Next, the managed PostgreSQL database will be created using **AWS RDS**, serving as the metadata store for the MLflow server.

### 2.2. Step 2: Create a PostgreSQL Database with AWS RDS

1.  **Navigate to the RDS Service**
    *   In the AWS Console search bar, type **`RDS`** and select it.
2.  **Create the Database**
    *   On the RDS dashboard, click **"Create database"**.
    *   Follow these settings in the creation wizard:
        *   **Choose a database creation method**: Select **"Standard Create"**.
        *   **Engine options**: Choose **"PostgreSQL"**.
        *   **Templates**: Select the **"Free tier"** template for cost efficiency.
        *   **Settings**:
            *   **DB instance identifier**: Provide a name (e.g., `mlflow-db`).
            *   **Master username**: Choose a username (e.g., `mlflow_user`).
            *   **Master password**: Create a strong password and **store it securely**. This will be required later.
        *   **DB instance size**: The free tier will pre-select an instance type (e.g., `db.t3.micro`), which is suitable.
        *   **Storage**: Default free tier settings are sufficient.
        *   **Connectivity**:
            *   **Public access**: Select **"No"** for security. The database will only be accessible from within the private AWS network.
            *   **VPC security group**: Choose **"Create new"** and name it (e.g., `mlflow-db-security-group`). This acts as a firewall for the database.
        *   **Database options** (under "Additional configuration"):
            *   **Initial database name**: Enter `mlflow_db`. This is crucial for automatic database creation.
3.  **Finalize Creation**
    *   Review settings and estimated monthly costs.
    *   Click **"Create database"**.

The database creation process typically takes 10-15 minutes. Upon completion, a professional-grade database will be ready to serve as the MLflow server's metadata store.

### 2.3. Step 3: Launch an EC2 Virtual Server

This EC2 instance will serve as the MLflow server, connecting to the RDS database for metadata and the S3 bucket for artifacts.

1.  **Navigate to the EC2 Service**
    *   In the AWS Console search bar, type **`EC2`** and select it.
2.  **Launch a New Instance**
    *   Click **"Launch instance"**.
    *   Configure the following:
        *   **Name**: Provide a clear name (e.g., `mlflow-server`).
        *   **Application and OS Images (AMI)**: Select **Ubuntu** (Free tier eligible). The latest Ubuntu Server LTS version is recommended.
        *   **Instance type**: Choose **`t2.micro`** (Free Tier eligible).
        *   **Key pair (for login)**: For secure SSH access:
            *   Click **"Create new key pair"**.
            *   Name it (e.g., `mlflow-key`).
            *   Keep defaults (`RSA`, `.pem`).
            *   Click **"Create key pair"**. The `.pem` file will download. **Store this file securely; it cannot be re-downloaded.**
        *   **Network settings**:
            *   Click **"Edit"**.
            *   **VPC**: Use the default VPC.
            *   **Subnet**: No specific preference is required.
            *   **Firewall (security groups)**: Create a new security group and configure these rules:
                1.  **Rule 1 (SSH)**: Allow SSH traffic. For enhanced security, restrict "Source type" to "My IP" instead of "Anywhere" (`0.0.0.0/0`).
                2.  **Rule 2 (HTTP)**: Allow HTTP traffic from "Anywhere" (`0.0.0.0/0`).
                3.  **Rule 3 (Custom TCP)**: Allow custom TCP traffic on port **`5000`** from "Anywhere" (`0.0.0.0/0`). This is the MLflow application port.
            *   Name the security group (e.g., `mlflow-server-sg`).
        *   **Configure storage**: The default 8 GB is sufficient.
        *   **Advanced details**: Can be left as default.
3.  **Launch the Instance**
    *   Review the summary.
    *   Click **"Launch instance"**.

The instance will launch within a few minutes. Once its status is "Running" in the EC2 dashboard, all infrastructure components will be ready for connection.

### 2.4. Step 4: Connecting the Parts

#### 2.4.1. Connect EC2 and RDS Security Groups

Establish a firewall rule to allow secure communication between your EC2 instance and RDS database over the private AWS network.

1.  Navigate to the **RDS** dashboard, select **"Databases"**, and choose your `mlflow-db`.
2.  Go to the **"Connectivity & security"** tab.
3.  Under "Security", click on the active **VPC security group** (e.g., `mlflow-db-security-group`).
4.  In the EC2 security group console, select the group, go to the **"Inbound rules"** tab, and click **"Edit inbound rules"**.
5.  Click **"Add rule"** and configure:
    *   **Type**: Select **PostgreSQL** (port `5432` will auto-fill).
    *   **Source**: Type and select your EC2 security group (e.g., `mlflow-server-sg`).
6.  Click **"Save rules"**. This rule permits connections to the database only from servers within the `mlflow-server-sg` group.

#### 2.4.2. Connect to Your EC2 Instance

Log in to the newly created EC2 server.

1.  Go to the **EC2** dashboard and select your `mlflow-server` instance.
2.  Copy the **"Public IPv4 address"**.
3.  Open a terminal. Make your key file private:
    ```bash
    chmod 400 /path/to/your/mlflow-key.pem
    ```
4.  Connect via SSH, replacing placeholders with your key file's path and the server's public IP:
    ```bash
    ssh -i /path/to/your/mlflow-key.pem ubuntu@YOUR_PUBLIC_IP_ADDRESS
    # Example: ssh -i flights-mlflow-key.pem ubuntu@65.2.142.127
    ```
5.  Type `yes` if prompted to confirm the connection.

#### 2.4.3. Creating and Attaching an IAM Role (for S3 Access)

Grant your EC2 instance the necessary permissions to interact with your S3 artifact bucket.

1.  **Create an IAM Policy**: Define permissions for S3 access.
    *   Go to the **IAM** service in AWS.
    *   Navigate to **"Policies"** > **"Create policy"**.
    *   Use the visual editor:
        *   **Service**: Choose **S3**.
        *   **Actions**: Allow `ListBucket`, `GetObject`, `PutObject`, and `DeleteObject`.
        *   **Resources**: Specify the ARN (Amazon Resource Name) for your bucket to restrict access. Include both the bucket itself (`arn:aws:s3:::your-bucket-name`) and all objects within it (`arn:aws:s3:::your-bucket-name/*`).

        **Explanation of S3 ARNs:**
        S3 distinguishes between the bucket and its objects. Permissions must be granted for both levels.
        1.  **`arn:aws:s3:::your-bucket-name`**: For `ListBucket` permission.
        2.  **`arn:aws:s3:::your-bucket-name/*`**: For `GetObject`, `PutObject`, `DeleteObject` permissions.

        To add these ARNs in the IAM policy editor:
        1.  In the "Resources" section, select **"Specific"**.
        2.  Click **"Add ARN"** under "bucket". Enter your S3 bucket name (e.g., `yourname-mlflow-artifacts-2025`). Click **"Add ARN"**.
        3.  Click **"Add ARN"** again. Check **"Any"** for "Object name" to add the `/*` wildcard. Click **"Add ARN"**.
    *   Name the policy (e.g., `MLflowS3AccessPolicy`).
2.  **Create an IAM Role**:
    *   In IAM, go to **"Roles"** > **"Create role"**.
    *   **Trusted entity type**: Select **"AWS service"**.
    *   **Use case**: Select **"EC2"**.
    *   Attach the `MLflowS3AccessPolicy` created previously.
    *   Name the role (e.g., `MLflowEC2Role`).
3.  **Attach the Role to Your EC2 Instance**:
    *   In the **EC2** dashboard, select your `mlflow-server` instance.
    *   Go to **"Actions"** > **"Security"** > **"Modify IAM role"**.
    *   Select the `MLflowEC2Role` and save.

Your EC2 instance now has the necessary permissions to read and write to your S3 bucket.

#### 2.4.4. Best Practices for Ubuntu Server Setup

Perform initial setup steps on your new Ubuntu server for enhanced security and manageability.

1.  **Update Your System**
    ```bash
    sudo apt update
    sudo apt upgrade -y
    ```
2.  **Create a New User**
    Create a new user account with administrative privileges for daily work, avoiding the default `ubuntu` or `root` user.
    ```bash
    # Replace 'your_username' with a chosen name
    sudo adduser your_username
    ```
    Set a password and optional information when prompted. Grant `sudo` privileges:
    ```bash
    sudo usermod -aG sudo your_username
    ```
    *   `-a` (or `--append`): Appends specified groups to the user's existing supplementary groups.
    *   `-G` (or `--groups`): Specifies groups to add the user to (e.g., `sudo`).

    Log out and log back in as the new user: `ssh -i /path/to/key.pem your_username@YOUR_PUBLIC_IP`

3.  **Set Up a Basic Firewall**
    Enable `ufw` (Uncomplicated Firewall) to allow only necessary connections (SSH, HTTP, and MLflow port).
    ```bash
    # Allow SSH connections
    sudo ufw allow OpenSSH

    # Allow HTTP and MLflow port
    sudo ufw allow 80/tcp
    sudo ufw allow 5000/tcp

    # Enable the firewall
    sudo ufw enable
    ```
    The firewall will now block other incoming connections, enhancing security.

### 2.5. Step 5: Install MLflow Software

Proceed with MLflow application setup on the secured server.

1.  **Install basic tools**
    ```bash
    sudo apt update
    sudo apt install python3-pip python3-venv -y
    ```
2.  **Create a virtual environment and install MLflow**
    ```bash
    python3 -m venv mlflow-env
    source mlflow-env/bin/activate
    pip install mlflow boto3 psycopg2-binary
    ```
    `boto3` (AWS SDK for Python) is essential for MLflow to interact with AWS services like S3 for artifact storage.

### 2.6. Step 6: Launch the MLflow Server

This command integrates all components. Required information includes your **RDS Endpoint**, **RDS Password**, and **S3 Bucket Name**.

**Determining the SQLAlchemy Connection String Format:**
1.  **Identify Standard**: MLflow documentation specifies a "SQLAlchemy-compatible string."
2.  **Search Standard**: Search for "SQLAlchemy PostgreSQL connection string format."
3.  **Find Pattern**: The standard format for PostgreSQL is:
    `postgresql://<user>:<password>@<host>:<port>/<database>`
4.  **Map to AWS Setup**: Map your RDS information to this pattern:
    *   `<user>`: `mlflow_user`
    *   `<password>`: `YOUR_RDS_PASSWORD`
    *   `<host>`: `YOUR_RDS_ENDPOINT`
    *   `<port>`: `5432` (default for PostgreSQL)
    *   `<database>`: `mlflow_db`

Resulting URI:
`postgresql://mlflow_user:YOUR_RDS_PASSWORD@YOUR_RDS_ENDPOINT/mlflow_db`

Execute the following command, replacing placeholders with your specific values.

```bash
mlflow server \
    --backend-store-uri postgresql://mlflow_user:YOUR_RDS_PASSWORD@YOUR_RDS_ENDPOINT/mlflow_db \
    --default-artifact-root s3://your-s3-bucket-name/ \
    --host 0.0.0.0 \
    --port 5000
```

*   **`--host 0.0.0.0` Explanation:** This networking term instructs the MLflow server to listen on **all available network interfaces**, including its public IP address, making it reachable from a web browser.

Upon server startup, the MLflow UI will be accessible at `http://YOUR_EC2_PUBLIC_IP:5000` in your web browser.

To maintain server operation after closing the terminal, run it within a `screen` session:
```bash
# Start a new screen session
screen -S mlflow

# Activate environment and run the mlflow server command
source mlflow-env/bin/activate
mlflow server --backend-store-uri postgresql://mlflow_user:YOUR_RDS_PASSWORD@YOUR_RDS_ENDPOINT/mlflow_db --default-artifact-root s3://your-s3-bucket-name/ --host 0.0.0.0 --port 5000

# Detach from the session by pressing Ctrl+A, then D.
```

### 2.7. Step 7: Local Setup

#### 2.7.1. Connecting Your Local Project

Project code is not required on the EC2 server. Local Python ML code can log results to the remote server.

Configure your **local machine**:

*   **Set the Tracking URI**: Define the MLflow server address via an environment variable.
    ```bash
    # In your local terminal (macOS/Linux)
    export MLFLOW_TRACKING_URI="http://YOUR_EC2_PUBLIC_IP:5000"
    ```
    MLflow scripts executed in this terminal session will automatically log to the remote server.

*   **Configure AWS Credentials**: Grant your local machine permissions to upload artifacts to your S3 bucket. Install and configure the AWS CLI:

#### 2.7.2. Step-by-Step Guide AWS CLI

1.  **Install the AWS CLI**
    Install the command-line tool on your local machine:
    ```bash
    pip install awscli
    # or for conda:
    # conda install -c conda-forge awscli
    # if incompatible, create a new environment:
    # conda create -n aws-tools -c conda-forge awscli python=3.13
    ```
2.  **Run the Configure Command**
    Execute in your terminal:
    ```bash
    aws configure
    ```
3.  **Enter Your Credentials**
    Provide the four requested pieces of information. The first two are obtained by creating an IAM user with S3 access permissions in your AWS account.
    *   **AWS Access Key ID**: Paste your access key.
    *   **AWS Secret Access Key**: Paste your secret key (text will be hidden).
    *   **Default region name**: Enter your S3 bucket's region (e.g., `ap-south-1`).
    *   **Default output format**: Press Enter for default (`json`).

    Example process:
    ```bash
    $ aws configure
    AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
    AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
    Default region name [None]: ap-south-1
    Default output format [None]: json
    ```
    The CLI securely stores credentials in a `.aws` folder. MLflow will automatically detect and use them for S3 access.

Your local `train.py` script will now send results to your production server without code changes.

### 2.8. Step 8: Persistent Server Operation (Systemd)

To ensure the MLflow server runs continuously as a **background service**, configure `systemd` on Ubuntu. This enables automatic startup on boot and restarts upon crashes.

1.  **SSH into your EC2 server**.
2.  **Create a `systemd` service file**
    ```bash
    sudo nano /etc/systemd/system/mlflow-server.service
    ```
3.  **Paste the following configuration**, replacing placeholders with your database URI, S3 bucket, and ensuring `ExecStart` points to the `mlflow` executable within your virtual environment.
    ```toml
    [Unit]
    Description=MLflow Tracking Server
    After=network.target

    [Service]
    User=daksh_linux
    Restart=on-failure
    ExecStart=/home/daksh_linux/mlflow-env/bin/mlflow server \
        --backend-store-uri postgresql://mlflow_user:YOUR_RDS_PASSWORD@YOUR_RDS_ENDPOINT/mlflow_db \
        --default-artifact-root s3://your-s3-bucket-name/ \
        --host 127.0.0.1 \
        --port 5000

    [Install]
    WantedBy=multi-user.target
    ```
    *Note: `--host` is set to `127.0.0.1` for enhanced security, accepting connections only from the server itself (a reverse proxy like Nginx can handle public traffic).*

4.  **Enable and Start the Service**:
    ```bash
    # Reload systemd to recognize the new file
    sudo systemctl daemon-reload

    # Enable the service to start on boot
    sudo systemctl enable mlflow-server.service

    # Start the service now
    sudo systemctl start mlflow-server.service

    # Check its status
    sudo systemctl status mlflow-server.service

    # View logs
    journalctl -u mlflow-server.service -f
    ```
    You can now safely log out of your SSH session. The MLflow server will continue running in the background, ready to accept data from your projects.
