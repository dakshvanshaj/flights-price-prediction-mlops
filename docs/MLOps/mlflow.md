# 🚀 MLflow Integration and Deployment

This document provides a comprehensive guide to integrating MLflow for experiment tracking and model management. It covers connecting a client application to a remote MLflow server and deploying a robust tracking server on AWS.

## 🔗 1. Connecting a Client to MLflow

This section explains how a client application (e.g., a prediction API in a Docker container) securely connects to a remote MLflow server to fetch models and artifacts.

### ⚙️ 1.1. The Connection Mechanism

The client connection is a two-step process managed by environment variables.

1.  **Initial Connection to the Tracking Server:**
    *   The client application needs the address of the MLflow tracking server on the EC2 instance.
    *   This is set using the `MLFLOW_TRACKING_URI` environment variable.
    *   The MLflow client library automatically uses this variable to initiate a network request to the server.

2.  **Fetching Artifacts from S3:**
    *   After connecting, the tracking server tells the client where the model files are located (typically an S3 bucket path).
    *   The client then needs permissions to access S3, which are provided via AWS credentials.
    *   These are set using the `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_DEFAULT_REGION` environment variables.

### 🕵️ 1.2. Automatic Credential Detection

> You do **not** need to install the AWS CLI or run `aws configure` inside the Docker container. The AWS SDK for Python (`boto3`), used by MLflow, automatically finds and uses credentials from environment variables.

When `boto3` detects `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in the container's environment, it authenticates with AWS services like S3 automatically.

### 🔒 1.3. Production Best Practice: Environment Files

The most secure and standard method for providing credentials to a Docker container is using an environment file (e.g., `prediction_app.env`).

> **Warning:** This file contains sensitive credentials and must **never** be committed to version control. Add it to your `.gitignore` file.

**1. Create the `.env` file:**

```env
# ----------------------------------
# MLflow Production Configuration
# ----------------------------------
# The public IP or domain of your EC2 instance running the MLflow server.
MLFLOW_TRACKING_URI="http://<YOUR_EC2_IP_ADDRESS>:5000"

# ----------------------------------
# AWS Credentials for MLflow Artifacts
# ----------------------------------
# Credentials for an IAM user with read-only access to your S3 artifact bucket.
AWS_ACCESS_KEY_ID="<YOUR_AWS_ACCESS_KEY_ID>"
AWS_SECRET_ACCESS_KEY="<YOUR_AWS_SECRET_ACCESS_KEY>"
AWS_DEFAULT_REGION="<YOUR_S3_BUCKET_REGION>"
```

**2. Run the container with the `--env-file` flag:**

This command injects the variables into the container at runtime.

```bash
docker run --env-file ./src/prediction_server/prediction_app.env -p 8000:8000 your-image-name
```

## ☁️ 2. Deploying an MLflow Tracking Server on AWS

This section provides a step-by-step guide to setting up a robust MLflow Tracking Server using **EC2** (server), **S3** (artifact storage), and **RDS** (backend database).

### 🪣 2.1. Step 1: Create an S3 Bucket for Artifacts

1.  **Navigate to S3:** Log into your AWS account and go to the **S3** service.
2.  **Create Bucket:** Click **"Create bucket"** and configure the following:
    *   **Bucket name**: Must be **globally unique** (e.g., `yourname-mlflow-artifacts-2025`).
    *   **AWS Region**: Choose a region (e.g., `ap-south-1`). **Important:** Launch all other services (EC2, RDS) in the *same region*.
    *   **Block Public Access**: Keep **"Block all public access"** checked for security.
    *   **Bucket Versioning**: **Enable** to protect against accidental data loss.
    *   **Tags (Recommended)**: Add a tag for cost tracking (e.g., `Key: Project`, `Value: mlflow-server`).
    *   **Default encryption**: Keep the default (`SSE-S3`).
3.  **Finalize:** Review your settings and click **"Create bucket"**.

### 🐘 2.2. Step 2: Create a PostgreSQL Database with RDS

1.  **Navigate to RDS:** In the AWS Console, go to the **RDS** service.
2.  **Create Database:** Click **"Create database"** and follow the wizard:
    *   **Creation method**: Select **"Standard Create"**.
    *   **Engine**: Choose **"PostgreSQL"**.
    *   **Templates**: Select the **"Free tier"** template.
    *   **Settings**:
        *   **DB instance identifier**: `mlflow-db`.
        *   **Master username**: `mlflow_user`.
        *   **Master password**: Create a strong password and **store it securely**.
    *   **Connectivity**:
        *   **Public access**: Select **"No"**.
        *   **VPC security group**: Choose **"Create new"** and name it `mlflow-db-security-group`.
    *   **Additional configuration**:
        *   **Initial database name**: Enter `mlflow_db`. This is crucial.
3.  **Finalize:** Review the settings and click **"Create database"**.

> **Note:** Database creation can take 10-15 minutes.

### 🖥️ 2.3. Step 3: Launch an EC2 Virtual Server

1.  **Navigate to EC2:** Go to the **EC2** service in the AWS Console.
2.  **Launch Instance:** Click **"Launch instance"** and configure:
    *   **Name**: `mlflow-server`.
    *   **AMI**: Select **Ubuntu** (Free tier eligible).
    *   **Instance type**: Choose `t2.micro` (Free Tier eligible).
    *   **Key pair (login)**:
        *   Click **"Create new key pair"**, name it `mlflow-key`, and keep the defaults.
        *   The `.pem` file will download. **Store this file securely.**
    *   **Network settings**:
        *   Click **"Edit"**.
        *   Create a new security group (`mlflow-server-sg`) with these **inbound rules**:
            1.  **SSH**: `Type: SSH`, `Source: My IP` (for better security).
            2.  **HTTP**: `Type: HTTP`, `Source: Anywhere`.
            3.  **Custom TCP**: `Type: Custom TCP`, `Port: 5000`, `Source: Anywhere`.
3.  **Launch:** Review the summary and click **"Launch instance"**.

### 🤝 2.4. Step 4: Connecting the Components

#### 2.4.1. Connect EC2 and RDS Security Groups

Create a firewall rule to allow the EC2 instance to communicate with the RDS database.

1.  Navigate to the **RDS** dashboard, select your `mlflow-db`, and go to the **"Connectivity & security"** tab.
2.  Click on the active **VPC security group** (`mlflow-db-security-group`).
3.  Go to the **"Inbound rules"** tab and click **"Edit inbound rules"**.
4.  Add a new rule:
    *   **Type**: `PostgreSQL`.
    *   **Source**: Select your EC2 security group (`mlflow-server-sg`).
5.  Click **"Save rules"**.

#### 2.4.2. Connect to Your EC2 Instance

1.  Go to the **EC2** dashboard, select your `mlflow-server`, and copy the **"Public IPv4 address"**.
2.  Open a terminal and make your key file private:

```bash
chmod 400 /path/to/your/mlflow-key.pem
```

3.  Connect via SSH:

```bash
ssh -i /path/to/your/mlflow-key.pem ubuntu@<YOUR_PUBLIC_IP_ADDRESS>
```

#### 2.4.3. Create and Attach an IAM Role for S3 Access

Grant your EC2 instance permissions to access the S3 bucket.

1.  **Create an IAM Policy**:
    *   Go to **IAM** > **Policies** > **"Create policy"**.
    *   Use the visual editor:
        *   **Service**: `S3`.
        *   **Actions**: `ListBucket`, `GetObject`, `PutObject`, `DeleteObject`.
        *   **Resources**: Specify the ARN for your bucket (`arn:aws:s3:::your-bucket-name`) and the objects within it (`arn:aws:s3:::your-bucket-name/*`).
    *   Name the policy `MLflowS3AccessPolicy`.

2.  **Create an IAM Role**:
    *   Go to **IAM** > **Roles** > **"Create role"**.
    *   **Trusted entity**: `AWS service`.
    *   **Use case**: `EC2`.
    *   Attach the `MLflowS3AccessPolicy` you just created.
    *   Name the role `MLflowEC2Role`.

3.  **Attach the Role to EC2**:
    *   In the **EC2** dashboard, select your `mlflow-server`.
    *   Go to **"Actions"** > **"Security"** > **"Modify IAM role"**.
    *   Select `MLflowEC2Role` and save.

#### 2.4.4. Ubuntu Server Setup Best Practices

1.  **Update Your System**:

```bash
sudo apt update && sudo apt upgrade -y
```

2.  **Create a New User**:

```bash
# Replace 'your_username' with a chosen name
sudo adduser your_username
sudo usermod -aG sudo your_username
```
    > Log out and log back in as the new, non-root user for daily work.

3.  **Set Up a Basic Firewall**:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 5000/tcp
sudo ufw enable
```

### 🛠️ 2.5. Step 5: Install MLflow Software

1.  **Install Tools**:

```bash
sudo apt update
sudo apt install python3-pip python3-venv -y
```

2.  **Create a Virtual Environment and Install Packages**:

```bash
python3 -m venv mlflow-env
source mlflow-env/bin/activate
pip install mlflow boto3 psycopg2-binary
```

### ▶️ 2.6. Step 6: Launch the MLflow Server

This command connects all the components. You will need your **RDS Endpoint**, **RDS Password**, and **S3 Bucket Name**.

> **SQLAlchemy Connection String**

> The required format is `postgresql://<user>:<password>@<host>:<port>/<database>`.

Execute the following, replacing all placeholders:

```bash
mlflow server \
    --backend-store-uri postgresql://mlflow_user:<YOUR_RDS_PASSWORD>@<YOUR_RDS_ENDPOINT>/mlflow_db \
    --default-artifact-root s3://<your-s3-bucket-name>/
    --host 0.0.0.0 \
    --port 5000
```

> **`--host 0.0.0.0` Explained:** This tells the server to listen on all available network interfaces, making the UI accessible via the instance's public IP address.

To keep the server running after you disconnect, use a `screen` session:

```bash
# Start a new named session
screen -S mlflow

# Activate environment and run the mlflow server command
source mlflow-env/bin/activate
mlflow server ... # (paste the full command from above)

# Detach from the session by pressing Ctrl+A, then D.
```

### 💻 2.7. Step 7: Local Machine Setup

#### 2.7.1. Connect Your Local Project

Configure your local machine to log experiments to the remote server.

*   **Set the Tracking URI**:

```bash
# In your local terminal (macOS/Linux)
export MLFLOW_TRACKING_URI="http://<YOUR_EC2_PUBLIC_IP>:5000"
```

*   **Configure AWS Credentials**: Grant your local machine S3 upload permissions.

#### 2.7.2. AWS CLI Configuration Guide

1.  **Install the AWS CLI**:

```bash
pip install awscli
```

2.  **Run Configure**:

```bash
aws configure
```

3.  **Enter Your Credentials**:
    *   **AWS Access Key ID**: Paste your key.
    *   **AWS Secret Access Key**: Paste your secret key.
    *   **Default region name**: Enter your S3 bucket's region (e.g., `ap-south-1`).
    *   **Default output format**: Press Enter for `json`.

The CLI securely stores these credentials, and MLflow will automatically use them.

### 🔄 2.8. Step 8: Persistent Server Operation with `systemd`

To run the MLflow server as a background service that starts on boot, use `systemd`.

1.  **SSH into your EC2 server**.
2.  **Create a `systemd` service file**:

```bash
sudo nano /etc/systemd/system/mlflow-server.service
```

3.  **Paste the following configuration**, replacing all placeholders.

```ini
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
User=<your_user>
Restart=on-failure
# Note: Use the absolute path to the mlflow executable in your venv
ExecStart=/home/<your_user>/mlflow-env/bin/mlflow server \
    --backend-store-uri postgresql://mlflow_user:<YOUR_RDS_PASSWORD>@<YOUR_RDS_ENDPOINT>/mlflow_db \
    --default-artifact-root s3://<your-s3-bucket-name>/
    --host 127.0.0.1 \
    --port 5000

[Install]
WantedBy=multi-user.target
```

    > **Security Note:** `--host` is set to `127.0.0.1`, meaning the server only accepts connections from the machine itself. A reverse proxy like Nginx should be used to handle public traffic securely.

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
```

## 📍 3. Using a Static IP with AWS Elastic IP

An EC2 instance's public IP changes on every restart, which breaks the `MLFLOW_TRACKING_URI`. An **Elastic IP (EIP)** provides a permanent, static IP address to solve this problem.

### 🏠 3.1. EIP Intuition: A Permanent Address

An EIP is a static public IPv4 address you allocate to your AWS account. You can attach it to your EC2 instance, and it will persist across all stop/start cycles, ensuring permanent connectivity.

### 🗺️ 3.2. Step-by-Step EIP Implementation

1.  **Allocate an Elastic IP**:
    *   Go to the **EC2 Dashboard** > **Elastic IPs**.
    *   Click **"Allocate Elastic IP address"** and confirm by clicking **"Allocate"**.

2.  **Associate the EIP with Your EC2 Instance**:
    *   On the **Elastic IPs** screen, select the new IP.
    *   Click **"Actions"** > **"Associate Elastic IP address"**.
    *   Choose **Instance** as the resource type and select your `mlflow-server` instance.
    *   Click **"Associate"**.

3.  **Update the `MLFLOW_TRACKING_URI`**:
    *   Replace the old dynamic IP in your `prediction_app.env` file and any other configurations with the new Elastic IP.
    *   Example: `MLFLOW_TRACKING_URI="http://<Your_Elastic_IP>:5000"`


### ⚠️ 3.3. EIP Cost and Security Considerations

| Consideration      | Detail                                                                                                                                                           | Best Practice/Tip                                                                                                                                                           |
| :----------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Cost** 💰        | **EIPs are free only when associated with a *running* EC2 instance.** AWS charges a small hourly fee for EIPs that are allocated but unassociated or on a *stopped* instance. | Since your server may stop/start, you will incur a minimal charge during the **stopped** period. This is usually worth the operational stability.                               |
| **Security** 🛡️   | The EIP is just an address. Your EC2 **Security Group** (`mlflow-server-sg`) must still allow inbound traffic on port `5000`.                                        | **No change is needed** if you configured the security group correctly in Step 2.3.                                                                                         |
| **Advanced DNS** 🏷️ | For maximum flexibility, use **Route 53** to create a friendly domain name (e.g., `mlflow.yourproject.com`) that points to the EIP.                               | If you ever change the EIP, you only need to update the DNS record, not every client configuration. This decouples your clients from the specific IP address. |
