# MachineGuard+ Email Notification Setup

Use this when you want Streamlit to email success/failure notifications after a live prediction or batch trend run.

## 1) Get Your Email App Password

For Gmail, do not use your normal Gmail password.

1. Go to your Google Account.
2. Open **Security**.
3. Enable **2-Step Verification** if it is not already enabled.
4. Search for **App passwords**.
5. Create an app password for **Mail**.
6. Copy the generated password.

That generated password is the value for `SMTP_PASSWORD`.

## 2) Create Your `.env` File

From the project root, copy the example file:

```bash
cp .env.example .env
```

Open `.env` and fill in your real values:

```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_gmail_app_password
SMTP_SENDER=your_email@gmail.com
SMTP_USE_TLS=true
SMTP_USE_SSL=false
NOTIFICATION_RECIPIENT=receiver_email@example.com
```

Important: keep `.env` private. It is already ignored by git, so your password should not be committed.

## 3) Run Streamlit

```bash
streamlit run streamlit_app.py
```

In the Streamlit sidebar:

1. Open **Email notifications**.
2. Tick **Send email after Streamlit runs**.
3. Check or edit the recipient email.
4. Run a live prediction or upload a batch CSV.

You should receive an email saying whether the run succeeded or failed.

## 4) Alternative: Streamlit Secrets

You can also use `.streamlit/secrets.toml` instead of `.env`:

```toml
[email]
smtp_host = "smtp.gmail.com"
smtp_port = "587"
smtp_username = "your_email@gmail.com"
smtp_password = "your_gmail_app_password"
smtp_sender = "your_email@gmail.com"
smtp_use_tls = "true"
smtp_use_ssl = "false"
notification_recipient = "receiver_email@example.com"
```

`.streamlit/secrets.toml` is also ignored by git.

## 5) Common Fixes

- If Gmail blocks login, confirm 2-Step Verification is on and use an app password.
- If your SMTP provider uses SSL, set `SMTP_PORT=465`, `SMTP_USE_SSL=true`, and `SMTP_USE_TLS=false`.
- If notification sending fails, the ML prediction still runs. Streamlit will show a warning with the email error.

## 6) Docker Setup

Docker Compose automatically reads `.env` from the project root. The `streamlit` service in `docker-compose.yml` passes these values into the container:

```yaml
SMTP_HOST
SMTP_PORT
SMTP_USERNAME
SMTP_PASSWORD
SMTP_SENDER
SMTP_USE_TLS
SMTP_USE_SSL
NOTIFICATION_RECIPIENT
```

Run with Docker:

```bash
docker compose up --build streamlit
```

Open Streamlit:

```text
http://localhost:8501
```

Then enable **Email notifications** in the sidebar and run a prediction.

Do not put real passwords directly inside `docker-compose.yml`. Keep them in `.env` locally, or in your deployment platform's secret manager.

## 7) GitHub Actions / CI-CD Secrets

For GitHub Actions, add secrets in your repository:

1. Open your GitHub repo.
2. Go to **Settings**.
3. Open **Secrets and variables**.
4. Open **Actions**.
5. Click **New repository secret**.
6. Add these names:

```text
SMTP_HOST
SMTP_PORT
SMTP_USERNAME
SMTP_PASSWORD
SMTP_SENDER
SMTP_USE_TLS
SMTP_USE_SSL
NOTIFICATION_RECIPIENT
```

Example workflow usage:

```yaml
env:
  SMTP_HOST: ${{ secrets.SMTP_HOST }}
  SMTP_PORT: ${{ secrets.SMTP_PORT }}
  SMTP_USERNAME: ${{ secrets.SMTP_USERNAME }}
  SMTP_PASSWORD: ${{ secrets.SMTP_PASSWORD }}
  SMTP_SENDER: ${{ secrets.SMTP_SENDER }}
  SMTP_USE_TLS: ${{ secrets.SMTP_USE_TLS }}
  SMTP_USE_SSL: ${{ secrets.SMTP_USE_SSL }}
  NOTIFICATION_RECIPIENT: ${{ secrets.NOTIFICATION_RECIPIENT }}
```

Your current CI builds and tests the project, but it does not run the Streamlit UI, so it does not need email secrets unless you add a CI step that sends a notification.
