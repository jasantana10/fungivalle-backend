import os
from fastapi_mail import ConnectionConfig

# Variables de entorno para email
MAIL_USERNAME = os.getenv("MAIL_USERNAME")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
MAIL_SERVER = os.getenv("MAIL_SERVER")
MAIL_FROM = os.getenv("MAIL_FROM")
MAIL_PORT = int(os.getenv("MAIL_PORT", 587))

# Configurar email SOLO si todas las variables existen
if all([MAIL_USERNAME, MAIL_PASSWORD, MAIL_SERVER, MAIL_FROM]):
    conf = ConnectionConfig(
        MAIL_USERNAME=MAIL_USERNAME,
        MAIL_PASSWORD=MAIL_PASSWORD,
        MAIL_FROM=MAIL_FROM,
        MAIL_PORT=MAIL_PORT,
        MAIL_SERVER=MAIL_SERVER,
        MAIL_STARTTLS=True,
        MAIL_SSL_TLS=False,
        USE_CREDENTIALS=True,
        VALIDATE_CERTS=True
    )
    EMAIL_ENABLED = True
    print("✅ Email service enabled")
else:
    conf = None
    EMAIL_ENABLED = False
    print("⚠️ Email service disabled (missing environment variables)")

# Esta función debe existir aunque el email esté deshabilitado
async def send_password_reset_email(email_to: str, token: str):
    if EMAIL_ENABLED:
        # Aquí va la lógica real de envío de email
        print(f"📧 Sending password reset email to {email_to}")
        # ... tu código real de envío ...
    else:
        print(f"⚠️ Email not sent to {email_to} - service disabled")