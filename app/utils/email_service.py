import os
from typing import List
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from pydantic import EmailStr, BaseModel
from dotenv import load_dotenv

load_dotenv()

class EmailSchema(BaseModel):
    email: List[EmailStr]
    body: dict

conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("EMAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("EMAIL_PASSWORD"),
    MAIL_FROM=os.getenv("EMAIL_FROM"),
    MAIL_FROM_NAME=os.getenv("EMAIL_FROM_NAME"),
    MAIL_PORT=int(os.getenv("EMAIL_PORT", 587)),
    MAIL_SERVER=os.getenv("EMAIL_HOST"),
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True
)

async def send_password_reset_email(email_to: str, token: str, username: str):
    """Envía email para recuperación de contraseña"""
    
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:63689")
    reset_link = f"{frontend_url}/reset-password?token={token}"
    
    message = MessageSchema(
        subject="Recupera tu contraseña - FungiValle",
        recipients=[email_to],
        body=f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="text-align: center; margin-bottom: 30px;">
                    <h1 style="color: #2E7D32;">🍄 FungiValle</h1>
                    <p style="color: #666;">Recuperación de contraseña</p>
                </div>
                
                <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px;">
                    <h2 style="color: #333;">Hola {username},</h2>
                    <p>Hemos recibido una solicitud para restablecer la contraseña de tu cuenta.</p>
                    
                    <div style="text-align: center; margin: 30px 0;">
                        <a href="{reset_link}" 
                           style="background-color: #2E7D32; color: white; 
                                  padding: 12px 30px; text-decoration: none; 
                                  border-radius: 5px; font-weight: bold;">
                            Restablecer Contraseña
                        </a>
                    </div>
                    
                    <p>O copia este enlace en tu navegador:</p>
                    <p style="background-color: #eee; padding: 10px; border-radius: 5px; 
                              word-break: break-all;">
                        {reset_link}
                    </p>
                    
                    <p>Este enlace expirará en 1 hora por seguridad.</p>
                    
                    <p style="color: #666; font-size: 14px; margin-top: 30px;">
                        Si no solicitaste este cambio, ignora este mensaje.
                    </p>
                </div>
                
                <div style="text-align: center; margin-top: 30px; color: #999; font-size: 12px;">
                    <p>© 2024 FungiValle. Todos los derechos reservados.</p>
                    <p>Valledupar, Colombia</p>
                </div>
            </div>
        </body>
        </html>
        """,
        subtype="html"
    )
    
    fm = FastMail(conf)
    await fm.send_message(message)

async def send_welcome_email(email_to: str, username: str):
    """Envía email de bienvenida al registrarse"""
    message = MessageSchema(
        subject="¡Bienvenido a FungiValle! 🍄",
        recipients=[email_to],
        body=f"""
        <html>
        <body>
            <h1>¡Bienvenido a FungiValle, {username}!</h1>
            <p>Gracias por unirte a nuestra comunidad de micólogos en Valledupar.</p>
            <p>Comienza a identificar y documentar especies de hongos locales.</p>
        </body>
        </html>
        """,
        subtype="html"
    )
    
    fm = FastMail(conf)
    await fm.send_message(message)