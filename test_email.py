# test_email.py
import os
import sys
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Cargar variables .env
load_dotenv()

def test_gmail_smtp():
    print("📧 Probando conexión SMTP de Gmail...")
    
    # Obtener credenciales
    email_user = os.getenv("EMAIL_USERNAME")
    email_pass = os.getenv("EMAIL_PASSWORD")
    
    print(f"Usuario: {email_user}")
    print(f"Contraseña: {'*' * len(email_pass)} ({len(email_pass)} chars)")
    
    # Quitar espacios si los hay
    if ' ' in email_pass:
        print("⚠️  Contraseña tiene espacios. Limpiando...")
        email_pass = email_pass.replace(' ', '')
        print(f"Contraseña limpia: {'*' * len(email_pass)}")
    
    try:
        # Conectar a SMTP
        print("\n🔗 Conectando a smtp.gmail.com:587...")
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.starttls()  # Upgrade a conexión segura
        print("✅ TLS activado")
        
        # Login
        print("🔐 Iniciando sesión...")
        server.login(email_user, email_pass)
        print("✅ Login exitoso!")
        
        # Crear email de prueba
        msg = MIMEMultipart()
        msg['From'] = email_user
        msg['To'] = email_user  # Enviar a ti mismo para prueba
        msg['Subject'] = "✅ Test FungiValle - SMTP Funciona"
        
        body = """
        <h1>¡FungiValle SMTP Test!</h1>
        <p>Si recibes este email, la configuración SMTP está funcionando correctamente.</p>
        <p><strong>Fecha:</strong> Ahora mismo</p>
        <p><strong>App:</strong> FungiValle Backend</p>
        <hr>
        <p>🍄 Valledupar, Colombia</p>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Enviar
        print("\n📤 Enviando email de prueba...")
        server.send_message(msg)
        print("✅ Email enviado exitosamente!")
        
        # Cerrar conexión
        server.quit()
        print("\n🎉 ¡Configuración SMTP funciona perfectamente!")
        print("Revisa la bandeja de entrada de: " + email_user)
        
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}")
        print(f"Detalle: {e}")
        
        # Soluciones comunes
        print("\n🔧 Posibles soluciones:")
        print("1. Verifica que la contraseña NO tenga espacios")
        print("2. Activa 'Acceso de apps menos seguras' en Google")
        print("3. Asegúrate de tener 'Verificación en 2 pasos' activada")
        print("4. Genera NUEVA contraseña de aplicación")
        print("5. Verifica que el email exista y puedas acceder")

if __name__ == "__main__":
    test_gmail_smtp()