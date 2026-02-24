import os
import mimetypes
from datetime import datetime
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType, NameEmail
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

# FastAPI-Mail configuration
conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME", ""),
    MAIL_PASSWORD=SecretStr(os.getenv("MAIL_PASSWORD", "")),
    MAIL_FROM=os.getenv("MAIL_FROM", ""),
    MAIL_PORT=int(os.getenv("MAIL_PORT", 587)),
    MAIL_SERVER=os.getenv("MAIL_SERVER", ""),
    MAIL_STARTTLS=os.getenv("MAIL_STARTTLS", "False").lower() == "true",
    MAIL_SSL_TLS=os.getenv("MAIL_SSL_TLS", "True").lower() == "true",
    MAIL_FROM_NAME=os.getenv("MAIL_FROM_NAME", "Alert System"),
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True
)

MAIL_RECEIVERS = os.getenv("MAIL_RECEIVER", "").split(",")
MAIL_RECEIVERS = [email.strip() for email in MAIL_RECEIVERS if email.strip()]


# HTML email template wrapper
def create_email_template(title, body_content, alert_color, icon):
    """Create a professional email template with consistent styling"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
            }}
            .email-container {{
                max-width: 600px;
                margin: 20px auto;
                background: #ffffff;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .header {{
                background: {alert_color};
                color: white;
                padding: 30px 20px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 28px;
                font-weight: 600;
            }}
            .icon {{
                font-size: 48px;
                margin-bottom: 10px;
            }}
            .content {{
                padding: 30px 20px;
            }}
            .info-box {{
                background: #f8f9fa;
                border-left: 4px solid {alert_color};
                padding: 15px;
                margin: 20px 0;
                border-radius: 4px;
            }}
            .info-box p {{
                margin: 8px 0;
            }}
            .info-box strong {{
                color: {alert_color};
            }}
            .action-box {{
                background: #fff3cd;
                border: 1px solid #ffc107;
                padding: 15px;
                margin: 20px 0;
                border-radius: 4px;
            }}
            .footer {{
                background: #f8f9fa;
                padding: 20px;
                text-align: center;
                font-size: 12px;
                color: #6c757d;
                border-top: 1px solid #dee2e6;
            }}
            .btn {{
                display: inline-block;
                padding: 12px 24px;
                background: {alert_color};
                color: white;
                text-decoration: none;
                border-radius: 4px;
                margin-top: 15px;
                font-weight: 600;
            }}
        </style>
    </head>
    <body>
        <div class="email-container">
            <div class="header">
                <div class="icon">{icon}</div>
                <h1>{title}</h1>
            </div>
            <div class="content">
                {body_content}
            </div>
            <div class="footer">
                <p><strong>AI Monitoring System</strong></p>
                <p>This is an automated alert. Please do not reply to this email.</p>
                <p>For support, contact: ai@alluvium.in</p>
                <p style="margin-top: 15px; color: #adb5bd;">
                    © 2025 Alluvium Iot Solution Pvt. Ltd.. All rights reserved.
                </p>
            </div>
        </div>
    </body>
    </html>
    """


async def send_alert_email(image_path: str, event_type: str, stream_name: str = ""):
    """
    Sends an alert email with different subject/body depending on event type.
    
    Args:
        image_path: Path to the detection image
        event_type: Type of alert (fire, smoke, ppe, etc.)
        stream_name: Name of the camera/stream
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    
    if not MAIL_RECEIVERS:
        print("[EMAIL ERROR] No receiver configured in .env")
        return False
    
    if not os.path.exists(image_path):
        print(f"[EMAIL ERROR] Image not found: {image_path}")
        return False
    
    now = datetime.now()
    event_type = event_type.lower()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
    date_formatted = now.strftime('%B %d, %Y at %I:%M %p')
    
    # -------------------------------
    # 🔥 Fire + Smoke (grouped)
    # -------------------------------
    if event_type in ("fire", "smoke", "fire_smoke"):
        subject = f"URGENT: Fire/Smoke Alert - {stream_name}"
        body_content = f"""
            <h2 style="color: #d32f2f; margin-top: 0;">Fire/Smoke Hazard Detected</h2>
            <p style="font-size: 16px;">
                A potential <strong>fire or smoke hazard</strong> has been detected by our AI monitoring system.
                Immediate attention is required.
            </p>
            
            <div class="info-box">
                <p><strong>Location:</strong> {stream_name}</p>
                <p><strong>Detection Time:</strong> {date_formatted}</p>
                <p><strong>Alert Type:</strong> Fire/Smoke Detection</p>
                <p><strong>Severity:</strong> <span style="color: #d32f2f; font-weight: bold;">CRITICAL</span></p>
            </div>
            
            <div class="action-box">
                <p style="margin: 0; color: #856404;">
                    <strong>⚠️ Required Action:</strong><br>
                    1. Verify the scene immediately<br>
                    2. Alert on-site safety personnel<br>
                    3. Activate emergency protocols if confirmed<br>
                    4. Review the attached image for assessment
                </p>
            </div>
            
            <p style="margin-top: 20px;">
                The detection image is attached to this email for your review.
                Please respond to this alert according to your safety procedures.
            </p>
        """
        alert_color = "#d32f2f"
        icon = "🔥"
    
    # -------------------------------
    # 🦺 PPE Violation
    # -------------------------------
    elif event_type == "ppe":
        subject = f"PPE Violation Alert - {stream_name}"
        body_content = f"""
            <h2 style="color: #f57f17; margin-top: 0;">Personal Protective Equipment Violation</h2>
            <p style="font-size: 16px;">
                An individual has been detected <strong>without required safety equipment (PPE)</strong> 
                in the monitored area.
            </p>
            
            <div class="info-box">
                <p><strong>Location:</strong> {stream_name}</p>
                <p><strong>Detection Time:</strong> {date_formatted}</p>
            </div>
            
            <div class="action-box">
                <p style="margin: 0; color: #856404;">
                    <strong>⚠️ Required Action:</strong><br>
                    1. Review the attached image to confirm violation<br>
                    2. Identify the individual if possible<br>
                    3. Issue safety reminder or disciplinary action<br>
                    4. Document the incident per company policy
                </p>
            </div>
            
            <p style="margin-top: 20px;">
                Maintaining PPE compliance is essential for workplace safety.
                Please address this violation promptly.
            </p>
        """
        alert_color = "#f57f17"
        icon = "🦺"
    
    # -------------------------------
    # 🚛 Load/Unload (Vehicle Tracking)
    # -------------------------------
    elif event_type in ("load_unload", "vehicle", "truck"):
        subject = f"Vehicle Activity Alert - {stream_name}"
        body_content = f"""
            <h2 style="color: #1976d2; margin-top: 0;">Loading/Unloading Activity Detected</h2>
            <p style="font-size: 16px;">
                Vehicle activity has been detected in the monitored loading zone.
            </p>
            
            <div class="info-box">
                <p><strong>Location:</strong> {stream_name}</p>
                <p><strong>Detection Time:</strong> {date_formatted}</p>
                <p><strong>Alert Type:</strong> Vehicle Tracking</p>
                <p><strong>Severity:</strong> <span style="color: #1976d2; font-weight: bold;">INFO</span></p>
            </div>
            
            <p style="margin-top: 20px;">
                This is an informational alert for activity logging and security monitoring purposes.
            </p>
        """
        alert_color = "#1976d2"
        icon = "🚛"
    
    # -------------------------------
    # Default/Unknown
    # -------------------------------
    else:
        subject = f"⚠️ Security Alert - {stream_name}"
        body_content = f"""
            <h2 style="color: #455a64; margin-top: 0;">Security Event Detected</h2>
            <p style="font-size: 16px;">
                An event has been detected that requires attention.
            </p>
            
            <div class="info-box">
                <p><strong>Location:</strong> {stream_name}</p>
                <p><strong>Detection Time:</strong> {date_formatted}</p>
                <p><strong>Alert Type:</strong> {event_type.upper()}</p>
            </div>
            
            <p style="margin-top: 20px;">
                Please review the attached image for details.
            </p>
        """
        alert_color = "#455a64"
        icon = "⚠️"
    
    # Create full HTML email
    html_body = create_email_template(subject, body_content, alert_color, icon)
    
    # MIME type detection
    mime_type, _ = mimetypes.guess_type(image_path)
    
    # Create message
    message = MessageSchema(
        subject=subject,
        recipients=[NameEmail(name='', email=email) for email in MAIL_RECEIVERS],
        body=html_body,
        subtype=MessageType.html,
        attachments=[image_path],
    )
    
    try:
        fm = FastMail(conf)
        await fm.send_message(message)
        print(f"[EMAIL SENT] ({event_type}) {stream_name}: {image_path}")
        print(f"[EMAIL INFO] Sent to: {', '.join(MAIL_RECEIVERS)}")
        return True
    
    except Exception as e:
        print(f"[EMAIL ERROR] {stream_name}: {e}")
        import traceback
        traceback.print_exc()
        return False
