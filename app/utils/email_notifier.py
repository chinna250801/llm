import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

from app.utils.strategy import check_risk


def send_email_alert(subject, body, to_email):
    """
    Function to send an email alert.
    """
    from_email = "20fe1a05b3@gmail.com"  # Replace with your email
    password = "Daddy@9565"  # Use App Password if 2FA is enabled
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    # Create the email content
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Establish a connection with the Gmail SMTP server
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Start TLS encryption
            server.login(from_email, password)  # Log in to the SMTP server
            text = msg.as_string()  # Convert the message to string format
            server.sendmail(from_email, to_email, text)  # Send the email
            logging.info("Email sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")


# Example use case: Send alert if a certain loss threshold is hit
balance = 5000
price = 250
risk_threshold = 2  # 2% risk threshold
alert_email = "vsnreddy65@gmail.com"


def check_risk_and_alert(balance, price, risk_threshold):
    result, position_size = check_risk(balance, price, risk_threshold)

    if result == "Position Size" and 0.001 < 0.01:
        subject = "Risk Alert: Position Size Too Small"
        body = f"Risk threshold exceeded. Current position size: {position_size}. Consider taking action."
        send_email_alert(subject, body, alert_email)
    else:
        logging.info(f"Risk check passed: Position size: {position_size}")


# Example check
check_risk_and_alert(balance, price, risk_threshold)
