import datetime
import typing
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from smtplib import SMTP

import pandas as pd

from anomalytics.notifications.abstract import Notification


class EmailNotification(Notification):
    """
    Notification class that setups the message for your anomalies and sends them via your email address.

    ## Attributes
    --------------
    sender_address : str
        The email address from which the email will be sent.

    password : str
        The password or app-specific password for authenticating the email account.

    recipient_addresses : typing.List[str]
        The list of your recipent's email addresses.

    smtp_host : str
        The SMTP host address for the email provider: smtp.YOUR_EMAIL_PROVIDER.com.

    smtp_port : int
        The SMTP server port for the email provider, default 587.

    __payload : str
        The payload for the notification message, by default an empty string.

    __subject : str
        The sibject of your email.
    """

    def __init__(
        self,
        sender_address: str,
        password: str,
        recipient_addresses: typing.List[str],
        smtp_host: str,
        smtp_port: int = 587,
    ) -> None:
        self.sender_address = sender_address
        self.password = password
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.recipient_addresses = recipient_addresses
        self.__subject = "🤖 Anomalytics - Anomaly Detected!"
        self.__payload: str = ""

    def setup(
        self,
        detection_summary: pd.DataFrame,
        message: str,
    ):
        """
        Prepares the email message with given data and a custom message.

        ## Parameters
        -------------
        detection_summary : pandas.DataFrame
            A DataFrame with summarized detection result.

        message : str
            A custom message to be included in the notification.
        """
        if not isinstance(detection_summary, pd.DataFrame):
            raise TypeError("Invalid type! `detection_summary` must be Pandas DataFrame")

        most_recent_data = detection_summary.iloc[[-1]]
        anomaly_report = f"Row: {most_recent_data.row.values[0]} | Date: {most_recent_data.datetime.values[0]} | Anomalous Data: {most_recent_data.anomalous_data.values[0]} | Anomaly Score: {most_recent_data.anomaly_score.values[0]} | Anomaly Threshold: {most_recent_data.anomaly_threshold.values[0]}"

        if not message:
            self.__payload = anomaly_report
        else:
            self.__payload = f"{message}" f"\n\n{anomaly_report}"

    @property
    def send(self):
        """
        Synchronously sends the prepared email to the specified addresses.
        """
        if len(self.__payload) == 0:
            raise ValueError("Payload not set. Please call `setup()` method first.")

        msg = MIMEMultipart()
        msg["From"] = self.sender_address
        msg["To"] = ", ".join(self.recipient_addresses)
        msg["Subject"] = self.__subject
        msg.attach(MIMEText(self.__payload, "plain"))

        try:
            server = SMTP(host=self.smtp_host, port=self.smtp_port)
            server.starttls()
            server.login(user=self.sender_address, password=self.password)
            server.send_message(msg=msg, from_addr=self.sender_address, to_addrs=self.recipient_addresses)
            server.quit()
            print("Email sent successfully.")
        except Exception as e:
            print(f"Failed to send email. Error: {e}")

    def __str__(self):
        return "Email Notification"
