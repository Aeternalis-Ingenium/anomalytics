import datetime
import typing
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from smtplib import SMTP

from anomalytics.notifications.notification import Notification


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
        self.__subject = "🤖 Detecto: Anomaly detected!"
        self.__payload: str = ""

    def __format_data(self, data: typing.Dict[str, typing.Union[str, float, int, datetime.datetime]], index: int):
        """
        Formats a single anomaly dictionary into a string for Slack message formatting.

        # Parameters
        ------------
        data : typing.Dict[str, typing.Union[str, float, int, datetime.datetime]]
            A dictionary containing details of an anomaly.

        index : int
            Index of the anomaly in the list, used for numbering in the message.

        # Returns
        ---------
        fmt_data : str
            Formatted string representing the anomaly.
        """
        date = data["date"]
        column = data["column"]
        anomaly = data["anomaly"]
        return f"{index + 1}. Date: {date} | Column: {column} | Anomaly: {anomaly}"

    def setup(
        self, data: typing.List[typing.Dict[str, typing.Union[str, float, int, datetime.datetime]]], message: str
    ):
        """
        Prepares the email message with given data and a custom message.

        # Parameters
        ------------
        data : typing.List[typing.Dict[str, typing.Union[str, float, int, datetime.datetime]]]
            A list of dictionaries which represent all the detected anomaly data.

        message : str
            A custom message to be included in the notification.
        """
        if not isinstance(data, list):
            raise TypeError("Data argument must be of type list")
        else:
            for element in data:
                if not isinstance(element, dict):
                    raise TypeError("Data argument must be of type dict")
                else:
                    for key in element.keys():
                        if key not in ["date", "column", "anomaly"]:
                            raise KeyError("Key needs to be one of these: date, column, anomaly")

        fmt_data = "\n".join(
            self.__format_data(data=anomaly_data, index=index) for index, anomaly_data in enumerate(data)
        )
        fmt_message = f"{message}\n\n{fmt_data}"
        self.__payload = fmt_message

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
