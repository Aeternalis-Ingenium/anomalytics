import json
import typing
from http import client
from urllib.parse import urlparse

import pandas as pd

from anomalytics.notifications.abstract import Notification


class SlackNotification(Notification):
    """
    Notification class that setups message for your anomalies and sends them to Slack via webhook.

    ## Attributes
    -------------
    webhook_url : str
        The URL of the Slack webhook used to send notifications.

    __headers : typing.Dict[str, str]
        The HTTP headers, by default - {"Content-Type": "application/json"}.

    __payload : str
        The payload for the notification message, by default an empty string.
    """

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.__headers: typing.Dict[str, str] = {"Content-Type": "application/json"}
        self.__payload: str = ""
        self.__subject: str = "🤖 Anomalytics - Anomaly Detected!"

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
            fmt_message = f"{self.__subject}\n\n{anomaly_report}"
        else:
            fmt_message = f"{self.__subject}\n\n{message}\n\n{anomaly_report}"
        self.__payload = json.dumps({"text": fmt_message})

    @property
    def send(self):
        """
        Synchronously sends the prepared message to a Slack channel.
        """
        if len(self.__payload) == 0:
            raise ValueError("Payload not set. Please call `setup()` method first.")

        parsed_url = urlparse(url=self.webhook_url)
        connection = client.HTTPSConnection(parsed_url.netloc)  # type: ignore

        connection.request(method="POST", url=parsed_url.path, body=self.__payload, headers=self.__headers)
        response = connection.getresponse()

        if response.status == 200:
            print("Notification sent successfully.")
        else:
            print(f"Failed to send notification. Status code: {response.status} - {response.reason}")

        connection.close()

    def __str__(self):
        return "Slack Notification"
