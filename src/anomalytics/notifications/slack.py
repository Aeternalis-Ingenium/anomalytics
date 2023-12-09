import datetime
import json
import typing
from http import client
from urllib.parse import urlparse

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
        self.__subject: str = "🤖 Detecto: Anomaly detected!"

    def __format_data(self, data: typing.Dict[str, typing.Union[str, float, int, datetime.datetime]], index: int):
        """
        Formats a single anomaly dictionary into a string for Slack message formatting.

        ## Parameters
        -------------
        data : typing.Dict[str, typing.Union[str, float, int, datetime.datetime]]
            A dictionary containing details of an anomaly.

        index : int
            Index of the anomaly in the list, used for numbering in the message.

        ## Returns
        ----------
        fmt_data : str
            Formatted string representing the anomaly.
        """
        date = data["date"]
        column = data["column"]
        anomaly = data["anomaly"]
        return f"{index + 1}. Date: {date} | Column: {column} | Anomaly: {anomaly}"

    def setup(
        self,
        data: typing.List[typing.Dict[str, typing.Union[str, typing.Union[float, int, datetime.datetime]]]],
        message: typing.Optional[str],
    ):
        """
        Prepares the Slack message with given data and a custom message.

        ## Parameters
        -------------
        data : typing.List[typing.Dict[str, typing.Union[str, typing.Union[float, int, datetime.datetime]]]]
            A list of dictionaries which represent all the detected anomaly data.

        message : typing.Optional[str]
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

        if not message:
            fmt_message = f"{self.__subject}\n" f"\n\n{fmt_data}"
        else:
            fmt_message = f"{self.__subject}\n" f"\n\n{message}\n" f"\n\n{fmt_data}"
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
