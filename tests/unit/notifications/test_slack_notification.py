from json import dumps
from unittest import TestCase
from unittest.mock import MagicMock, patch

from anomalytics.notifications.abstract import Notification
from anomalytics.notifications.slack import SlackNotification


class TestSlackNotification(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.slack_notification = SlackNotification(webhook_url="https://hooks.slack.com/services/TEST/TOKEN/WEBHOOK")
        self.test_message = "Test notification message"
        self.test_data = [
            {"date": "2023-10-23T00:00:00.000Z", "column": "col_1", "anomaly": 2003.214},
            {"date": "2023-10-23T00:00:00.000Z", "column": "col_2", "anomaly": 1055.67},
        ]

    def test_instance_is_abstract_class(self):
        self.assertIsInstance(obj=self.slack_notification, cls=Notification)
        self.assertEqual(first=type(self.slack_notification._SlackNotification__payload), second=str)  # type: ignore
        self.assertEqual(first=len(self.slack_notification._SlackNotification__payload), second=0)  # type: ignore
        self.assertEqual(first=self.slack_notification._SlackNotification__payload, second="")  # type: ignore
        self.assertEqual(first=type(self.slack_notification._SlackNotification__subject), second=str)  # type: ignore
        self.assertEqual(
            first=self.slack_notification._SlackNotification__subject, second="🤖 Detecto: Anomaly detected!"  # type: ignore
        )

    def test_string_method(self):
        self.assertTrue(expr=str(self.slack_notification) == "Slack Notification")

    def test_setup_with_wrong_data_type(self):
        with self.assertRaises(expected_exception=TypeError):
            self.slack_notification.setup(data=self.test_data[0], message=self.test_message)  # type: ignore

    def test_setup_with_wrong_data_type_first_nested(self):
        with self.assertRaises(expected_exception=TypeError):
            self.slack_notification.setup(
                data=[("date", "2023-10-23T00:00:00.000Z"), ("column", "col_1"), ("anomaly", 2003.214)],  # type: ignore
                message=self.test_message,
            )

    def test_setup_with_wrong_key_name(self):
        with self.assertRaises(expected_exception=KeyError):
            self.slack_notification.setup(  # type: ignore
                data=[{"date": "2023-10-23T00:00:00.000Z", "col": "col_1", "anomaly_value": 1055.67}],  # type: ignore
                message=self.test_message,
            )

    def test_format_data_method(self):
        expected_fmt_data = "1. Date: 2023-10-23T00:00:00.000Z | Column: col_1 | Anomaly: 2003.214"
        fmt_data = self.slack_notification._SlackNotification__format_data(data=self.test_data[0], index=0)  # type: ignore

        self.assertEqual(first=fmt_data, second=expected_fmt_data)

    def test_setup(self):
        expected_payload = dumps(
            {
                "text": (
                    "🤖 Detecto: Anomaly detected!\n"
                    f"\n\n{self.test_message}\n"
                    "\n\n1. Date: 2023-10-23T00:00:00.000Z | Column: col_1 | Anomaly: 2003.214"
                    "\n2. Date: 2023-10-23T00:00:00.000Z | Column: col_2 | Anomaly: 1055.67"
                )
            }
        )

        self.slack_notification.setup(data=self.test_data, message=self.test_message)  # type: ignore

        self.assertIsNotNone(obj=self.slack_notification._SlackNotification__payload)  # type: ignore
        self.assertEqual(first=self.slack_notification._SlackNotification__payload, second=expected_payload)  # type: ignore

    def test_setup_without_message(self):
        expected_payload = dumps(
            {
                "text": (
                    "🤖 Detecto: Anomaly detected!\n"
                    "\n\n1. Date: 2023-10-23T00:00:00.000Z | Column: col_1 | Anomaly: 2003.214"
                    "\n2. Date: 2023-10-23T00:00:00.000Z | Column: col_2 | Anomaly: 1055.67"
                )
            }
        )

        self.slack_notification.setup(data=self.test_data, message=None)  # type: ignore

        self.assertIsNotNone(obj=self.slack_notification._SlackNotification__payload)  # type: ignore
        self.assertEqual(first=self.slack_notification._SlackNotification__payload, second=expected_payload)  # type: ignore

    @patch("anomalytics.notifications.slack.client.HTTPSConnection")
    def test_send_method(self, mock_https_connection):
        mock_connection = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_connection.getresponse.return_value = mock_response
        mock_https_connection.return_value = mock_connection

        self.slack_notification.setup(data=self.test_data, message=self.test_message)  # type: ignore
        self.slack_notification.send

        mock_connection.request.assert_called_once()
        mock_connection.close.assert_called_once()

    def test_send_without_setup(self):
        with self.assertRaises(expected_exception=ValueError):
            self.slack_notification.send

    def tearDown(self) -> None:
        return super().tearDown()
