import unittest
from json import dumps
from unittest.mock import MagicMock, patch

import pytest

from anomalytics.notifications.abstract import Notification
from anomalytics.notifications.slack import SlackNotification


@pytest.mark.usefixtures("get_sample_1_detection_summary")
class TestSlackNotification(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.slack_notification = SlackNotification(webhook_url="https://hooks.slack.com/services/TEST/TOKEN/WEBHOOK")
        self.test_message = "Test notification message"

    def test_instance_is_abstract_class(self):
        self.assertIsInstance(obj=self.slack_notification, cls=Notification)
        self.assertEqual(first=type(self.slack_notification._SlackNotification__payload), second=str)  # type: ignore
        self.assertEqual(first=len(self.slack_notification._SlackNotification__payload), second=0)  # type: ignore
        self.assertEqual(first=self.slack_notification._SlackNotification__payload, second="")  # type: ignore
        self.assertEqual(first=type(self.slack_notification._SlackNotification__subject), second=str)  # type: ignore
        self.assertEqual(
            first=self.slack_notification._SlackNotification__subject, second="🤖 Anomalytics - Anomaly Detected!"  # type: ignore
        )

    def test_string_method(self):
        self.assertTrue(expr=str(self.slack_notification) == "Slack Notification")

    def test_setup_with_wrong_data_type(self):
        with self.assertRaises(expected_exception=TypeError):
            self.slack_notification.setup(data=self.sample_1_detection_summary.values, message=self.test_message)  # type: ignore

    def test_setup_with_message(self):
        expected_payload = dumps(
            {
                "text": "🤖 Anomalytics - Anomaly Detected!"
                "\n\nTest notification message"
                "\n\nDate: 2023-01-10\n\n"
                "Row: 9\n\n"
                "Anomaly: 75521\n\n"
                "Anomaly Score: 8.123\n\n"
                "Anomaly Threshold: 7.3"
            }
        )

        self.slack_notification.setup(detection_summary=self.sample_1_detection_summary, message=self.test_message)  # type: ignore

        self.assertIsNotNone(obj=self.slack_notification._SlackNotification__payload)  # type: ignore
        self.assertEqual(first=self.slack_notification._SlackNotification__payload, second=expected_payload)  # type: ignore

    def test_setup_without_message(self):
        expected_payload = dumps(
            {
                "text": "🤖 Anomalytics - Anomaly Detected!"
                "\n\nDate: 2023-01-10\n\n"
                "Row: 9\n\n"
                "Anomaly: 75521\n\n"
                "Anomaly Score: 8.123\n\n"
                "Anomaly Threshold: 7.3"
            }
        )

        self.slack_notification.setup(detection_summary=self.sample_1_detection_summary, message=None)  # type: ignore

        self.assertIsNotNone(obj=self.slack_notification._SlackNotification__payload)  # type: ignore
        self.assertEqual(first=self.slack_notification._SlackNotification__payload, second=expected_payload)  # type: ignore

    @patch("anomalytics.notifications.slack.client.HTTPSConnection")
    def test_send_method(self, mock_https_connection):
        mock_connection = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_connection.getresponse.return_value = mock_response
        mock_https_connection.return_value = mock_connection

        self.slack_notification.setup(detection_summary=self.sample_1_detection_summary, message=self.test_message)  # type: ignore
        self.slack_notification.send

        mock_connection.request.assert_called_once()
        mock_connection.close.assert_called_once()

    def test_send_without_setup(self):
        with self.assertRaises(expected_exception=ValueError):
            self.slack_notification.send

    def tearDown(self) -> None:
        return super().tearDown()
