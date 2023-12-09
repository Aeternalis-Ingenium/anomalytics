from unittest import TestCase
from unittest.mock import MagicMock, patch

from anomalytics.notifications.abstract import Notification
from anomalytics.notifications.email import EmailNotification


class TestEmailNotification(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.gmail_notification = EmailNotification(
            sender_address="example@gmail.com",
            password="password",
            recipient_addresses=["recipient1@gmail.com", "recipient2@web.de"],
            smtp_host="smtp.gmail.com",
            smtp_port=587,
        )
        self.webde_notification = EmailNotification(
            sender_address="example@web.de",
            password="password",
            recipient_addresses=["recipient1@gmail.com", "recipient2@web.de"],
            smtp_host="smtp.web.de",
            smtp_port=587,
        )
        self.test_message = "Test notification message"
        self.test_data = [
            {"date": "2023-10-23T00:00:00.000Z", "column": "col_1", "anomaly": 2003.214},
            {"date": "2023-10-23T00:00:00.000Z", "column": "col_2", "anomaly": 1055.67},
        ]

    def test_instance_is_abstract_class(self):
        self.assertIsInstance(obj=self.gmail_notification, cls=Notification)
        self.assertEqual(first=type(self.gmail_notification._EmailNotification__payload), second=str)  # type: ignore
        self.assertEqual(first=len(self.gmail_notification._EmailNotification__payload), second=0)  # type: ignore
        self.assertEqual(first=self.gmail_notification._EmailNotification__payload, second="")  # type: ignore
        self.assertEqual(first=type(self.gmail_notification._EmailNotification__subject), second=str)  # type: ignore
        self.assertEqual(
            first=self.gmail_notification._EmailNotification__subject, second="🤖 Detecto: Anomaly detected!"  # type: ignore
        )

        self.assertIsInstance(obj=self.webde_notification, cls=Notification)
        self.assertEqual(first=type(self.webde_notification._EmailNotification__payload), second=str)  # type: ignore
        self.assertEqual(first=len(self.webde_notification._EmailNotification__payload), second=0)  # type: ignore
        self.assertEqual(first=self.webde_notification._EmailNotification__payload, second="")  # type: ignore
        self.assertEqual(first=type(self.webde_notification._EmailNotification__subject), second=str)  # type: ignore
        self.assertEqual(
            first=self.webde_notification._EmailNotification__subject, second="🤖 Detecto: Anomaly detected!"  # type: ignore
        )

    def test_string_method(self):
        self.assertTrue(expr=str(self.gmail_notification) == "Email Notification")
        self.assertTrue(expr=str(self.webde_notification) == "Email Notification")

    def test_setup_with_wrong_data_type(self):
        with self.assertRaises(expected_exception=TypeError):
            self.gmail_notification.setup(data=self.test_data[0], message=self.test_message)  # type: ignore

        with self.assertRaises(expected_exception=TypeError):
            self.webde_notification.setup(data=self.test_data[0], message=self.test_message)  # type: ignore

    def test_setup_with_wrong_data_type_first_nested(self):
        with self.assertRaises(expected_exception=TypeError):
            self.gmail_notification.setup(
                data=[("date", "2023-10-23T00:00:00.000Z"), ("column", "col_1"), ("anomaly", 2003.214)],  # type: ignore
                message=self.test_message,
            )

        with self.assertRaises(expected_exception=TypeError):
            self.webde_notification.setup(
                data=[("date", "2023-10-23T00:00:00.000Z"), ("column", "col_1"), ("anomaly", 2003.214)],  # type: ignore
                message=self.test_message,
            )

    def test_setup_with_wrong_key_name(self):
        with self.assertRaises(expected_exception=KeyError):
            self.gmail_notification.setup(  # type: ignore
                data=[{"date": "2023-10-23T00:00:00.000Z", "col": "col_1", "anomaly_value": 1055.67}],  # type: ignore
                message=self.test_message,
            )

        with self.assertRaises(expected_exception=KeyError):
            self.webde_notification.setup(  # type: ignore
                data=[{"date": "2023-10-23T00:00:00.000Z", "col": "col_1", "anomaly_value": 1055.67}],  # type: ignore
                message=self.test_message,
            )

    def test_format_data_method(self):
        expected_fmt_data = "1. Date: 2023-10-23T00:00:00.000Z | Column: col_1 | Anomaly: 2003.214"
        fmt_gmail_data = self.gmail_notification._EmailNotification__format_data(data=self.test_data[0], index=0)  # type: ignore
        fmt_webde_data = self.gmail_notification._EmailNotification__format_data(data=self.test_data[0], index=0)  # type: ignore

        self.assertEqual(first=fmt_gmail_data, second=expected_fmt_data)
        self.assertEqual(first=fmt_webde_data, second=expected_fmt_data)

    def test_setup(self):
        expected_payload = (
            f"{self.test_message}\n\n"  # Note the double \n here for the gap between message and first anomaly
            "1. Date: 2023-10-23T00:00:00.000Z | Column: col_1 | Anomaly: 2003.214\n"  # \n at the end of this line
            "2. Date: 2023-10-23T00:00:00.000Z | Column: col_2 | Anomaly: 1055.67"
        )

        self.gmail_notification.setup(data=self.test_data, message=self.test_message)  # type: ignore
        self.webde_notification.setup(data=self.test_data, message=self.test_message)  # type: ignore

        self.assertNotEqual(first=self.gmail_notification._EmailNotification__payload, second="")  # type: ignore
        self.assertNotEqual(first=self.webde_notification._EmailNotification__payload, second="")  # type: ignore
        self.assertEqual(first=self.gmail_notification._EmailNotification__payload, second=expected_payload)  # type: ignore
        self.assertEqual(first=self.webde_notification._EmailNotification__payload, second=expected_payload)  # type: ignore

    @patch("anomalytics.notifications.email.SMTP")
    def test_send_gmail(self, mock_smtp):
        mock_smtp.return_value = MagicMock()

        self.gmail_notification.setup(data=self.test_data, message=self.test_message)  # type: ignore
        self.gmail_notification.send

        mock_smtp.assert_called_with(host="smtp.gmail.com", port=587)
        mock_smtp.return_value.starttls.assert_called_once()
        mock_smtp.return_value.login.assert_called_once_with(user="example@gmail.com", password="password")
        mock_smtp.return_value.quit.assert_called_once()

    @patch("anomalytics.notifications.email.SMTP")
    def test_send_webde(self, mock_smtp):
        mock_smtp.return_value = MagicMock()

        self.webde_notification.setup(data=self.test_data, message=self.test_message)  # type: ignore
        self.webde_notification.send

        mock_smtp.assert_called_with(host="smtp.web.de", port=587)
        mock_smtp.return_value.starttls.assert_called_once()
        mock_smtp.return_value.login.assert_called_once_with(user="example@web.de", password="password")
        mock_smtp.return_value.quit.assert_called_once()

    def tearDown(self) -> None:
        return super().tearDown()
