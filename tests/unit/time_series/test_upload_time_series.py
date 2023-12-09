import os
import unittest

import pandas as pd

from anomalytics import read_ts
from anomalytics.time_series.upload import create_ts_from_csv, create_ts_from_xlsx


class TestTimeSeriesReaders(unittest.TestCase):
    csv_file: str
    xlsx_file: str

    @classmethod
    def setUpClass(cls):
        cls.csv_file = "test_data.csv"
        cls.xlsx_file = "test_data.xlsx"
        test_data = pd.Series(range(10), index=pd.date_range("2023-01-01", periods=10))
        test_data.to_csv(cls.csv_file, header=False)
        test_data.to_excel(cls.xlsx_file, header=False)

    def test_read_csv(self):
        ts = create_ts_from_csv(path_to_file=self.csv_file, header=None)
        self.assertIsInstance(ts, pd.Series)
        self.assertEqual(len(ts), 10)

    def test_read_xlsx(self):
        ts = create_ts_from_xlsx(path_to_file=self.xlsx_file, index_col=0)
        self.assertIsInstance(ts, pd.Series)
        self.assertEqual(len(ts), 10)

    def test_read_ts_csv(self):
        ts = read_ts(path_to_file=self.csv_file, file_type="csv", header=None)
        self.assertIsInstance(ts, pd.Series)
        self.assertEqual(len(ts), 10)

    def test_read_ts_xlsx(self):
        ts = read_ts(path_to_file=self.xlsx_file, file_type="xlsx", index_col=0)
        self.assertIsInstance(ts, pd.Series)
        self.assertEqual(len(ts), 10)

    def test_invalid_file_type(self):
        with self.assertRaises(ValueError):
            read_ts(path_to_file="test_data.txt", file_type="txt")  # type: ignore

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            read_ts(path_to_file="non_existent_file.csv", file_type="csv")

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.csv_file)
        os.remove(cls.xlsx_file)
