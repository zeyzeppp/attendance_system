import unittest
import face_recognition
from main import findEncodings, markAttendance
from unittest.mock import MagicMock
import numpy as np
import os

#   Unit Test
class TestFaceRecognition(unittest.TestCase):
    def test_imageEncodings(self):
        mock_encoding = np.random.rand(128) # Generate a random encoding of length 128
        face_recognition.face_encodings = MagicMock(return_value=[mock_encoding])

        images = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
        encodings = findEncodings(images)

        self.assertEqual(len(encodings), 3)
        for encoding in encodings:
            self.assertEqual(len(encoding), 128)

    def test_markAttendance(self):
        def test_mark_attendance(self):
            tmp_file = 'test_attendance.csv'
            with open(tmp_file, 'w') as f:
                f.write('Alp Arselan, 01/01/2024 12:00:00')

            markAttendance('Mehmet Mustafa', tmp_file)
            with open(tmp_file, 'r') as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 2)
                self.assertIn('Mehmet Mustafa', lines[1])

            markAttendance('Alp Arselan', tmp_file)
            with open(tmp_file, 'r') as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 2)
            os.remove(tmp_file)


if __name__ == '__main__':
    unittest.main()