import ControlFalconMainWindow
import unittest

class TestMyCode(unittest.TestCase):
    def setUp(self):
        print('test start')

    def test_add(self):
        MyMain()
        self.assertEqual(j.add(), 5, 'error！')

    def tearDown(self):
        print('test end')


if __name__ == '__main__':
     suite = unittest.TestSuite()
     suite.addTest(TestMyCode('test_add'))

     runner = unittest.TextTestRunner()
     runner.run(suite)
     print('Finish!')
