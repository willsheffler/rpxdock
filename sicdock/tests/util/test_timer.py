import time, numpy, pytest
from sicdock import Timer

def test_timer():
   with Timer() as timer:
      time.sleep(0.01)
      timer.checkpoint('foo')
      time.sleep(0.03)
      timer.checkpoint('bar')
      time.sleep(0.02)
      timer.checkpoint('baz')

   times = timer.report_dict()
   assert numpy.allclose(times['foo'], 0.01, atol=0.002)
   assert numpy.allclose(times['bar'], 0.03, atol=0.002)
   assert numpy.allclose(times['baz'], 0.02, atol=0.002)

   times = timer.report_dict(order='longest')
   assert list(times.keys()) == ['total', 'bar', 'baz', 'foo', 'none']

   times = timer.report_dict(order='callorder')
   assert list(times.keys()) == ['foo', 'bar', 'baz', 'none', 'total']

   with pytest.raises(ValueError):
      timer.report_dict(order='oarenstoiaen')

if __name__ == '__main__':
   test_timer()
