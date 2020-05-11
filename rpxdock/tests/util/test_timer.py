import time, numpy, pytest
from rpxdock import Timer

def test_timer():
   with Timer() as timer:
      time.sleep(0.02)
      timer.checkpoint('foo')
      time.sleep(0.06)
      timer.checkpoint('bar')
      time.sleep(0.04)
      timer.checkpoint('baz')

   times = timer.report_dict()
   assert numpy.allclose(times['foo'], 0.02, atol=0.05)
   assert numpy.allclose(times['bar'], 0.06, atol=0.05)
   assert numpy.allclose(times['baz'], 0.04, atol=0.05)

   times = timer.report_dict(order='longest')
   assert list(times.keys()) == ['total', 'bar', 'baz', 'foo']

   times = timer.report_dict(order='callorder')
   assert list(times.keys()) == ['foo', 'bar', 'baz', 'total']

   with pytest.raises(ValueError):
      timer.report_dict(order='oarenstoiaen')

def test_summary():
   with Timer() as timer:
      time.sleep(0.01)
      timer.checkpoint('foo')
      time.sleep(0.03)
      timer.checkpoint('foo')
      time.sleep(0.02)
      timer.checkpoint('foo')

   times = timer.report_dict(summary=sum)
   assert numpy.allclose(times['foo'], 0.06, atol=0.02)

   times = timer.report_dict(summary=numpy.mean)
   assert numpy.allclose(times['foo'], 0.02, atol=0.01)

   times = timer.report_dict(summary='mean')
   assert numpy.allclose(times['foo'], 0.02, atol=0.01)

   times = timer.report_dict(summary='min')
   assert numpy.allclose(times['foo'], 0.01, atol=0.01)

   with pytest.raises(ValueError):
      timer.report(summary='foo')

   with pytest.raises(ValueError):
      timer.report(summary=1)

if __name__ == '__main__':
   test_timer()
   test_summary()
