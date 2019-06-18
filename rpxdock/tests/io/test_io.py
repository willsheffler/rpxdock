from rpxdock.io import aname_to_elem

def test_aname_to_elem():
   anames = [
      ' OG ', '1HH1', ' CA ', ' HE2', '1HD ', '1HG ', ' OD1', '1HH2', ' NH2', ' H  ', '1HG1',
      '1HD1', ' HG ', ' HG1', ' HE1', '2HE ', '1HZ ', ' CG1', ' CD1', '2HG1', ' CE1', ' HZ ',
      ' C  ', ' OG1', '1HE2', '2HZ ', ' CD2', '3HB ', ' N  ', '2HG2', '2HD2', ' NH1', ' NE ',
      '1HG2', '2HG ', ' CZ ', ' OE1', '1HD2', '2HE2', '3HD2', '2HD1', '1HB ', '2HB ', ' HA ',
      '2H  ', ' NE2', '1HE ', ' SD ', ' HH ', ' NZ ', ' OXT', ' ND2', ' HD2', ' OE2', '3HG1',
      ' HD1', ' HB ', ' OH ', '1H  ', '2HH2', ' O  ', ' CG2', ' OD2', ' HE ', '3HE ', '2HH1',
      ' CE2', '3HG2', '1HA ', ' CG ', '3H  ', '3HZ ', ' CE ', ' CB ', ' CD ', '2HA ', '2HD ',
      '3HD1'
   ]

if __name__ == '__main__':
   test_aname_to_elem()
