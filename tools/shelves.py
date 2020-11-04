import numpy as np
from math import log
from fractions import Fraction

# window sill 26
# depth to window sill 27.75
# width to stonework 123
# width of molding 0.5 x 2
# depth of molding 0.5

np.set_printoptions(formatter=dict(float=lambda x: '%7.3f' % x))

def ladder():
   heights = np.array([26.25, 48.75, 68.25, 85, 99.25])
   print('heights', heights)
   step0 = heights[0]
   step1 = heights[1]
   start = 2.75
   ceil = 108.3
   ceilclear = 0.3  #0.25
   floorclear = 0.25

   tiltfactor = 1 / np.cos(np.radians(12))
   print('ladder diag12 height', imperial(108 * tiltfactor - floorclear))

   # for nstep in (6, 7, 8, 9, 10, 11, 12, 13):
   for nstep in (7, ):
      print(f'{nstep:*^80}')

      rise = (ceil - ceilclear + 1 - start) / (nstep + 2)
      print('rise', imperial(rise))

      for i in range(1, nstep + 2):
         tiltfactor = 1 / np.cos(np.radians(12))
         Hladder = start + i * rise + floorclear
         cenh = Hladder - 0.5
         tiltcenh = tiltfactor * cenh
         print(
            f'{i} {tiltcenh:7.3f} {imperial(tiltcenh)} {tiltcenh/tiltfactor-0.5+floorclear:7.3f}  {Hladder:7.3f}'
         )

      wilheight = 5 * 12 + 8.25
      camheight = 4 * 12 + 10.25
      wb, cb = 99, 99
      for i in range(1, nstep + 2):
         err = ''
         h = start + i * rise
         if heights[0] - 11 < h < heights[0]: err = 'isect step 0 %7.3f' % (heights[0] - h - 7)

         if heights[1] - 7 < h < heights[1]: err = 'isect step 1 %7.3f' % (heights[1] - h - 3.5)
         if heights[3] - 7 < h < heights[3] - 4: err = 'isect carriage 3'
         if ceil - h - wilheight > 0: wb = min(wb, ceil - h - wilheight)
         if ceil - h - camheight > 0: cb = min(cb, ceil - h - camheight)
         print(f'step {i:2} toph {h:7.3f}  {imperial(h)}   cenh {imperial(h-0.5)}' \
            # f' {max(0,ceil-h-wilheight):7.3f}w' \
            # f' {max(0,ceil-h-camheight):7.3f}c' \
            f'  {err}')
      print(f' wb {wb:7.3f} cb {cb:7.3f}')

def diagonal_side():

   vert_angle = 12  # get extra 4 degrees from step taper np.tan(np.radians(4)) * 36
   angle = np.radians(90 - vert_angle)

   # print((1.5 - 1 / 32) / np.sin(angle))
   # returnct

   height = 108
   lip = 0.25
   height0 = 26.0 - 1.75 + lip
   height1 = 10.55
   width1 = 120
   depth_at_sill = 27.75 - 1.25 - 1.25 - 0.5 + 1.75 / np.tan(angle)  # include slope

   height0 += 1.75
   height1 += 1.75

   print('rise/run 3.5', 3.5 / 2 / np.tan(angle))

   width2 = width1 - depth_at_sill
   toth = height + 3.5
   d0 = depth_at_sill + height0 / np.tan(angle)
   d1 = d0 - height / np.tan(angle)
   print(f'depth at bottom {d0:5.2f} top {d1:5.2f}')

   assert np.allclose(np.pi / 2 - angle, np.arctan((d0 - d1) / height))

   shelf_heights = np.logspace(np.log(height0), np.log(height1), 6, base=np.e)
   if not np.allclose(np.sum(shelf_heights) - 3.5, height, atol=0.1):
      print('wrong total height', np.sum(shelf_heights) - 3.5, 'should be 108')
      assert 0
   total_heights = np.cumsum(shelf_heights)

   h = np.cumsum(shelf_heights)
   shelf_depths = h / height * (d1 - d0) + d0
   # print(np.round(h - 1.75, 2))
   # verify shelves make corrent angle
   for i in range(6):
      for j in range(i):
         tan = (total_heights[i] - total_heights[j]) / (shelf_depths[j] - shelf_depths[i])
         assert np.allclose(np.degrees(np.arctan(tan)), 90 - vert_angle)

   print()
   print('bottoms......', total_heights - 1.75 - 1.75)
   print('centers......', total_heights - 1.75 + 0.00)
   print('shelftop.....', total_heights - 1.75 + 1.75 - lip)
   print('tops.........', total_heights - 1.75 + 1.75)

   print()
   gaps = shelf_heights - 3.5
   assert np.allclose(np.sum(gaps) + 5 * 3.5, 108, atol=0.1)
   print('cencen height', shelf_heights)
   print('shelf ingress', gaps)  #gaps[1:], f'floor: {gaps[0]:7.3f}')

   print()
   print('depths<->wall', shelf_depths[:-1] + 1.25 + 1.25 + 0.5)
   print('shelf depths.', shelf_depths[:-1] - 2 * 0.75)
   print('lip cen depth', shelf_depths[:-1])
   shelf_offsets = shelf_depths[:-1] - shelf_depths[1:]
   print('shelf offsets', shelf_offsets)
   print()
   print(f'total plywood width {np.sum(shelf_depths[:-1] - 2 * 0.75):7.3f}')

   side1 = shelf_depths - 2.25 * 2 + 0 / np.tan(angle)  # sub cen to post edge

   if True:  # right side (window)
      print(f'{"SIDE A along window":*^80}')
      print_fracional(
         side1[:-1],
         f'sideA bottom %i vertical angle: 0 / {vert_angle:2d} length:',
      )
      print_fracional(total_heights[:-1] - 1.75, f'postA %i vert cen height:')
      print('post B height %7.3f' % (108 / np.sin(angle)), 110,
            MyFrac(int(32 * (108 / np.sin(angle) - 110)), 32))
      print_fracional((total_heights[:-1] - 1.75) / np.sin(angle), f'postB %i diag cen height:')

   L = [26, 21 + 3 / 16, 17 + 11 / 32, 14 - 1 / 16, 11 + 5 / 64]

   if True:  # left side
      print(f'{"SIDE R fireplace":*^80}')
      # right_back_offset = 7 / 8  # relative to post back
      # right_front_offset = 3.5
      # left_back_offset = 13 / 16 * np.cos(np.radians(22.5))
      # left_front_offset = (2 + 9 / 16) * np.cos(np.radians(22.5))
      # print('right offsets', imperial(right_back_offset), imperial(right_front_offset))
      # print('left offsets ', imperial(left_back_offset), imperial(left_front_offset))
      prev = None
      for i, d in enumerate(shelf_depths[:-1]):
         if L[i] is 0: continue
         print('----------------')
         alpha = np.arctan(d / shelf_depths[0])
         diaglen = shelf_depths[0] / np.cos(alpha)
         print('diaglen     ', diaglen)

         diagh2by = 0.75 / np.cos(alpha)
         diagh2byback = 0.75 / np.cos(alpha - np.pi / 8)
         diaghp12 = 1.75 * np.tan(np.radians(vert_angle))

         front_left_left_top_offset = diagh2by + 0.25
         front_left_left_offset = front_left_left_top_offset + diaghp12
         front_left_right_top_offset = .25 + .75
         front_left_right_offset = front_left_right_top_offset + diaghp12
         fllo_domino_offset = 1.1 / 2 * np.tan(alpha)
         print('alpha       ', np.degrees(alpha))
         print('alpha-22.5  ', np.degrees(alpha) - 22.5)
         print('fllo_domino_offset', imperial(fllo_domino_offset))
         D = L[i] + fllo_domino_offset - 3 / 8 - 2.375
         diag = D / np.sin(alpha)
         print(diag * np.cos(alpha), diag * np.sin(alpha))
         print('depth        ', D)
         print('H to bottom  ', imperial(D + 4.25 - 2.5 * np.sin(np.radians(22.5))))
         print('diag         ', imperial(diag))

         H = 27 + 3 / 32 - 4.25 + 2.5 * np.sin(np.radians(22.5))
         print('D', imperial(D))
         print('H', imperial(H))
         print(
            'A',
            np.degrees(np.arctan(D / H)),
            90 - np.degrees(np.arctan(D / H)),
            'degrees',
         )
         print(
            'B',
            np.degrees(np.arctan(D / H)) - 22.5,
            90 - np.degrees(np.arctan(D / H)) + 22.5,
            'degrees',
         )
         print('X', imperial(np.sqrt(D * D + H * H)))

         front_left_burial = np.sqrt(2) * (front_left_left_offset - front_left_right_offset)
         lawofsines = (1 + 11 / 16) / np.sin(alpha)
         left_back_front_bur = lawofsines * np.sin(np.radians(67.5))
         left_back_back_bur = lawofsines * np.sin(np.radians(112.5) - alpha)

         left_2x_len_cen = diaglen - front_left_burial - left_back_front_bur

         left_front_cen_to_front = np.tan(alpha) * (0.75 - 1 / 32)
         left_back_cen_to_front = np.tan(alpha - np.pi / 8) * (0.75 - 1 / 32)

         # current = shelf_depths[i] + front_left_right_offset - front_left_left_offset
         # if prev:
         # print('!!!!!', current, prev - current, shelf_offsets[i - 1])
         # prev = current

         o1 = front_left_right_offset
         o2 = front_left_left_offset
         delta = 1.7  # ???????

         continue

         d0 = shelf_depths[0]
         tmp = d0 * np.tan(alpha)
         diag = d / np.sin(alpha)
         FLLBur = (o2 - o1) / np.sin(alpha)
         BLFBur = np.sin(3 / 8 * np.pi) / np.sin(alpha) * delta
         BLBBur = np.sin(5 / 8 * np.pi - alpha) / np.sin(alpha) * delta
         X = diag - FLLBur - BLFBur
         assert np.allclose(alpha, np.arctan(d / d0))
         assert np.allclose(d, tmp)

         print('###############################3')
         print(f'{"alpha":>12} {np.degrees(alpha):7.3f}')
         print(f'{"alpha-22.5":>12} {np.degrees(alpha)-22.5:7.3f}')
         print(f'{"d0":>12} {d0:7.3f}', shelf_depths[0])
         print(f'{"d":>12} {d:7.3f}', shelf_depths[i])
         print(f'{"diag":>12} {diag:7.3f}', diaglen)
         print(f'{"FLLBur":>12} {FLLBur:7.3f}', front_left_burial)
         print(f'{"BLFBur":>12} {BLFBur:7.3f}', left_back_front_bur)
         print(f'{"BLBBur":>12} {BLBBur:7.3f}', left_back_back_bur)
         print(f'{"depth_delta":>12} {shelf_depths[i] - shelf_depths[i + 1]:7.3f}')
         print(f'{"X":>12} {X:7.3f}', left_2x_len_cen - X)
         print('###############################3')

         if 0:
            print('#########################################')
            print(f'{"alpha":>27} {alpha:7.3f}')
            print(f'{"diaglen":>27} {diaglen:7.3f}')
            print(f'{"diagh2by":>27} {diagh2by:7.3f}')
            print(f'{"diagh2byback":>27} {diagh2byback:7.3f}')
            print(f'{"diaghp12":>27} {diaghp12:7.3f}')
            print(f'{"front_left_left_top_offset":>27} {front_left_left_top_offset:7.3f}')
            print(f'{"front_left_left_offset":>27} {front_left_left_offset:7.3f}')
            print(f'{"front_left_right_top_offset":>27} {front_left_right_top_offset:7.3f}')
            print(f'{"front_left_right_offset":>27} {front_left_right_offset:7.3f}')
            print(f'{"fllo_domino_offset":>27} {fllo_domino_offset:7.3f}')
            print(f'{"front_left_burial":>27} {front_left_burial:7.3f}')
            print(f'{"lawofsines":>27} {lawofsines:7.3f}')
            print(f'{"left_back_front_bur":>27} {left_back_front_bur:7.3f}')
            print(f'{"left_back_back_bur":>27} {left_back_back_bur:7.3f}')
            print(f'{"left_2x_len_cen":>27} {left_2x_len_cen:7.3f}')
            print(f'{"left_front_cen_to_front":>27} {left_front_cen_to_front:7.3f}')
            print(f'{"left_back_cen_to_front":>27} {left_back_cen_to_front:7.3f}')
            print('#########################################')

         # print(left_front_cen_to_front)
         # print(left_back_cen_to_front)
         print('Angle Front %5.2f' % np.round(np.degrees(alpha), 1))
         print('Angle Back  %5.2f' % np.round(np.degrees(alpha - np.pi / 8), 2))
         print('Len Front   %s' %
               imperial(left_2x_len_cen + left_front_cen_to_front - left_back_cen_to_front))
         print('Len Center  %s' % imperial(left_2x_len_cen))
         print('Len Back    %s' %
               imperial(left_2x_len_cen - left_front_cen_to_front + left_back_cen_to_front))
         print('Len cut     %s' %
               np.round(left_2x_len_cen + left_front_cen_to_front + left_back_cen_to_front + 1.5))
         print('FLL Domino height obtuse +????', mm(diagh2by * 15 / 16 - fllo_domino_offset)[1:])
         print('BLF Domino height obtuse +????', mm(diagh2byback * 15 / 16)[1:])

         dummy(
            'H',
            imperial((total_heights[i] - 1.75) / np.sin(angle)),
            #
            'AF %5.2f' % np.round(np.degrees(alpha), 1),
            'AB %5.2f' % np.round(np.degrees(alpha - np.pi / 8), 2),
            'Lf %s' % imperial(left_2x_len_cen),
            'Lc %s' % imperial(left_2x_len_cen),
            'Lb %s' % imperial(left_2x_len_cen),

            # 'cFLLO',
            # imperial(front_left_left_offset)[1:],
            'o2xofst',
            imperial(diagh2by - fllo_domino_offset)[1:],
            # 'dFLLO',
            # imperial(front_left_left_offset + fllo_domino_offset)[1:],
            # 'dfence',
            # imperial(front_left_left_offset + fllo_domino_offset - 0.385)[1:],
         )
         assert np.allclose(front_left_left_offset - diaghp12 - diagh2by - 1 / 4, 0)
         assert np.allclose(
            1 / 4 + diagh2by + fllo_domino_offset + diaghp12,
            front_left_left_offset + fllo_domino_offset,
         )

def print_fracional(n, label):
   frac = n % 1.0
   whole = n.astype('i').astype('f')
   # print(whole)
   frac = np.round(frac * 32)
   print(f'{f" {label[:5]} ":=^80}')
   for i, w in enumerate(whole):
      print(label % i, '...', int(w), MyFrac(int(frac[i]), 32))
   # print(frac)

dummy = lambda *a: a

class MyFrac(Fraction):
   def __repr__(self):
      frac = (self.numerator / self.denominator) % 1.0
      whole = int(self.numerator / self.denominator)
      if whole == 0:
         return f'{self.numerator%self.denominator:2d}/{self.denominator:2d}'
      elif self.numerator % self.denominator == 0:
         return f'{whole:2d}'
      else:
         return f'{whole:2d}-{self.numerator%self.denominator:2d}/{self.denominator:2d}'

   def __str__(self):
      return repr(self)

def mm(n):
   return '%5.1fmm' % (np.round(25.4 * n, 1))

def imperial(n):
   # print(whole)
   return str(MyFrac(int(np.round(32 * n)), 32))

if __name__ == '__main__':
   # diagonal_side()
   ladder()