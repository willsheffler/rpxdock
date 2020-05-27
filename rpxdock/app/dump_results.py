import _pickle, rpxdock as rp, numpy as np
from tabulate import tabulate

def main():
   kw = rp.options.get_cli_args()

   for fn in kw.inputs:
      with open(fn, 'rb') as inp:
         result = _pickle.load(inp)

         result.dump_pdbs_top_score(**kw)

         result.data = result.drop('xforms')
         df = result.data.to_dataframe()
         # print(result.to_dataframe().to_fwf())

         for i in range(len(result.bodies[0])):
            compnames = np.array([bs[i].pdbfile for bs in result.bodies])
            df['comp%i' % (i + 1)] = compnames[df['ijob']]
         df['rank'] = np.arange(len(result), dtype='i')

         content = tabulate(df.values.tolist(), list(df.columns), tablefmt="plain")
         print(content)

if __name__ == '__main__':
   main()